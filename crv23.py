#!/usr/bin/env python3
"""CRv23 - ask questions about a codebase; get answers that cite file:line.

Read-only: the model can list, grep and read files inside the repo and nothing else. No shell, no edits.
Talks to any OpenAI-compatible server (llama.cpp, vLLM, ...). Python 3 stdlib only, nothing to install.

usage:
  python3 crv23.py REPO "How does a request get routed?"
  python3 crv23.py REPO --question-file q.txt --out run1/      # keep the transcript
  python3 crv23.py REPO "..." --url http://HOST:PORT/v1 --model MODEL_ID
  (or set CRV23_URL / CRV23_MODEL; OPENAI_API_KEY is sent as a bearer token if set)

Defaults are the best settings we measured with Qwen3.8 Flash Next (thinking on, 64k context):
  - a code map (file tree, symbols with line numbers, imports, entry points) is built from REPO and given
    to the model up front, so it knows where to look before it reads anything
  - the model stops when it replies without a tool call; that reply is the answer (no step counter)
  - reads return up to 2000 lines; tool results are capped at 50 KB
  - the model keeps terse path:line notes; file:line facts it writes only in its thinking are copied into
    its visible reply, since thinking is not kept between turns
  - when the context passes 65%, old tool output is dropped oldest-first by use: superseded re-reads and
    grep/list output go first, and a read keeps only the lines the notes cite
  - if it answers with 10 or more steps left, it is asked once to check its notes for unread code or
    unconfirmed risks, then either keeps exploring or rewrites the complete answer
  - 60 steps / 45 min are a safety net only; if hit, the model is asked for its answer from what it has read
Sampling is set for Qwen (T1.0, top_p 0.95, top_k 20, effort medium). For another model, edit THINK / NOTHINK.
The server must support tool calls (llama.cpp: start llama-server with --jinja). Set --ctx to the server's real
per-request context; the 65% trim depends on it.
Measured on a 12-fact answer key (Qwen3.8 Flash Next, DGX Spark): 11/12 facts on both runs, 99% of citations valid.
"""
import argparse, ast, json, os, re, subprocess, sys, time, urllib.request

SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'dist', 'build', '.venv'}
CODE_EXT = {'.py', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.sh', '.bash'}
TS_EXT = ('.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs')

# ------------------------------------------------------------------ code map


def walk(root):
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            p = os.path.join(d, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            out.append(os.path.relpath(p, root))
    return sorted(out)


def read(root, rel):
    try:
        with open(os.path.join(root, rel), 'rb') as fh:
            b = fh.read()
    except OSError:
        return None
    if b'\0' in b[:4096]:
        return None
    return b.decode('utf-8', 'replace')


def nlines(txt):
    return 0 if txt is None else len(txt.splitlines())


def is_shell(rel, txt):
    if rel.endswith(('.sh', '.bash')):
        return True
    return bool(txt) and re.match(r'#!.*\b(ba|z|da)?sh\b', txt.split('\n', 1)[0] or '') is not None


# ---------------------------------------------------------------- python
def py_symbols(txt):
    """-> (symbols [(line, kind, name, [methods])], imports [(level, module, [names])], entries [(line, what)])"""
    try:
        tree = ast.parse(txt)
    except SyntaxError:
        return None
    syms, imps, ents = [], [], []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            syms.append((n.lineno, 'async def' if isinstance(n, ast.AsyncFunctionDef) else 'def', n.name, []))
            if n.name == 'main':
                ents.append((n.lineno, 'def main'))
        elif isinstance(n, ast.ClassDef):
            meths = [(m.lineno, m.name) for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            syms.append((n.lineno, 'class', n.name, meths))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tg:
                if isinstance(t, ast.Name) and re.fullmatch(r'_?[A-Z][A-Z0-9_]*', t.id):
                    syms.append((n.lineno, 'const', t.id, []))
        elif isinstance(n, ast.If):
            src = ast.dump(n.test)
            if "'__name__'" in src and "'__main__'" in src:
                ents.append((n.lineno, 'if __name__ == "__main__"'))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imps.append((0, a.name, []))
        elif isinstance(n, ast.ImportFrom):
            imps.append((n.level or 0, n.module or '', [a.name for a in n.names]))
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else '')
            if name in ('ArgumentParser',):
                ents.append((n.lineno, 'CLI parser (argparse.ArgumentParser)'))
            elif name in ('run_app',):
                ents.append((n.lineno, 'server start (web.run_app)'))
            elif name in ('serve_forever', 'uvicorn_run') or (name == 'run' and isinstance(f, ast.Attribute)
                                                               and isinstance(f.value, ast.Name) and f.value.id == 'uvicorn'):
                ents.append((n.lineno, 'server start (%s)' % name))
            elif name == 'run' and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'asyncio':
                ents.append((n.lineno, 'asyncio.run'))
    return syms, imps, sorted(set(ents))


def py_resolve(root, rel, level, module, names, files):
    """Resolve a python import to internal files (relative or absolute, against every ancestor dir)."""
    fs = set(files)
    out = []

    def cand(base, dotted):
        parts = [p for p in dotted.split('.') if p]
        p = os.path.normpath(os.path.join(base, *parts)) if parts else os.path.normpath(base)
        for c in (p + '.py', os.path.join(p, '__init__.py')):
            c = os.path.normpath(c)
            if c in fs:
                return c
        return None

    if level:
        base = os.path.dirname(rel)
        for _ in range(level - 1):
            base = os.path.dirname(base)
        bases = [base]
    else:
        bases, d = [], os.path.dirname(rel)
        while True:
            bases.append(d)
            if not d:
                break
            d = os.path.dirname(d)
    for b in bases:
        hit = False
        for nm in names:  # from pkg import submodule
            c = cand(b, (module + '.' + nm) if module else nm)
            if c and nm != '*':
                out.append(c); hit = True
        if module:
            c = cand(b, module)
            if c:
                out.append(c); hit = True
        if hit:
            break
    return [o for o in out if o != rel]


# ---------------------------------------------------------------- ts / js
TS_DECL = [
    (re.compile(r'^export\s+default\s+(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)?'), 'export default function'),
    (re.compile(r'^export\s+(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)'), 'export function'),
    (re.compile(r'^export\s+(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)'), 'export class'),
    (re.compile(r'^export\s+(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)'), 'export interface'),
    (re.compile(r'^export\s+(?:declare\s+)?type\s+([A-Za-z_$][\w$]*)'), 'export type'),
    (re.compile(r'^export\s+(?:const\s+|declare\s+)?enum\s+([A-Za-z_$][\w$]*)'), 'export enum'),
    (re.compile(r'^export\s+(?:declare\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)'), 'export const'),
    (re.compile(r'^(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)'), 'function'),
    (re.compile(r'^(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)'), 'class'),
    (re.compile(r'^interface\s+([A-Za-z_$][\w$]*)'), 'interface'),
    (re.compile(r'^type\s+([A-Za-z_$][\w$]*)\s*[=<]'), 'type'),
    (re.compile(r'^(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)'), 'enum'),
    (re.compile(r'^(?:const|let|var)\s+([A-Za-z_$][\w$]*)'), 'const'),
]
TS_REEXPORT = re.compile(r'^export\s+(?:type\s+)?\{([^}]*)\}')
TS_DEFAULT_ID = re.compile(r'^export\s+default\s+([A-Za-z_$][\w$]*)\s*;?\s*$')
TS_IMPORT = re.compile(r'''(?:^|\s)(?:import|export)\s[^'"]*?\sfrom\s+['"]([^'"]+)['"]|^import\s+['"]([^'"]+)['"]|\brequire\(\s*['"]([^'"]+)['"]\s*\)|\bimport\(\s*['"]([^'"]+)['"]\s*\)|\bnew\s+Worker\(\s*new\s+URL\(\s*['"]([^'"]+)['"]''')
TS_ENTRY = [
    (re.compile(r'\b(?:ReactDOM\.)?createRoot\s*\('), 'React root mount (createRoot)'),
    (re.compile(r'\bReactDOM\.render\s*\('), 'React root mount (ReactDOM.render)'),
    (re.compile(r'\.listen\s*\(\s*[\w.$]'), 'server listen()'),
    (re.compile(r'\bnew\s+WebSocket\s*\('), 'websocket client opened (new WebSocket)'),
    (re.compile(r'\bprocess\.argv\b'), 'CLI (process.argv)'),
    (re.compile(r'\bimport\.meta\.main\b|\brequire\.main\s*===\s*module\b'), 'main guard'),
    (re.compile(r'\bcreateBrowserRouter\s*\('), 'router setup (createBrowserRouter)'),
]


def strip_ts_comments(txt):
    """Blank out comments and string bodies line-preservingly, so declarations are matched only in code."""
    out, i, n = [], 0, len(txt)
    while i < n:
        c = txt[i]
        if txt.startswith('//', i):
            j = txt.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j
        elif txt.startswith('/*', i):
            j = txt.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r'[^\n]', ' ', txt[i:j])); i = j
        else:
            out.append(c); i += 1
    return ''.join(out)


def ts_symbols(txt):
    code = strip_ts_comments(txt)
    syms, imps, ents = [], [], []
    depth = 0
    for ln, line in enumerate(code.split('\n'), 1):
        s = line.strip()
        if depth == 0 and s:
            m = TS_REEXPORT.match(s)
            if m:
                for nm in m.group(1).split(','):
                    nm = nm.strip().split(' as ')[-1].strip()
                    if re.fullmatch(r'[A-Za-z_$][\w$]*', nm):
                        syms.append((ln, 'export', nm, []))
            else:
                m = TS_DEFAULT_ID.match(s)
                if m:
                    syms.append((ln, 'export default', m.group(1), []))
                else:
                    for rx, kind in TS_DECL:
                        m = rx.match(s)
                        if m:
                            syms.append((ln, kind, m.group(1) or '(anonymous)', []))
                            break
        for m in TS_IMPORT.finditer(line):
            imps.append(next(g for g in m.groups() if g))
        for rx, what in TS_ENTRY:
            if rx.search(line):
                ents.append((ln, what))
        depth += line.count('{') - line.count('}')
        if depth < 0:
            depth = 0
    # multi-line imports: `import {\n a,\n b\n} from './x'` -> the closing line carries `from`
    for m in re.finditer(r'''\}\s*from\s+['"]([^'"]+)['"]''', code):
        imps.append(m.group(1))
    return syms, sorted(set(imps)), sorted(set(ents))


def ts_resolve(rel, spec, files):
    if not spec.startswith('.'):
        return None
    fs = set(files)
    base = os.path.normpath(os.path.join(os.path.dirname(rel), spec))
    stem = re.sub(r'\.(js|jsx|mjs|cjs)$', '', base)
    cands = [base]
    for b in (base, stem):
        cands += [b + e for e in ('.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.d.ts')]
        cands += [os.path.join(b, 'index' + e) for e in ('.ts', '.tsx', '.js', '.jsx')]
    for c in cands:
        if c in fs and c != rel:
            return c
    return None


# ---------------------------------------------------------------- shell
SH_FUNC = re.compile(r'^\s*(?:function\s+)?([A-Za-z_][\w:.-]*)\s*\(\s*\)\s*\{?|^\s*function\s+([A-Za-z_][\w:.-]*)\s*\{?')
SH_CASE_ARG = re.compile(r'^\s*case\s+"?\$\{?(?:1|@|\*|arg|ARG|a|cmd|CMD|action|ACTION|mode|MODE)\b')
SH_LOOP_ARGS = re.compile(r'^\s*(?:for\s+\w+\s+in\s+"?\$@|while\s+\[\[?\s+\$#\s+-gt\s+0)')


def sh_symbols(txt, rel, files):
    syms, refs, ents = [], set(), []
    others = [f for f in files if f != rel]
    for ln, line in enumerate(txt.split('\n'), 1):
        s = line.strip()
        if s.startswith('#') and not (ln == 1 and s.startswith('#!')):
            continue
        m = SH_FUNC.match(line)
        if m and not s.startswith(('if ', 'while ', 'for ', 'elif ')):
            nm = m.group(1) or m.group(2)
            if nm and nm not in ('if', 'then', 'else', 'fi', 'do', 'done'):
                syms.append((ln, 'fn', nm, []))
        if SH_CASE_ARG.match(line):
            ents.append((ln, 'argument dispatch (case on script args)'))
        if SH_LOOP_ARGS.match(line):
            ents.append((ln, 'argument loop'))
        code = line.split(' #', 1)[0]
        for o in others:
            # a reference = the file's relative path (or basename when it sits in the same dir) appears in the code text
            if o in code or (os.path.dirname(o) == os.path.dirname(rel) and re.search(r'(^|[\s/"\'$}])' + re.escape(os.path.basename(o)) + r'($|[\s"\';)])', code)):
                refs.add(o)
    if txt.startswith('#!'):
        ents.insert(0, (1, 'executable script (shebang)'))
    return syms, sorted(refs), sorted(set(ents))


def generic_refs(txt, rel, files):
    """Python files that run other target files by path (subprocess etc.): path literal of another target file."""
    refs = set()
    for o in files:
        if o != rel and o.count('/') >= 1 and o in txt:
            refs.add(o)
    return refs


def yaml_symbols(txt):
    syms = []
    sec = None
    for ln, line in enumerate(txt.split('\n'), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        m = re.match(r'^([A-Za-z_][\w.-]*):', line)
        if m:
            sec = m.group(1); syms.append((ln, 'key', sec, [])); continue
        m = re.match(r'^  ([A-Za-z_][\w.-]*):\s*$', line)
        if m and sec in ('services', 'jobs', 'volumes', 'networks'):
            syms.append((ln, sec[:-1] if sec.endswith('s') else sec, m.group(1), []))
    return syms


def json_symbols(txt):
    try:
        d = json.loads(txt)
    except Exception:
        return []
    if isinstance(d, dict):
        return [(0, 'key', k, []) for k in list(d)[:40]]
    if isinstance(d, list):
        return [(0, 'array', 'len=%d' % len(d), [])]
    return []


def docker_entries(txt):
    ents = []
    for ln, line in enumerate(txt.split('\n'), 1):
        m = re.match(r'^\s*(CMD|ENTRYPOINT)\b', line)
        if m:
            ents.append((ln, 'container %s' % m.group(1)))
    return ents


# ---------------------------------------------------------------- build
def analyse(root):
    files = walk(root)
    info = {}
    for rel in files:
        txt = read(root, rel)
        rec = {'lines': nlines(txt), 'syms': [], 'deps': set(), 'ents': [], 'note': ''}
        info[rel] = rec
        if txt is None:
            rec['note'] = 'binary'; continue
        ext = os.path.splitext(rel)[1].lower()
        base = os.path.basename(rel)
        if ext == '.py':
            r = py_symbols(txt)
            if r is None:
                rec['note'] = 'python (unparseable)'; continue
            syms, imps, ents = r
            rec['syms'], rec['ents'] = syms, ents
            for lvl, mod, names in imps:
                rec['deps'].update(py_resolve(root, rel, lvl, mod, names, files))
            rec['deps'].update(generic_refs(txt, rel, files))
        elif ext in TS_EXT:
            syms, imps, ents = ts_symbols(txt)
            rec['syms'], rec['ents'] = syms, ents
            for spec in imps:
                r = ts_resolve(rel, spec, files)
                if r:
                    rec['deps'].add(r)
        elif is_shell(rel, txt):
            syms, refs, ents = sh_symbols(txt, rel, files)
            rec['syms'], rec['ents'] = syms, ents
            rec['deps'].update(refs)
        elif ext in ('.yaml', '.yml'):
            rec['syms'] = yaml_symbols(txt)
        elif ext == '.json':
            rec['syms'] = json_symbols(txt)
        elif base == 'Dockerfile' or base.endswith('.Dockerfile'):
            rec['ents'] = docker_entries(txt)
        rec['deps'] = sorted(rec['deps'] - {rel})
    return files, info


def fmt_sym(s, with_methods):
    ln, kind, name, meths = s
    at = '@%d' % ln if ln else ''
    out = '%s %s%s' % (kind, name, at)
    if with_methods and meths:
        out += ' {' + ', '.join('%s@%d' % (m, l) for l, m in meths) + '}'
    return out


KIND_RANK = {'class': 0, 'export class': 0, 'export default function': 0, 'export function': 1, 'def': 1, 'async def': 1,
             'function': 2, 'fn': 1, 'export const': 2, 'export': 2, 'export default': 0, 'export interface': 3,
             'export type': 3, 'export enum': 3, 'const': 4, 'interface': 5, 'type': 5, 'enum': 5}


def render(name, src_label, files, info, with_methods, cap, trim_note):
    total = sum(info[f]['lines'] for f in files)
    o = ['# Code map: %s' % name, '',
         'Structure only, generated mechanically from the source (%s).' % src_label,
         'It lists file names, line counts, symbol names with their line numbers, internal imports and entry points.',
         'It contains no values, string literals, comments or docstrings and makes no claims about behaviour:',
         'read the files to learn what the code does, and cite only lines you have actually read.']
    if trim_note:
        o.append('TRIMMED: ' + trim_note)
    o += ['', '## Files (%d files, %d lines; "N" = line count)' % (len(files), total), '```']
    for f in files:
        n = info[f]['note']
        o.append('%6d  %s%s' % (info[f]['lines'], f, ('  [%s]' % n) if n else ''))
    o += ['```', '', '## Symbols per file (kind name@line; {methods@line} for classes)']
    for f in files:
        syms = info[f]['syms']
        if not syms:
            continue
        shown = syms
        if cap is not None and len(syms) > cap:
            shown = sorted(sorted(syms, key=lambda s: (KIND_RANK.get(s[1], 9), s[0]))[:cap], key=lambda s: s[0])
        o.append('- %s: ' % f + '; '.join(fmt_sym(s, with_methods) for s in shown)
                 + (' (+%d more not shown)' % (len(syms) - len(shown)) if len(shown) < len(syms) else ''))
    o += ['', '## Internal dependency graph (file -> target files it imports, sources or runs by path)']
    any_dep = False
    for f in files:
        if info[f]['deps']:
            any_dep = True
            o.append('- %s -> %s' % (f, ', '.join(info[f]['deps'])))
    if not any_dep:
        o.append('- (no internal imports found)')
    rev = {}
    for f in files:
        for d in info[f]['deps']:
            rev.setdefault(d, []).append(f)
    if rev:
        o += ['', 'Imported by (reverse edges):']
        for d in sorted(rev):
            o.append('- %s <- %s' % (d, ', '.join(sorted(rev[d]))))
    o += ['', '## Entry points (file:line what)']
    any_e = False
    for f in files:
        for ln, what in info[f]['ents']:
            any_e = True
            o.append('- %s:%d %s' % (f, ln, what))
    if not any_e:
        o.append('- (none detected: library-style modules)')
    return '\n'.join(o) + '\n'


def build(name, root, budget):
    files, info = analyse(root)
    src_label = root
    text = render(name, src_label, files, info, True, None, '')
    if len(text.encode()) <= budget:
        return text, ''
    note = 'the full map exceeded the %d-byte budget; class methods are omitted' % budget
    text = render(name, src_label, files, info, False, None, note)
    if len(text.encode()) <= budget:
        return text, note
    for cap in (40, 30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1):
        note = ('the full map exceeded the %d-byte budget; class methods are omitted and at most %d symbols per file are shown '
                '(classes and functions first, then exports and constants); "+N more" marks the cut' % (budget, cap))
        text = render(name, src_label, files, info, False, cap, note)
        if len(text.encode()) <= budget:
            return text, note
    return text, note + ' (STILL OVER BUDGET)'



# ------------------------------------------------------------------ tools
TOOLS = [
    {'type': 'function', 'function': {'name': 'list_dir', 'description': 'List files and subdirectories of a directory (relative path; "." is the root).',
     'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'grep', 'description': 'Search file contents with an extended regular expression (case-insensitive). Returns up to 80 matching lines as path:line:text.',
     'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}, 'path': {'type': 'string', 'description': 'optional file or directory to limit the search'}}, 'required': ['pattern']}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read numbered lines of a file (default the first 2000 lines; at most 2000 per call; use start_line to continue).',
     'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}}, 'required': ['path']}}},
]


def safe(root, p):
    p = str(p or '.').strip()
    if p.startswith(root): p = p[len(root):]
    p = p.lstrip('/') or '.'
    full = os.path.realpath(os.path.join(root, p))
    return full if full == root or full.startswith(root + os.sep) else None


def run_tool(root, name, a):
    if name == 'list_dir':
        f = safe(root, a.get('path'))
        if not f or not os.path.isdir(f): return f'no such directory: {a.get("path")}'
        return '\n'.join(sorted(e + ('/' if os.path.isdir(os.path.join(f, e)) else '') for e in os.listdir(f)))[:6000]
    if name == 'grep':
        f = safe(root, a.get('path'))
        if not f: return 'path outside the repository'
        r = subprocess.run(['grep', '-rnIiE', '--exclude-dir=.git', '--exclude-dir=node_modules', '--', str(a.get('pattern', '')), os.path.relpath(f, root)],
                           cwd=root, capture_output=True, text=True, timeout=60)
        lines = [l[2:] if l.startswith('./') else l for l in r.stdout.splitlines()]
        return ('\n'.join(l[:300] for l in lines[:80]) + (f'\n... {len(lines) - 80} more matches' if len(lines) > 80 else '')) if lines else ('no matches' if r.returncode == 1 else r.stderr[:300] or 'no matches')
    if name == 'read_file':
        f = safe(root, a.get('path'))
        if not f or not os.path.isfile(f): return f'no such file: {a.get("path")}'
        lines = open(f, errors='replace').read().split('\n')
        try: s = max(1, int(a.get('start_line') or 1))
        except Exception: s = 1
        try: e = int(a.get('end_line') or s + 1999)
        except Exception: e = s + 1999
        e = min(len(lines), e, s + 1999)
        rel = os.path.relpath(f, root)
        return f'{rel} ({len(lines)} lines), lines {s}-{e}:\n' + '\n'.join(f'{i}: {lines[i - 1]}' for i in range(s, e + 1))
    return f'unknown tool {name}'


# ------------------------------------------------------------------ model
THINK = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}}
NOTHINK = {'temperature': 0.2, 'top_p': 0.8, 'top_k': 20, 'chat_template_kwargs': {'enable_thinking': False}}


def chat(url, body, timeout):
    u = url.rstrip('/')
    if not u.endswith('/v1'): u += '/v1'
    h = {'Content-Type': 'application/json'}
    if os.environ.get('OPENAI_API_KEY'): h['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    req = urllib.request.Request(u + '/chat/completions', json.dumps(body).encode(), h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ------------------------------------------------------------------ notes and context retention
CITE_FILE = re.compile(r'([\w./-]+\.(?:py|ts|tsx|js|jsx|mjs|sh|json|ya?ml|md|toml|rs|go|c|h|cpp))')
CITE_NUM = re.compile(r'(?:[:@L]|lines? )(\d+)(?:\s*[-–]\s*(\d+))?')
NOTE_LINE = re.compile(r'(\b\d{1,5}\s*[-–]\s*\d{1,5}\b|@\d{1,5}\b|[\w.-]+\.\w{1,4}:\d{1,5}|\blines? \d{1,5}|^\s*[-*]?\s*\d{1,5}\s*:)')
PLAN = re.compile(r'^(now|next|then|continue|let me|let\'s|i will|i\'ll|i need|read|re-read|maybe|also check)\b', re.I)


def cited_lines(texts):
    """{basename: set(line numbers)} from the model's visible notes; numbers in a line belong to the last file named before them."""
    out = {}
    for t in texts:
        for ln in (t or '').split('\n'):
            marks = [(m.start(), 'f', os.path.basename(m.group(1))) for m in CITE_FILE.finditer(ln)]
            marks += [(m.start(), 'n', (int(m.group(1)), int(m.group(2) or m.group(1)))) for m in CITE_NUM.finditer(ln)]
            cur = None
            for _, k, v in sorted(marks, key=lambda x: x[0]):
                if k == 'f': cur = v
                elif cur:
                    a0, b0 = v
                    if 0 < a0 <= b0 and b0 - a0 < 400: out.setdefault(cur, set()).update(range(a0, b0 + 1))
    return out


def shrink(messages, blocks, ctx, target_frac=0.45):
    """Free context oldest-first by use. blocks: {msg index: dict(kind, path, s, e)}. Returns how many blocks were changed."""
    tool_idx = [i for i, mm in enumerate(messages) if mm['role'] == 'tool']
    keep_last = set(tool_idx[-2:])
    est = lambda: sum(len(json.dumps(mm)) for mm in messages) / 3.2
    cites = cited_lines([mm.get('content') for mm in messages if mm['role'] == 'assistant'])
    def superseded(i):
        b = blocks.get(i)
        return b and b['kind'] == 'read' and any(j > i and c['kind'] == 'read' and c['path'] == b['path'] and c['s'] <= b['s'] and c['e'] >= b['e'] for j, c in blocks.items())
    read_paths = {b['path'] for b in blocks.values() if b['kind'] == 'read'}
    # order: superseded reads, then grep/list, then reads with nothing cited, then the rest; oldest first in each
    def rank(i):
        b = blocks.get(i) or {'kind': 'other'}
        if superseded(i): return 0
        if b['kind'] in ('grep', 'list'): return 1
        if b['kind'] == 'read' and not (cites.get(os.path.basename(b['path'])) or set()) & set(range(b['s'], b['e'] + 1)): return 2
        return 3
    changed = 0
    for i in sorted((i for i in tool_idx if i not in keep_last and len(messages[i]['content']) > 1500 and not messages[i].get('_stub')), key=lambda i: (rank(i), i)):
        if est() <= target_frac * ctx: break
        b = blocks.get(i) or {'kind': 'other'}; r = rank(i)
        if b['kind'] == 'read':
            head = f"[{b['path']} lines {b['s']}-{b['e']} were read earlier"
            if r == 0: new = head + '; a later read covers the same lines.]'
            else:
                want = set()
                for n in (cites.get(os.path.basename(b['path'])) or set()):
                    if b['s'] <= n <= b['e']: want.update(range(n - 3, n + 4))
                kept = [l for l in messages[i]['content'].split('\n')[1:] if l.split(':', 1)[0].isdigit() and int(l.split(':', 1)[0]) in want]
                if kept: new = head + '. Kept only the lines your notes cite; the rest was removed and your notes cover it.]\n' + '\n'.join(kept)
                else: new = head + '; none of it is cited in your notes, so it was removed. Re-read only a specific range if you need it.]'
        elif b['kind'] in ('grep', 'list'):
            new = f"[{b['kind']} output removed to save space" + ('; the files it found have since been read' if b['kind'] == 'grep' and read_paths else '') + '.]'
        else:
            new = '[Older tool output removed to save space. Use your notes.]'
        messages[i]['content'] = new; messages[i]['_stub'] = True; changed += 1
    return changed


def harvest(reasoning, last_path, cap=1500):
    """Lines of the thinking that cite a line number, as short notes; prefix the file of the last read when the line names no file."""
    out = []; base = os.path.basename(last_path) if last_path else ''
    for ln in (reasoning or '').split('\n'):
        ln = ln.strip()
        if not ln or len(ln) > 300 or not NOTE_LINE.search(ln): continue
        if PLAN.match(ln): continue                     # a plan ("Now read 230-484") is not a fact
        mm = re.match(r'^[-*]?\s*(\d{1,5}(?:\s*[-–]\s*\d{1,5})?)\s*:\s*(.*)', ln)
        if base and mm: ln = '%s:%s %s' % (base, re.sub(r'\s', '', mm.group(1)), mm.group(2))
        elif base and not CITE_FILE.search(ln): ln = base + ' ' + ln
        out.append(ln)
    txt = '\n'.join(out)
    return txt[-cap:] if len(txt) > cap else txt


# ------------------------------------------------------------------ loop
SYSTEM = ('You analyse source code with tools. One tool call per response. '
          'After each result, if it holds something the answer needs, or something that builds toward the answer, note it tersely in your message as path:line. '
          'Old tool outputs get removed to save space; your notes stay, so rely on them instead of re-reading files. '
          'Keep every intermediate step maximally concise: while exploring, write your thinking as terse notes (short phrases, file:line, no restating, no filler). Concise is not brief: never skip a check or a read to save words. '
          'The final answer is different: give it the full detail it needs. When you have enough evidence, stop calling tools and reply with ONLY the answer, in the format the question asks for, citing file:line.')
FINAL_NUDGE = 'Stop exploring: no tools are available any more. Write the complete answer now from what you have read, citing file:line.'
RECHECK = ('You still have %d tool calls left. Before this is final, go through your notes: is there code you meant to read and did not, '
           'or a risk you suspected but did not confirm at the line? If so, keep exploring with tools. '
           'If not, reply with the complete answer again, adding anything you missed.')
RECHECK_MIN_LEFT = 10


def ask(root, question, url, model, ctx, max_steps, budget_s, max_tokens, use_map, log=None):
    """-> (answer text, meta dict, messages)"""
    user = question
    if use_map:
        cmap, _ = build(os.path.basename(root) or root, root, 48000)
        user = ('CODE MAP of the repository. It was generated mechanically from the source and shows structure only '
                '(files, line counts, symbol names with line numbers, internal imports, entry points). '
                'Use it to decide what to read. It is not evidence: cite only lines you have actually read with the tools.\n\n'
                '<code_map>\n' + cmap + '</code_map>\n\nQUESTION\n\n' + question)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
    clean = lambda: [{k: v for k, v in mm.items() if k != '_stub'} for mm in messages]
    t0 = time.time(); steps = 0; answer = None; forced = None; err = None
    blocks = {}; nshrink = 0; noteless = 0; nharv = 0; last_read = ''; rechecked = False; toks = 0; ptoks = 0
    say = log or (lambda s: None)
    try:
        while steps < max_steps - 1 and time.time() - t0 <= 0.8 * budget_s:
            r = chat(url, {'model': model, 'messages': clean(), 'tools': TOOLS, 'tool_choice': 'auto', 'max_tokens': max_tokens, **THINK},
                     timeout=max(600, int(budget_s * 1.2)))
            steps += 1
            u = r.get('usage') or {}; toks += u.get('completion_tokens') or 0; ptoks += u.get('prompt_tokens') or 0
            m = r['choices'][0]['message']; calls = m.get('tool_calls') or []
            left = max_steps - steps
            if not calls:
                txt = (m.get('content') or '').strip()
                if txt and not rechecked and left >= RECHECK_MIN_LEFT:
                    # an early answer with steps left gets one check-back; the next answer is final
                    rechecked = True; say('[%d] early answer, checking back once' % steps)
                    messages.append({'role': 'assistant', 'content': txt})
                    messages.append({'role': 'user', 'content': RECHECK % left})
                    continue
                if txt: answer = txt; break            # a reply without a tool call is the answer
                messages.append({'role': 'user', 'content': 'Use a tool, or write the answer.'}); continue
            c = calls[0]
            hv = harvest(m.get('reasoning_content'), last_read)
            if hv: nharv += 1
            vis = ((m.get('content') or '').strip() + ('\nNotes:\n' + hv if hv else '')).strip()
            messages.append({'role': 'assistant', 'content': vis, 'tool_calls': [c]})
            try: args = json.loads(c['function'].get('arguments') or '{}')
            except Exception: args = {}
            nm = c['function']['name']
            say('[%d] %s %s' % (steps, nm, json.dumps(args)[:160]))
            out = run_tool(root, nm, args)[:50000]
            if not vis:
                noteless += 1
                out += '\n\n[Reminder: write your notes (facts with file:line) in your reply text. Your thinking is not kept between turns.]'
            if nm == 'read_file' and out[:1] != 'n':
                mt = re.match(r'(.+?) \(\d+ lines\), lines (\d+)-(\d+):', out)
                if mt: blocks[len(messages)] = {'kind': 'read', 'path': mt.group(1), 's': int(mt.group(2)), 'e': int(mt.group(3))}; last_read = mt.group(1)
            elif nm in ('grep', 'list_dir'):
                blocks[len(messages)] = {'kind': 'grep' if nm == 'grep' else 'list', 'path': str(args.get('path') or '.'), 's': 0, 'e': 0}
            messages.append({'role': 'tool', 'tool_call_id': c.get('id', ''), 'content': out})
            if (u.get('prompt_tokens') or 0) > 0.65 * ctx:
                nshrink += shrink(messages, blocks, ctx)
        if answer is None:                              # safety net hit: answer from what was read, no tools, thinking off
            forced = 'steps' if steps >= max_steps - 1 else 'time'
            messages.append({'role': 'user', 'content': FINAL_NUDGE})
            for _ in range(2):
                r = chat(url, {'model': model, 'messages': clean(), 'max_tokens': max_tokens * 2, **NOTHINK}, timeout=max(900, int(budget_s * 1.2)))
                steps += 1
                u = r.get('usage') or {}; toks += u.get('completion_tokens') or 0; ptoks += u.get('prompt_tokens') or 0
                txt = (r['choices'][0]['message'].get('content') or '').strip()
                if txt: answer = txt; break
                messages.append({'role': 'assistant', 'content': ''}); messages.append({'role': 'user', 'content': FINAL_NUDGE})
    except Exception as e:
        err = repr(e)[:300]
    meta = {'version': 'CRv23', 'model': model, 'repo': root, 'steps': steps, 'forced': forced, 'rechecked': rechecked,
            'wall_s': round(time.time() - t0, 1), 'answered': answer is not None, 'out_tokens': toks, 'prompt_tokens': ptoks,
            'harvested_turns': nharv, 'shrunk_blocks': nshrink, 'noteless_calls': noteless, 'error': err}
    return answer, meta, clean()


def main():
    ap = argparse.ArgumentParser(description='Ask a question about a codebase; the answer cites file:line. Read-only.')
    ap.add_argument('repo')
    ap.add_argument('question', nargs='?')
    ap.add_argument('--question-file')
    ap.add_argument('--url', default=os.environ.get('CRV23_URL', 'http://127.0.0.1:8010/v1'))
    ap.add_argument('--model', default=os.environ.get('CRV23_MODEL', 'qwen3.8-flash-next'))
    ap.add_argument('--ctx', type=int, default=65536, help="the server's per-request context, in tokens")
    ap.add_argument('--max-steps', type=int, default=60)
    ap.add_argument('--budget-s', type=int, default=2700)
    ap.add_argument('--max-tokens', type=int, default=16384)
    ap.add_argument('--no-map', action='store_true', help='do not give the model a code map up front')
    ap.add_argument('--out', help='directory for answer.md, meta.json and the full transcript')
    ap.add_argument('-q', '--quiet', action='store_true', help='do not print tool calls to stderr')
    a = ap.parse_args()
    q = open(a.question_file).read() if a.question_file else a.question
    if not q: ap.error('give a question or --question-file')
    root = os.path.realpath(a.repo)
    if not os.path.isdir(root): ap.error('not a directory: ' + a.repo)
    log = None if a.quiet else (lambda s: print(s, file=sys.stderr, flush=True))
    answer, meta, messages = ask(root, q, a.url, a.model, a.ctx, a.max_steps, a.budget_s, a.max_tokens, not a.no_map, log)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, 'answer.md'), 'w').write(answer or '')
        json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
        json.dump(messages, open(os.path.join(a.out, 'messages.json'), 'w'))
    print(answer or '')
    print(json.dumps(meta), file=sys.stderr)
    return 0 if answer else 1


if __name__ == '__main__':
    sys.exit(main())
