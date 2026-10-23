# CRv23

Ask questions about any codebase; get answers citing file:line.

CRv23 is a single Python file. Point it at a repo and an OpenAI-compatible model server (llama.cpp, vLLM, ...), ask a question, and it explores the code with read-only tools (list, grep, read) and answers with file:line citations. It cannot run commands or edit files, and it talks only to the server you give it.

## Requirements

- Python 3 (standard library only; nothing to install; tested on 3.13)
- `grep` on your PATH
- An OpenAI-compatible server with tool calling. For llama.cpp, start `llama-server` with `--jinja`.

## Quick start

```bash
git clone https://github.com/distilos/crv23.git
cd crv23
python3 crv23.py /path/to/repo "How does a request get routed?" \
  --url http://localhost:8010/v1 --model YOUR_MODEL_ID
```

The answer prints to stdout; progress (each tool call) prints to stderr.

## Options

| Option | Default | What it does |
|---|---|---|
| `--url` | `http://127.0.0.1:8010/v1` (or `$CRV23_URL`) | Server base URL |
| `--model` | `qwen3.8-flash-next` (or `$CRV23_MODEL`) | Model id on the server |
| `--question-file FILE` | | Read the question from a file instead |
| `--ctx` | `65536` | The server's per-request context, in tokens. Set it to your server's real value. |
| `--max-steps` | `60` | Safety cap on tool calls |
| `--budget-s` | `2700` | Safety cap on wall time, in seconds |
| `--max-tokens` | `16384` | Max tokens per model turn |
| `--no-map` | off | Don't give the model a code map up front |
| `--out DIR` | | Save `answer.md`, `meta.json` and the full transcript |
| `-q` | off | Don't print tool calls to stderr |

If `OPENAI_API_KEY` is set, it is sent as a bearer token.

## How it works

- It builds a code map first (file tree, symbols with line numbers, imports, entry points), so the model knows where to look.
- The model keeps short file:line notes as it reads. When the context fills, old tool output is trimmed, but the notes and the lines they cite are kept.
- If the model answers early, it is asked once to check for code it meant to read and risks it didn't confirm.
- The model stops when it replies without a tool call. The step and time caps are only a safety net.

The sampling is tuned for Qwen (temperature 1.0, top_p 0.95, top_k 20, thinking on, reasoning effort medium). For another model, edit `THINK` and `NOTHINK` near the top of the model section.
