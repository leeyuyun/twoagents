# TwoBotsChat

Two-agent chat loop for Ollama (or OpenAI-compatible) with structured JSON output.

## Requirements
- Python 3.10+
- Ollama running locally

## Install
```bash
pip install -r requirements.txt
```

Optional (editable):
```bash
pip install .
```

## Run
```bash
python main.py --model qwen3:14b --max-turns 40 --min-sat 95 --stable-rounds 2
```

## New options
- `--timeout-s` : HTTP read timeout in seconds (default 120)
- `--summary-keep-last` : keep last N transcript entries, summarize the rest (default 6)
- `--summary-max-points` : max bullet points in the summary (default 12)

Example:
```bash
python main.py --timeout-s 300 --summary-keep-last 6 --summary-max-points 12
```

## Files
- `main.py` : CLI entry
- `ollama_client.py` : Ollama/OpenAI-compatible streaming client
- `agents.py` : agent configs + message builder
- `orchestrator.py` : turn loop, transcript, summary window
- `transcript_*.jsonl` : saved runs (if enabled)

## Notes
- Default `base_url` is `http://127.0.0.1:11434`
- Use `--transcript-path` to set output file path
