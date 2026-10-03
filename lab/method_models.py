"""Method models: LLM/VLM calls made *inside* the assembly pipeline (not the lab agents' own models).

Backends use the CLIs already logged in on this machine, so no API keys are needed:
  codex           -> `codex exec -i image --output-schema` (ChatGPT login)
  claude-<model>  -> `claude -p --json-schema` (Claude login), e.g. claude-haiku, claude-sonnet
Calls are deterministic per (model, prompt file content, inputs): a response cache makes reruns of a
method version reproduce the same outputs. Every call is appended to record/model_calls.jsonl.
"""

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "record" / "model_cache"
LOG = ROOT / "record" / "model_calls.jsonl"


def call_model(model: str, prompt_path: str, text: str = "", image: str | None = None, timeout_s: int = 300) -> dict:
    """Run a prompt template (prompts/<stage>/<version>.md with sibling <version>.schema.json) on inputs.

    Returns {"output": parsed JSON, "cached": bool, "latency_s": float, "key": cache key}.
    """
    prompt_file = ROOT / prompt_path
    schema_file = prompt_file.with_suffix(".schema.json")
    template = prompt_file.read_text()
    schema = json.loads(schema_file.read_text())
    img_bytes = Path(image).read_bytes() if image else b""
    key = hashlib.sha256(json.dumps([model, template, schema, text]).encode() + img_bytes).hexdigest()[:24]
    cache_file = CACHE / f"{key}.json"
    if cache_file.exists():
        return {"output": json.loads(cache_file.read_text()), "cached": True, "latency_s": 0.0, "key": key}

    prompt = template.replace("{{text}}", text)
    t0 = time.time()
    if model == "codex":
        out = _codex(prompt, schema_file, image, timeout_s)
    elif model.startswith("claude-"):
        out = _claude(model.removeprefix("claude-"), prompt, schema, image, timeout_s)
    else:
        raise ValueError(f"unknown method model {model!r}; use 'codex' or 'claude-<model>'")
    latency = time.time() - t0

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out))
    with LOG.open("a") as f:
        f.write(json.dumps({"t": t0, "model": model, "prompt": prompt_path, "key": key, "latency_s": round(latency, 2),
                            "image": image, "chars_in": len(prompt), "chars_out": len(json.dumps(out))}) + "\n")
    return {"output": out, "cached": False, "latency_s": latency, "key": key}


def _codex(prompt, schema_file, image, timeout_s):
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "out.json"
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-C", tmp,
               "--output-schema", str(schema_file), "-o", str(out_file)]
        if image:
            cmd += ["-i", str(image)]
        subprocess.run(cmd + ["--", prompt], check=True, capture_output=True, text=True, timeout=timeout_s,
                       stdin=subprocess.DEVNULL)
        return json.loads(out_file.read_text())


def _claude(model, prompt, schema, image, timeout_s):
    if image:
        prompt = f"First read the image at {image} with the Read tool.\n\n{prompt}"
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--json-schema", json.dumps(schema),
           "--allowedTools", "Read", "--", prompt]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    payload = json.loads(res.stdout)
    out = payload.get("structured_output") or payload.get("result")
    return json.loads(out) if isinstance(out, str) else out
