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


def call_model(
    model: str,
    prompt_path: str,
    text: str = "",
    image: str | None = None,
    timeout_s: int = 300,
    use_cache: bool = True,
) -> dict:
    """Run a prompt template (prompts/<stage>/<version>.md with sibling <version>.schema.json) on inputs.

    Returns parsed output, cache status/key, latency, and prompt/image/response hashes.
    """
    prompt_file = ROOT / prompt_path
    schema_file = prompt_file.with_suffix(".schema.json")
    template = prompt_file.read_text()
    schema = json.loads(schema_file.read_text())
    img_bytes = Path(image).read_bytes() if image else b""
    prompt = template.replace("{{text}}", text)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    image_hash = hashlib.sha256(img_bytes).hexdigest()
    key = hashlib.sha256(json.dumps([model, template, schema, text]).encode() + img_bytes).hexdigest()[:24]
    cache_file = CACHE / f"{key}.json"
    if use_cache and cache_file.exists():
        out = json.loads(cache_file.read_text())
        return {
            "output": out,
            "cached": True,
            "latency_s": 0.0,
            "key": key,
            "prompt_sha256": prompt_hash,
            "image_sha256": image_hash,
            "response_sha256": hashlib.sha256(
                json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    t0 = time.time()
    if model == "codex":
        out = _codex(prompt, schema_file, image, timeout_s)
    elif model.startswith("claude-"):
        out = _claude(model.removeprefix("claude-"), prompt, schema, image, timeout_s)
    else:
        raise ValueError(f"unknown method model {model!r}; use 'codex' or 'claude-<model>'")
    latency = time.time() - t0

    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(out))
    with LOG.open("a") as f:
        f.write(json.dumps({"t": t0, "model": model, "prompt": prompt_path, "key": key, "latency_s": round(latency, 2),
                            "image": image, "chars_in": len(prompt), "chars_out": len(json.dumps(out))}) + "\n")
    return {
        "output": out,
        "cached": False,
        "latency_s": latency,
        "key": key,
        "prompt_sha256": prompt_hash,
        "image_sha256": image_hash,
        "response_sha256": hashlib.sha256(
            json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


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
