import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI


MODEL_NAME = os.getenv("P2C_MODEL", "gpt-4o-mini")
_client = None

def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def chat_json(prompt: str,
              system: str = "You are a precise extraction assistant.",
              log_dir: Optional[Path] = None,
              log_name: str = "mentions") -> Dict[str, Any]:
    """
    Call the model, prefer JSON, but robustly parse raw content if needed
    """
    resp = client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":prompt}],

        response_format={"type":"json_object"},
        temperature=1,
    )
    msg = resp.choices[0].message
    raw = getattr(msg, "content", None) or ""

    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}

    if log_dir:
        (log_dir / "logs").mkdir(parents=True, exist_ok=True)
        (log_dir / "logs" / f"{log_name}_prompt.txt").write_text(prompt, encoding="utf-8")
        (log_dir / "logs" / f"{log_name}_response.raw.json").write_text(raw, encoding="utf-8")
        (log_dir / "logs" / f"{log_name}_response.parsed.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return data
