from pathlib import Path
import re
from typing import List, Dict

from papers2code.llm.openai_client import chat_json

KAGGLE_URL_RE = r"https?://(?:www\.)?kaggle\.com/(?:datasets|competitions)/[^\s\)\]]+"

def extract_dataset_mentions(paper_text: str, log_dir: Path) -> List[Dict]:
    """
    Return candidate datasets mentioned in the paper
    Strategy:
      (1) deterministic scrape of Kaggle URLs
      (2) LLM extraction of specific named datasets (no generic phrases)
    """
    candidates: List[Dict] = []

    # 1. deterministic links
    for m in re.finditer(KAGGLE_URL_RE, paper_text, flags=re.IGNORECASE):
        url = m.group(0)
        candidates.append({
            "name": None, "url_if_any": url,
            "context_snippet": url, "confidence": 0.95
        })

    # 2. LLM extraction
    excerpt = paper_text[:80_000]
    prompt = (
        "You read a research paper excerpt. Extract concrete dataset references.\n"
        "Rules:\n"
        "- Only return specific named datasets (e.g., 'CIFAR-10', 'UCI Adult', 'COCO'), NOT generic phrases.\n"
        "- If a dataset URL appears in text, include it; otherwise url_if_any=null.\n"
        "- Prefer mentions in sections like Data/Dataset/Experimental Setup.\n"
        "- Return strict JSON: {\"candidates\": [{\"name\": str|null, \"url_if_any\": str|null, "
        "\"context_snippet\": str, \"confidence\": float}]}\n\n"
        f"Paper excerpt:\n{excerpt}"
    )
    data = chat_json(prompt, log_dir=log_dir, log_name="mentions")
    llm_items = data.get("candidates") if isinstance(data, dict) else None

    if llm_items:
        seen = set()
        out = []
        for c in llm_items:
            key = ((c.get("name") or "").strip().lower(), (c.get("url_if_any") or "").strip().lower())
            if key not in seen:
                seen.add(key)
                out.append(c)
        candidates.extend(out)

    return candidates
