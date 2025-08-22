from typing import List, Dict, Any

from rapidfuzz import fuzz

from papers2code.tools.kaggle_client import kaggle_search_datasets, kaggle_files_and_size

def probe_kaggle_matches(candidates: List[Dict], max_checks_per_name: int = 8) -> List[Dict[str, Any]]:
    """
    For each named dataset from the paper, query Kaggle and compute a fuzzy score
    between the paper name and {title, ref}. Return enriched matches:
    {ref, title, url, license, score, total_mb, files}
    """
    results: List[Dict[str, Any]] = []
    names = [c.get("name") for c in candidates if (c.get("name") or "").strip()]
    seen_refs = set()

    for name in names:
        q = name
        items = kaggle_search_datasets(q, limit=max_checks_per_name)
        for it in items:
            ref = it.get("ref")
            if not ref or ref in seen_refs:
                continue
            seen_refs.add(ref)
            title = (it.get("title") or "").lower().strip()
            score = max(
                fuzz.ratio((name or "").lower().strip(), title),
                fuzz.ratio((name or "").lower().strip(), (ref or "").lower().strip())
            )
            files, mb = kaggle_files_and_size(ref)
            results.append({
                "paper_name": name,
                "ref": ref,
                "title": it.get("title"),
                "url": it.get("url"),
                "license": it.get("license"),
                "score": float(score),
                "total_mb": mb,
                "files": files,
            })

    results.sort(key=lambda d: d["score"], reverse=True)
    return results
