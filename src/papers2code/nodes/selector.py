from typing import List, Dict, Any, Tuple


def choose_best_match(matches: List[Dict[str, Any]], paper_primary_name: str | None = None) -> Tuple[Dict[str, Any] | None, List[str]]:
    """
    Choose one match using transparent tie-breakers
    Returns (winner_match_or_none, rationale_steps)
    """
    if not matches:
        return None, ["No matches to choose from"]

    steps: List[str] = []
    items = matches[:]  # copy

    # 1. Highest fuzzy score already sorted by caller
    top_score = items[0]["score"]
    steps.append(f"Start with highest fuzzy score = {top_score}")

    # 2. Keep only those within 5 points of top score (to avoid picking poor matches)
    top_band = [m for m in items if m["score"] >= top_score - 5]
    steps.append(f"{len(top_band)} candidates within 5 points of top score")

    # 3. Prefer non-empty file lists
    non_empty = [m for m in top_band if m.get("files")]
    if non_empty:
        top_band = non_empty
        steps.append(f"{len(top_band)} candidates have non-empty file lists")

    # 4. Prefer usable file patterns per modality
    def usable_score(m):
        files = m.get("files") or []
        names = [ (f.get("name") or "").lower() for f in files ]
        score = 0
        if any(n.endswith((".csv", ".parquet")) for n in names): score += 2
        if any(n.endswith((".png",".jpg",".jpeg")) for n in names): score += 1
        if any(n.startswith("data_batch_") or n == "test_batch" for n in names): score += 1  # CIFAR-like
        return score

    top_band.sort(key=usable_score, reverse=True)
    steps.append("Ranked by usable file patterns (csv/parquet > image files > CIFAR batch files).")

    # 5. Prefer explicit license
    licensed = [m for m in top_band if m.get("license")]
    if licensed:
        top_band = licensed
        steps.append(f"{len(top_band)} candidates have explicit license; prefer those")

    # 6. Prefer refs/titles containing token of paper name if provided
    if paper_primary_name:
        token = paper_primary_name.lower().replace("-", "").replace(" ", "")
        def has_token(m):
            ref = (m.get("ref") or "").lower().replace("-", "").replace(" ", "")
            title = (m.get("title") or "").lower().replace("-", "").replace(" ", "")
            return (token in ref) or (token in title)
        tokened = [m for m in top_band if has_token(m)]
        if tokened:
            top_band = tokened
            steps.append(f"Filtered by presence of token '{token}' in ref/title")

    # 7. Final tie-breaker: shortest ref (often simpler/maintained)
    top_band.sort(key=lambda m: len(m.get("ref") or "z" * 999))
    winner = top_band[0]
    steps.append(f"Winner: {winner.get('ref')} (title='{winner.get('title')}')")

    return winner, steps
