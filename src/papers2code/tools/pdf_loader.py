from pathlib import Path
from unstructured.partition.pdf import partition_pdf

def load_pdf_text(path_or_url: str) -> tuple[str, dict]:
    """Return concatenated text and a simple section map"""
    elements = partition_pdf(filename=path_or_url) if Path(path_or_url).exists() else partition_pdf(url=path_or_url)
    text_parts, sections = [], {"titles": [], "narrative": []}
    for e in elements:
        et = e.category
        if et == "Title":
            sections["titles"].append(str(e))
            text_parts.append(str(e))
        elif et in ("NarrativeText","ListItem","Table"):
            sections["narrative"].append(str(e))
            text_parts.append(str(e))
    return "\n\n".join(text_parts), sections
