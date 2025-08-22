import json
import time
from pathlib import Path

from papers2code.state import PipelineState
from papers2code.config import settings

# Step A: PDF -> text
from papers2code.tools.pdf_loader import load_pdf_text
from papers2code.tools.artifacts import write_text

# Step B: Paper -> dataset mentions (LLM, logs prompt/response)
from papers2code.nodes.dataset_mention_extractor import extract_dataset_mentions

# Step C: Probe Kaggle (no download): search + list files + score
from papers2code.nodes.dataset_resolver import probe_kaggle_matches

# Step D: Select one winner with transparent rationale
from papers2code.nodes.selector import choose_best_match

# Step E: Download chosen dataset
from papers2code.tools.kaggle_client import kaggle_download_dataset

# Step F/G/H/I: Modality-aware sampling + profiling + EDA + dataset card
from papers2code.tools.modality import guess_modality
from papers2code.tools.image_sampler import sample_images_auto
from papers2code.tools.image_profiler import profile_images
from papers2code.tools.image_eda import save_class_bar_chart, save_sample_grid

# Step J/K/L: Methods -> Code scaffold -> Wiki
from papers2code.nodes.methods_extractor import extract_methods
from papers2code.nodes.code_synthesizer import render_code_templates
from papers2code.nodes.wiki_composer import compose_wiki


def _step(title: str):
    print(f"\n=== {title} ===")


t0 = time.perf_counter()


def _write_image_dataset_card(
    out_dir: Path,
    title: str,
    url: str | None,
    license_name: str | None,
    img_profile: dict,
) -> None:
    lines: list[str] = []
    lines.append(f"# Dataset Card — {title}")
    if url:
        lines.append(f"- Kaggle: {url}")
    lines.append(f"- License: {license_name or 'Unknown'}")
    lines.append("")
    lines += [
        "## Image Sample Profile",
        f"- Total images (sample): {img_profile.get('total_images', 0)}",
        "### Per-class (sample)",
    ]
    for k, v in (img_profile.get("per_class") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- Approx duplicate rate (phash): {img_profile.get('approx_duplicate_rate', 0.0):.3f}")
    lines.append("")
    lines += [
        "## Quick EDA",
        "- See `eda/class_counts.png` and `eda/sample_grid.png`.",
    ]
    write_text(out_dir / "dataset_card.md", "\n".join(lines))


def run_pipeline(paper_source: str, out_dir: Path) -> PipelineState:
    """
    Main graph workflow
    Runs the agent in 8 steps from paper ingestion to template eneration
    parameters:
        paper_source (str): The paper path (from the cli call relative call)
        out_dir (str): The save folder (created if doesn't exist)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    st = PipelineState(paper_source=paper_source)

    # Step A: Load paper text from pdf
    _step("A. Load paper")
    paper_text, sections = load_pdf_text(paper_source)
    st.paper_text = paper_text
    st.sections = sections

    # Step B: Extract dataset mentions (LLM)
    _step("B. Extract dataset mentions")
    st.dataset_candidates = extract_dataset_mentions(st.paper_text, log_dir=out_dir)
    (out_dir / "candidates.json").write_text(
        json.dumps(st.dataset_candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not st.dataset_candidates:
        msg = "No concrete dataset mentions found in the paper"
        st.issues.append(msg)
        write_text(out_dir / "report.txt", msg)
        print(msg)
        return st

    # Step C: Probe Kaggle matches
    _step("C. Probe Kaggle")
    matches = probe_kaggle_matches(st.dataset_candidates, max_checks_per_name=8)
    (out_dir / "resolver_matches.json").write_text(
        json.dumps(matches, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not matches:
        msg = "No Kaggle matches found for extracted names."
        st.issues.append(msg)
        write_text(out_dir / "report.txt", msg)
        print(msg)
        return st

    # Step D: Select one winner with transparent rationale
    _step("D. Select match")
    paper_primary = next((c.get("name") for c in st.dataset_candidates if c.get("name")), None)
    winner, rationale = choose_best_match(matches, paper_primary_name=paper_primary)
    selection = {"winner": winner, "rationale": rationale, "alternatives": matches[:10]}
    (out_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not winner:
        msg = "Selection failed - no winner after tie-breakers"
        st.issues.append(msg)
        write_text(out_dir / "report.txt", msg)
        print(msg)
        return st

    # Console: short summary pre-download
    modality = guess_modality(winner.get("files") or [])
    print(f"Chosen: {winner['ref']}  |  Title: {winner.get('title')}  |  Modality: {modality}")
    if winner.get("url"):
        print(f"Dataset URL: {winner.get('url')}")

    # Step E: Download ONLY the chosen dataset
    _step("E. Download dataset")
    slug = winner["ref"]
    ds_dir = out_dir / f"dataset_{slug.replace('/','_')}"
    if ds_dir.exists() and any(ds_dir.iterdir()):
        print(f"Cache hit: {ds_dir} already exists — skipping download.")
    else:
        kaggle_download_dataset(slug, ds_dir)
    print(f"Downloaded to: {ds_dir}")
    st.sample_dir = str(ds_dir)

    # Step F/G: Sample images automatically
    _step("F-G-H. Sample, profile & EDA")
    per_class = settings.image_sample_max if hasattr(settings, "image_sample_max") else 50
    sample_dir, per_class_counts, broken = sample_images_auto(
        ds_dir, out_dir, per_class=int(per_class), max_total=int(per_class) * 10
    )

    # Step H: Profile + EDA
    img_profile = profile_images(sample_dir)
    (out_dir / "eda").mkdir(parents=True, exist_ok=True)
    save_class_bar_chart(per_class_counts, out_dir / "eda" / "class_counts.png")
    save_sample_grid(sample_dir, out_dir / "eda" / "sample_grid.png")

    # Step I: Dataset Card
    _write_image_dataset_card(
        out_dir=out_dir,
        title=winner.get("title") or slug,
        url=winner.get("url"),
        license_name=winner.get("license"),
        img_profile=img_profile,
    )

    # Console: Sampling Summary
    print(f"Sample dir: {sample_dir}")
    print(f"Classes (sample): {len(per_class_counts)} | Broken files skipped: {broken}")
    print("Artifacts: dataset_card.md, eda/class_counts.png, eda/sample_grid.png")

    # Step J: Methods extractor (LLM with logging & CIFAR-10 defaults)
    _step("J. Extract Methods (LLM)")
    method_spec = extract_methods(st.paper_text, st.sections, log_dir=out_dir)
    # Saved as artifacts/method_spec.json by the extractor
    print("Methods extracted -> method_spec.json")

    # Step K: Code scaffold (Jinja2 templates)
    _step("K. Render code scaffold")
    code_paths = render_code_templates(method_spec, templates_dir=None, out_dir=out_dir)
    print("Code scaffold generated under artifacts/code/")

    # Step L: Paper -> Code Wiki
    _step("L. Compose paper->code wiki")
    compose_wiki(method_spec, code_paths, out_dir / "paper_to_code_wiki.md")
    print("Wiki written -> artifacts/paper_to_code_wiki.md")

    return st
