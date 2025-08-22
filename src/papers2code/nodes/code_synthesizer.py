import json
import os
from pathlib import Path
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape


try:
    from importlib.resources import files as ir_files
except Exception:
    ir_files = None


def _resolve_templates_dir() -> Path:
    """
    Resolution order:
      1. P2C_TEMPLATES_DIR env var, if set and exists
      2. package resource path papers2code/templates (importlib.resources)
      3. repo fallback: src/papers2code/templates (relative to this file)
    """
    env_path = os.getenv("P2C_TEMPLATES_DIR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    # 2. package data
    if ir_files is not None:
        try:
            pkg_path = ir_files("papers2code").joinpath("templates")
            # convert to real FS path
            p = Path(str(pkg_path))
            if p.exists():
                return p
        except Exception:
            pass

    # 3. repo fallback (editable install)
    p = Path(__file__).resolve().parent.parent / "templates"
    return p


def _env(templates_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(disabled_extensions=("py","yml","md","ipynb","txt")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_code_templates(spec: Dict[str, Any], templates_dir: Path | None, out_dir: Path) -> Dict[str, str]:
    """
    Renders code files from templates using the extracted spec
    Returns a dict of {relative_path: absolute_path}
    """
    if templates_dir is None:
        templates_dir = _resolve_templates_dir()

    env = _env(templates_dir)
    outputs = {}

    mapping = {
        "preprocess.py.j2": "code/src/preprocess.py",
        "model.py.j2": "code/src/model.py",
        "train.py.j2": "code/src/train.py",
        "environment.yml.j2": "code/environment.yml",
        "eda_notebook.ipynb.j2": "code/notebooks/EDA.ipynb",
        "dataset_card.md.j2": "code/DATASET_CARD_TEMPLATE.md",
        "config.yaml.j2": "code/config.yaml",
        "README.md.j2": "code/README.md",
    }
    mapping["Makefile.j2"] = "code/Makefile"

    out_dir.mkdir(parents=True, exist_ok=True)
    for tmpl, rel_out in mapping.items():
        tpl = env.get_template(tmpl)  # will raise TemplateNotFound with clear path
        rendered = tpl.render(**spec)
        path = out_dir / rel_out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        outputs[rel_out] = str(path)

    (out_dir / "method_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return outputs
