"""Create both databases without opening Jupyter.

Executes the code cells of 01_external_db_setup.ipynb and
02_core_db_setup.ipynb in order, in-process, so the notebooks stay the single
source of truth for the data. Used by the test-suite and the evaluation to
build fresh, isolated copies.

    python scripts/setup_databases.py                 # into ./data (same as running the notebooks)
    python scripts/setup_databases.py --root /tmp/x   # into a scratch copy
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("01_external_db_setup.ipynb", "02_core_db_setup.ipynb")


def prepare_root(root: Path) -> None:
    """Copy what the notebooks need (models, jsonl files, utils.py) into ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for nb in NOTEBOOKS:
        shutil.copy2(SOLUTION_DIR / nb, root / nb)
    shutil.copy2(SOLUTION_DIR / "utils.py", root / "utils.py")
    shutil.copytree(SOLUTION_DIR / "data" / "models", root / "data" / "models", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "data" / "core").mkdir(parents=True, exist_ok=True)
    ext = root / "data" / "external"
    ext.mkdir(parents=True, exist_ok=True)
    for f in (SOLUTION_DIR / "data" / "external").glob("*.jsonl"):
        shutil.copy2(f, ext / f.name)


def run_notebook(path: Path, quiet: bool = True) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__notebook__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
            try:
                exec(compile(src, f"{path.name}[cell {i}]", "exec"), ns)  # noqa: S102 - our own notebook
            except Exception as exc:
                raise RuntimeError(f"{path.name} cell {i} failed: {exc}\n--- source ---\n{src}") from exc


def setup(root: Path | None = None, quiet: bool = True) -> Path:
    """Build cultpass.db and udahub.db under ``root`` (default: the solution folder)."""
    root = Path(root or SOLUTION_DIR).resolve()
    if root != SOLUTION_DIR:
        prepare_root(root)
    cwd, path0 = os.getcwd(), list(sys.path)
    # The notebooks import `data.models` and `utils` relative to their folder:
    # hide already-imported copies while they run, then put them back.
    def ours(m):
        return m == "utils" or m == "data" or m.startswith("data.")
    saved = {m: sys.modules.pop(m) for m in [m for m in sys.modules if ours(m)]}
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        for nb in NOTEBOOKS:
            run_notebook(root / nb, quiet=quiet)
    finally:
        os.chdir(cwd)
        sys.path[:] = path0
        for m in [m for m in sys.modules if ours(m)]:
            sys.modules.pop(m)
        sys.modules.update(saved)
    return root / "data"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="build into this folder instead of the solution")
    ap.add_argument("--verbose", action="store_true", help="show the notebooks' printed output")
    args = ap.parse_args()
    data = setup(args.root, quiet=not args.verbose)
    for db in ("core/udahub.db", "external/cultpass.db"):
        print("created", data / db)
