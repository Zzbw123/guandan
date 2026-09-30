from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def ensure_output_dirs(project_root: Path) -> tuple[Path, Path]:
    figures_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def export_figure_bundle(
    fig,
    figure_stem: str,
    project_root: str | Path,
    contract: dict[str, Any],
    source_data_paths: Iterable[str | Path],
    script_path: str | Path,
    dpi: int = 300,
    include_pdf: bool = True,
    include_tiff: bool = False,
    tiff_dpi: int = 600,
    backend: str | None = None,
    notes: dict[str, Any] | None = None,
) -> dict[str, str]:
    project_root = Path(project_root)
    figures_dir, _ = ensure_output_dirs(project_root)

    svg_path = figures_dir / f"{figure_stem}.svg"
    png_path = figures_dir / f"{figure_stem}.png"
    pdf_path = figures_dir / f"{figure_stem}.pdf"
    tiff_path = figures_dir / f"{figure_stem}.tiff"
    metadata_path = figures_dir / f"{figure_stem}.trace.json"

    # Nature-style default: editable vector first, raster preview second.
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=dpi)
    exports = {
        "svg": str(svg_path),
        "png": str(png_path),
    }

    if include_pdf:
        fig.savefig(pdf_path)
        exports["pdf"] = str(pdf_path)
    if include_tiff:
        fig.savefig(tiff_path, dpi=tiff_dpi)
        exports["tiff"] = str(tiff_path)

    metadata = {
        "figure_stem": figure_stem,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "exports": exports,
        "contract": contract,
        "source_data_paths": [str(Path(p)) for p in source_data_paths],
        "script_path": str(Path(script_path)),
        "primary_output": str(svg_path),
        "notes": notes or {},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    exports["trace"] = str(metadata_path)
    return exports
