# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "openpyxl",
#   "matplotlib",
#   "matplotlib-venn",
# ]
# ///
"""Compare Ex15 POS/NEG/common interactor lists against Roth et al. 2023 AP-MS preys.

Reads `mmc3.xlsx` (Roth et al., Molecular Cell 2023 supplementary table —
SAINTexpress AP-MS/BioID results across 17 gene-expression regulators profiled
in N2a mouse cells), filters by BFDR, and intersects the high-confidence preys
with the three clean HGNC gene lists produced by `clean_lists.py`.

Because Roth's data uses mouse gene symbols (Title Case) and our data uses
human HGNC symbols (UPPERCASE), we uppercase PreyGene on the Roth side to align.
This is the pragmatic shortcut from the project notes; it catches orthologs
that share spelling modulo case (~majority of mRNA-processing factors) and
misses the small tail where symbols diverge.

Outputs (under `<folder>/roth_comparison/` — default folder is isoforms_distinct):
  all_baits/                         every Roth bait pooled
    roth_significant_preys.txt       flattened uppercase gene set used for overlap
    overlap_{POS_specific,NEG_specific,common}.txt
    figures/venn_{POS,NEG,common}_vs_roth.png
    summary.md                       counts + per-bait contribution table
  chtop_sap30bp/                     Chtop + Sap30bp baits only (substring match)
    (same structure)
  bait_scope_comparison.md           which genes does the chtop/sap30bp filter drop?

Usage:
  uv run roth_comparison.py
  uv run roth_comparison.py --bfdr 0.01
  uv run roth_comparison.py --bait-scope all_baits
  uv run roth_comparison.py --bait-scope chtop_sap30bp
  uv run roth_comparison.py --sheet "1_SAINTexpress_AP-MS"
  uv run roth_comparison.py --folder isoforms_collapsed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2, venn2_circles

HERE = Path(__file__).parent
MMC3_PATH = HERE / "mmc3.xlsx"

LISTS = {
    "common":       "Ex15_common_clean_genes.txt",
    "POS_specific": "Ex15_POS_specific_clean_genes.txt",
    "NEG_specific": "Ex15_NEG_specific_clean_genes.txt",
}

POS_COLOR = "#E63946"   # matches venn_diagram.py
NEG_COLOR = "#1D7874"   # matches venn_diagram.py
COMMON_COLOR = "#6A4C93"
ROTH_COLOR = "#457B9D"

LIST_COLORS = {
    "POS_specific": POS_COLOR,
    "NEG_specific": NEG_COLOR,
    "common":       COMMON_COLOR,
}

LIST_DISPLAY = {
    "POS_specific": "Ex15 POS specific",
    "NEG_specific": "Ex15 NEG specific",
    "common":       "Ex15 common",
}

CHTOP_SAP30BP_PATTERNS = ("chtop", "sap30bp")

BAIT_SCOPES = ("all_baits", "chtop_sap30bp")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gene_list(path: Path) -> set[str]:
    """Read a _clean_genes.txt file; strip, skip blanks / comments."""
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def pick_sheet(xlsx_path: Path, requested: str | None) -> str:
    """Return a sheet name: exact match if --sheet given, else first one
    whose name contains 'saint' case-insensitively."""
    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names
    print(f"  sheets in {xlsx_path.name}: {sheets}")
    if requested:
        if requested not in sheets:
            sys.exit(f"ERROR: sheet '{requested}' not found. Available: {sheets}")
        return requested
    for s in sheets:
        if "saint" in s.lower():
            print(f"  auto-selected sheet: {s}")
            return s
    sys.exit(
        f"ERROR: no sheet matching 'SAINT' found in {xlsx_path.name}. "
        f"Pass --sheet explicitly. Available: {sheets}"
    )


def load_roth(xlsx_path: Path, sheet: str | None) -> pd.DataFrame:
    sheet_name = pick_sheet(xlsx_path, sheet)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    print(f"  loaded {len(df):,} rows, columns: {list(df.columns)}")

    required = {"BFDR", "PreyGene", "Bait"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: Roth sheet missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    # Drop rows with no gene symbol
    df = df.dropna(subset=["PreyGene"]).copy()
    df["PreyGene_upper"] = df["PreyGene"].astype(str).str.strip().str.upper()
    df["Bait"] = df["Bait"].astype(str).str.strip()
    return df


def filter_roth(
    df: pd.DataFrame,
    *,
    bfdr_cutoff: float,
    bait_scope: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply BFDR and (optionally) bait-name filter.

    Returns (filtered_df, matched_bait_labels).
    """
    sig = df[df["BFDR"] <= bfdr_cutoff].copy()

    if bait_scope == "all_baits":
        matched_baits = sorted(sig["Bait"].unique().tolist())
        return sig, matched_baits

    if bait_scope == "chtop_sap30bp":
        lower = sig["Bait"].str.lower()
        mask = pd.Series(False, index=sig.index)
        for pat in CHTOP_SAP30BP_PATTERNS:
            mask = mask | lower.str.contains(pat, na=False)
        filtered = sig[mask].copy()
        matched_baits = sorted(filtered["Bait"].unique().tolist())
        return filtered, matched_baits

    raise ValueError(f"unknown bait_scope: {bait_scope!r}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_venn(
    our_genes: set[str],
    roth_genes: set[str],
    list_key: str,
    out_path: Path,
    *,
    bait_scope_label: str,
    bfdr_cutoff: float,
) -> None:
    our_only = our_genes - roth_genes
    roth_only = roth_genes - our_genes
    both = our_genes & roth_genes

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    fig.patch.set_facecolor("white")

    subsets = (len(our_only), len(roth_only), len(both))
    our_color = LIST_COLORS[list_key]

    v = venn2(
        subsets=subsets,
        set_labels=(LIST_DISPLAY[list_key], "Roth et al. preys"),
        set_colors=(our_color, ROTH_COLOR),
        alpha=0.55,
        ax=ax,
    )
    venn2_circles(
        subsets=subsets,
        linewidth=1.8,
        linestyle="solid",
        color="#222222",
        ax=ax,
    )

    for label in v.set_labels:
        if label is not None:
            label.set_fontsize(16)
            label.set_fontweight("bold")
            label.set_color("#111111")

    for sid in ("10", "01", "11"):
        lab = v.get_label_by_id(sid)
        if lab is not None:
            lab.set_fontsize(22)
            lab.set_fontweight("bold")
            lab.set_color("white")

    ax.set_title(
        f"{LIST_DISPLAY[list_key]} \u2014 overlap with Roth et al. (2023)",
        fontsize=16,
        fontweight="bold",
        color="#111111",
        pad=16,
    )

    union = len(our_genes | roth_genes)
    pct = (len(both) / union) if union else 0.0
    subtitle = (
        f"ours = {len(our_genes)}    |    Roth = {len(roth_genes)}    |    "
        f"shared = {len(both)}  ({pct:.0%} of union)"
    )
    ax.text(
        0.5, -0.02, subtitle,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11, color="#444444",
    )

    footer = (
        f"Roth bait scope: {bait_scope_label}  |  BFDR \u2264 {bfdr_cutoff:g}  |  "
        f"mouse\u2192human by uppercase shortcut"
    )
    ax.text(
        0.5, -0.07, footer,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9, style="italic", color="#777777",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _fmt_list(genes: list[str], per_line: int = 8) -> str:
    if not genes:
        return "_(none)_"
    chunks = []
    for i in range(0, len(genes), per_line):
        chunks.append(" ".join(f"`{g}`" for g in genes[i : i + per_line]))
    return "<br>".join(chunks)


def write_summary(
    out_dir: Path,
    *,
    bait_scope: str,
    bfdr_cutoff: float,
    folder: str,
    sheet: str | None,
    matched_baits: list[str],
    roth_genes: set[str],
    our_lists: dict[str, set[str]],
    overlaps: dict[str, set[str]],
    prey_to_baits: dict[str, set[str]],
) -> None:
    lines: list[str] = []
    lines.append(f"# Roth et al. (2023) comparison — bait scope: `{bait_scope}`\n")
    lines.append("## Parameters\n")
    lines.append(f"- Source: `{MMC3_PATH.name}`")
    lines.append(f"- Sheet: `{sheet or '(auto-detected)'}`")
    lines.append(f"- BFDR cutoff: `<= {bfdr_cutoff:g}`")
    lines.append(f"- Our folder: `{folder}`")
    lines.append(
        "- Orthology: `PreyGene.str.upper()` shortcut — relies on mouse/human "
        "symbols sharing spelling modulo case. A proper MGI\u2192HGNC ortholog "
        "map is a follow-up if known interactors are missing."
    )
    lines.append(f"- Roth baits included ({len(matched_baits)}): "
                 f"{', '.join(f'`{b}`' for b in matched_baits) if matched_baits else '_none matched_'}")
    lines.append(f"- Roth significant preys (post-filter, deduped): **{len(roth_genes):,}**")
    lines.append("")

    lines.append("## Overlap counts\n")
    lines.append("| Our list | Our size | Overlap with Roth | % of our list |")
    lines.append("|---|---:|---:|---:|")
    for key in ("common", "POS_specific", "NEG_specific"):
        n_ours = len(our_lists[key])
        n_over = len(overlaps[key])
        pct = (n_over / n_ours * 100) if n_ours else 0.0
        lines.append(f"| {LIST_DISPLAY[key]} | {n_ours} | {n_over} | {pct:.1f}% |")
    lines.append("")

    lines.append("## Overlapping genes\n")
    for key in ("common", "POS_specific", "NEG_specific"):
        genes = sorted(overlaps[key])
        lines.append(f"### {LIST_DISPLAY[key]} \u2229 Roth ({len(genes)} genes)\n")
        lines.append(_fmt_list(genes))
        lines.append("")

    lines.append("## Per-bait contribution\n")
    lines.append(
        "For each overlapping gene, which Roth bait(s) recovered it as a "
        "significant prey. Useful for pinning Ex15 interactors to specific "
        "regulators profiled by Roth.\n"
    )
    for key in ("common", "POS_specific", "NEG_specific"):
        genes = sorted(overlaps[key])
        if not genes:
            continue
        lines.append(f"### {LIST_DISPLAY[key]}\n")
        lines.append("| Gene | Roth bait(s) |")
        lines.append("|---|---|")
        for g in genes:
            baits = sorted(prey_to_baits.get(g, set()))
            lines.append(f"| `{g}` | {', '.join(baits) if baits else '_(unknown)_'} |")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_bait_scope_comparison(
    comparison_root: Path,
    overlaps_by_scope: dict[str, dict[str, set[str]]],
    *,
    bfdr_cutoff: float,
) -> None:
    """Top-level diff: which genes does the chtop/sap30bp filter remove vs all_baits?"""
    lines: list[str] = []
    lines.append("# Roth comparison: bait-scope differences\n")
    lines.append(
        f"Compares `all_baits/` (every Roth bait pooled, BFDR \u2264 {bfdr_cutoff:g}) against "
        f"`chtop_sap30bp/` (Chtop + Sap30bp only, same BFDR).\n"
    )
    lines.append(
        "Biological read: genes in `all_baits` but not in `chtop_sap30bp` overlap with "
        "the broader gene-expression regulator interactome but are *not* specifically "
        "recovered on the Chtop/Sap30bp axis. Use this to disentangle generic vs "
        "Chtop/Sap30bp-specific Ex15 interactors.\n"
    )

    if "all_baits" not in overlaps_by_scope or "chtop_sap30bp" not in overlaps_by_scope:
        lines.append(
            "_Note: only one bait scope was run; cross-scope diff not available. "
            "Re-run without `--bait-scope` to populate both sides._\n"
        )
        (comparison_root / "bait_scope_comparison.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        return

    all_b = overlaps_by_scope["all_baits"]
    cs = overlaps_by_scope["chtop_sap30bp"]

    lines.append("## Invariant check\n")
    lines.append(
        "`chtop_sap30bp` overlap MUST be a subset of `all_baits` overlap (the bait "
        "filter can only remove preys, never add them).\n"
    )
    lines.append("| List | Subset? | all_baits | chtop_sap30bp |")
    lines.append("|---|---|---:|---:|")
    any_violation = False
    for key in ("common", "POS_specific", "NEG_specific"):
        a = all_b[key]
        c = cs[key]
        subset_ok = c.issubset(a)
        any_violation = any_violation or not subset_ok
        lines.append(
            f"| {LIST_DISPLAY[key]} | "
            f"{'OK' if subset_ok else 'VIOLATED'} | {len(a)} | {len(c)} |"
        )
    lines.append("")

    lines.append("## Genes unique to each scope\n")
    for key in ("common", "POS_specific", "NEG_specific"):
        a = all_b[key]
        c = cs[key]
        only_all = sorted(a - c)
        only_cs = sorted(c - a)   # should always be empty if invariant holds
        shared = sorted(a & c)

        lines.append(f"### {LIST_DISPLAY[key]}\n")
        lines.append(
            f"- **Shared** between scopes ({len(shared)}): {_fmt_list(shared)}"
        )
        lines.append("")
        lines.append(
            f"- **Only in `all_baits`** (dropped by Chtop/Sap30bp filter) "
            f"({len(only_all)}): {_fmt_list(only_all)}"
        )
        lines.append("")
        if only_cs:
            lines.append(
                f"- **INVARIANT VIOLATION — in `chtop_sap30bp` but NOT `all_baits`** "
                f"({len(only_cs)}): {_fmt_list(only_cs)}"
            )
            lines.append("")

    (comparison_root / "bait_scope_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if any_violation:
        print(
            "  WARNING: subset invariant violated. Inspect bait_scope_comparison.md.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_scope(
    *,
    roth_df_full: pd.DataFrame,
    bait_scope: str,
    bfdr_cutoff: float,
    our_lists: dict[str, set[str]],
    scope_root: Path,
    folder_arg: str,
    sheet_arg: str | None,
) -> dict[str, set[str]]:
    """Returns {list_key: overlap_gene_set}."""
    print(f"\n[bait-scope: {bait_scope}]")
    scope_root.mkdir(parents=True, exist_ok=True)
    (scope_root / "figures").mkdir(exist_ok=True)

    filtered, matched_baits = filter_roth(
        roth_df_full, bfdr_cutoff=bfdr_cutoff, bait_scope=bait_scope
    )
    print(f"  {len(filtered):,} rows after BFDR + bait filter")
    print(f"  {len(matched_baits)} bait label(s) matched: {matched_baits}")

    roth_genes = set(filtered["PreyGene_upper"].dropna())
    roth_genes.discard("")
    print(f"  {len(roth_genes):,} unique Roth prey genes (uppercased)")

    (scope_root / "roth_significant_preys.txt").write_text(
        "\n".join(sorted(roth_genes)) + "\n", encoding="utf-8"
    )

    # PreyGene -> set(Baits) for the per-bait contribution table
    prey_to_baits: dict[str, set[str]] = (
        filtered.groupby("PreyGene_upper")["Bait"]
        .apply(lambda s: set(s.dropna()))
        .to_dict()
    )

    overlaps: dict[str, set[str]] = {}
    for key, genes in our_lists.items():
        overlap = genes & roth_genes
        overlaps[key] = overlap
        (scope_root / f"overlap_{key}.txt").write_text(
            "\n".join(sorted(overlap)) + "\n", encoding="utf-8"
        )
        png = scope_root / "figures" / f"venn_{key}_vs_roth.png"
        render_venn(
            genes, roth_genes, key, png,
            bait_scope_label=bait_scope, bfdr_cutoff=bfdr_cutoff,
        )
        print(
            f"  {LIST_DISPLAY[key]:20}  ours={len(genes):4}  "
            f"overlap={len(overlap):4}  -> {png.relative_to(HERE)}"
        )

    write_summary(
        scope_root,
        bait_scope=bait_scope,
        bfdr_cutoff=bfdr_cutoff,
        folder=folder_arg,
        sheet=sheet_arg,
        matched_baits=matched_baits,
        roth_genes=roth_genes,
        our_lists=our_lists,
        overlaps=overlaps,
        prey_to_baits=prey_to_baits,
    )
    print(f"  wrote summary.md")
    return overlaps


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--folder", default="isoforms_distinct",
                    help="subfolder containing the _clean_genes.txt files "
                         "(default: isoforms_distinct)")
    ap.add_argument("--bfdr", type=float, default=0.05,
                    help="SAINTexpress BFDR cutoff for 'significant' preys "
                         "(default: 0.05)")
    ap.add_argument("--bait-scope", choices=[*BAIT_SCOPES, "both"], default="both",
                    help="which bait scope(s) to run (default: both)")
    ap.add_argument("--sheet", default=None,
                    help="sheet name inside mmc3.xlsx (default: auto-detect by "
                         "matching 'SAINT' case-insensitively)")
    args = ap.parse_args()

    folder = HERE / args.folder
    if not folder.is_dir():
        sys.exit(f"ERROR: folder not found: {folder}")
    if not MMC3_PATH.exists():
        sys.exit(f"ERROR: Roth supplementary file not found: {MMC3_PATH}")

    print(f"Loading Roth data from {MMC3_PATH.name} ...")
    roth_df_full = load_roth(MMC3_PATH, args.sheet)

    our_lists: dict[str, set[str]] = {}
    for key, fname in LISTS.items():
        path = folder / fname
        if not path.exists():
            sys.exit(f"ERROR: missing input gene list: {path}")
        our_lists[key] = load_gene_list(path)
        print(f"  loaded {LIST_DISPLAY[key]:20}  {len(our_lists[key]):4} genes")

    comparison_root = folder / "roth_comparison"
    comparison_root.mkdir(exist_ok=True)

    scopes_to_run: tuple[str, ...] = (
        BAIT_SCOPES if args.bait_scope == "both" else (args.bait_scope,)
    )
    overlaps_by_scope: dict[str, dict[str, set[str]]] = {}
    for scope in scopes_to_run:
        overlaps_by_scope[scope] = run_one_scope(
            roth_df_full=roth_df_full,
            bait_scope=scope,
            bfdr_cutoff=args.bfdr,
            our_lists=our_lists,
            scope_root=comparison_root / scope,
            folder_arg=args.folder,
            sheet_arg=args.sheet,
        )

    write_bait_scope_comparison(
        comparison_root, overlaps_by_scope, bfdr_cutoff=args.bfdr
    )
    print(f"\nwrote {(comparison_root / 'bait_scope_comparison.md').relative_to(HERE)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
