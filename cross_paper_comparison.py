# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib",
#   "matplotlib-venn",
# ]
# ///
"""Cross-paper evidence matrix for Ex15 interactors.

Combines the outputs of `roth_comparison.py` and `mabin_comparison.py` into a
single per-gene evidence table: for every Ex15 gene, which literature dataset(s)
independently recover it. This is the correct way to compare Roth (2023) and
Mabin (2018) given that the two papers ask different biological questions:
Roth profiled 17 general gene-expression regulators (transcription, chromatin,
splicing) in mouse N2a cells; Mabin profiled 3 EJC components (MAGOH, CASC3,
RNPS1) in human HEK293. Direct overlap-count comparisons between them are
misleading; *per-gene support profiles* are not.

Tier categorization (per Ex15 gene):

  Tier A  - supported by BOTH papers        (strongest literature backing)
  Tier B  - Mabin only                      (EJC/splicing/NMD-flavored interactor)
  Tier C  - Roth only                       (transcription/chromatin-flavored)
  Tier D  - neither                         (Ex15-exclusive or cell-type-specific)

Inputs (expected to exist — run the two upstream scripts first):
  <folder>/Ex15_{common,POS_specific,NEG_specific}_clean_genes.txt
  <folder>/roth_comparison/all_baits/roth_significant_preys.txt
  <folder>/roth_comparison/chtop_sap30bp/roth_significant_preys.txt
  <folder>/mabin_comparison/all_ejcs/mabin_interactors_genes.txt
  <folder>/mabin_comparison/alternate_only/mabin_interactors_genes.txt

Outputs (under `<folder>/cross_paper_comparison/`):
  evidence_matrix_{common,POS_specific,NEG_specific}.tsv   per-list tables
  evidence_matrix_all.tsv                                   combined, one row per (gene, list)
  figures/venn_{common,POS_specific,NEG_specific}_tripartite.png   3-way Venn
  summary.md                                                tier counts + tier-B / tier-C gene lists

Usage:
  uv run cross_paper_comparison.py
  uv run cross_paper_comparison.py --folder isoforms_collapsed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3, venn3_circles

HERE = Path(__file__).parent

LISTS = {
    "common":       "Ex15_common_clean_genes.txt",
    "POS_specific": "Ex15_POS_specific_clean_genes.txt",
    "NEG_specific": "Ex15_NEG_specific_clean_genes.txt",
}

LIST_DISPLAY = {
    "common":       "Ex15 common",
    "POS_specific": "Ex15 POS specific",
    "NEG_specific": "Ex15 NEG specific",
}

LIST_COLORS = {
    "common":       "#6A4C93",
    "POS_specific": "#E63946",
    "NEG_specific": "#1D7874",
}
ROTH_COLOR = "#457B9D"
MABIN_COLOR = "#D08C34"

EVIDENCE_COLS = (
    "in_roth_all_baits",
    "in_roth_chtop_sap30bp",
    "in_mabin_all_ejcs",
    "in_mabin_alternate_only",
)

TIER_DESC = {
    "A": "both papers",
    "B": "Mabin only (EJC/splicing flavored)",
    "C": "Roth only (transcription/chromatin flavored)",
    "D": "neither (Ex15-exclusive or cell-type-specific)",
}


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_gene_list(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def load_reference_sets(folder: Path) -> dict[str, set[str]]:
    """Return {evidence_column_name: set_of_genes} from upstream script outputs."""
    sources = {
        "in_roth_all_baits":        folder / "roth_comparison"  / "all_baits"       / "roth_significant_preys.txt",
        "in_roth_chtop_sap30bp":    folder / "roth_comparison"  / "chtop_sap30bp"   / "roth_significant_preys.txt",
        "in_mabin_all_ejcs":        folder / "mabin_comparison" / "all_ejcs"        / "mabin_interactors_genes.txt",
        "in_mabin_alternate_only":  folder / "mabin_comparison" / "alternate_only"  / "mabin_interactors_genes.txt",
    }
    missing = [p for p in sources.values() if not p.exists()]
    if missing:
        lines = [f"ERROR: required upstream output files not found:"]
        for p in missing:
            lines.append(f"  {p}")
        lines.append("")
        lines.append("Run the upstream scripts first:")
        lines.append("  uv run roth_comparison.py")
        lines.append("  uv run mabin_comparison.py")
        sys.exit("\n".join(lines))

    return {name: load_gene_list(path) for name, path in sources.items()}


# ---------------------------------------------------------------------------
# Evidence matrix construction
# ---------------------------------------------------------------------------

def build_matrix(
    our_genes: set[str],
    refs: dict[str, set[str]],
    list_key: str,
) -> pd.DataFrame:
    rows = []
    for g in sorted(our_genes):
        row = {"gene": g, "ex15_list": list_key}
        for col in EVIDENCE_COLS:
            row[col] = g in refs[col]
        row["n_roth_scopes"] = int(row["in_roth_all_baits"]) + int(row["in_roth_chtop_sap30bp"])
        row["n_mabin_scopes"] = int(row["in_mabin_all_ejcs"]) + int(row["in_mabin_alternate_only"])
        row["in_roth"] = row["n_roth_scopes"] > 0
        row["in_mabin"] = row["n_mabin_scopes"] > 0
        if row["in_roth"] and row["in_mabin"]:
            row["tier"] = "A"
        elif row["in_mabin"]:
            row["tier"] = "B"
        elif row["in_roth"]:
            row["tier"] = "C"
        else:
            row["tier"] = "D"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering — 3-way Venn
# ---------------------------------------------------------------------------

def render_tripartite_venn(
    our_genes: set[str],
    roth_genes: set[str],
    mabin_genes: set[str],
    list_key: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)
    fig.patch.set_facecolor("white")

    # venn3 wants 7 subset sizes in a specific order: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    # where A=our, B=Roth, C=Mabin
    A, B, C = our_genes, roth_genes, mabin_genes
    subsets = (
        len(A - B - C),   # Abc: ours only
        len(B - A - C),   # aBc: Roth only
        len(A & B - C),   # ABc
        len(C - A - B),   # abC: Mabin only
        len(A & C - B),   # AbC
        len(B & C - A),   # aBC
        len(A & B & C),   # ABC
    )
    v = venn3(
        subsets=subsets,
        set_labels=(LIST_DISPLAY[list_key], "Roth et al. 2023", "Mabin et al. 2018"),
        set_colors=(LIST_COLORS[list_key], ROTH_COLOR, MABIN_COLOR),
        alpha=0.55,
        ax=ax,
    )
    venn3_circles(
        subsets=subsets,
        linewidth=1.6,
        linestyle="solid",
        color="#222222",
        ax=ax,
    )
    for label in v.set_labels or []:
        if label is not None:
            label.set_fontsize(14)
            label.set_fontweight("bold")
            label.set_color("#111111")
    for sid in ("100", "010", "001", "110", "101", "011", "111"):
        lab = v.get_label_by_id(sid)
        if lab is not None:
            lab.set_fontsize(18)
            lab.set_fontweight("bold")
            lab.set_color("white")

    ax.set_title(
        f"{LIST_DISPLAY[list_key]} \u2014 three-way literature evidence",
        fontsize=16,
        fontweight="bold",
        color="#111111",
        pad=16,
    )
    both_papers = len(A & B & C)
    subtitle = (
        f"ours = {len(A)}    |    Roth (all baits) = {len(B)}    |    "
        f"Mabin (all EJCs) = {len(C)}    |    in ALL three = {both_papers}"
    )
    ax.text(
        0.5, -0.02, subtitle,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11, color="#444444",
    )
    footer = (
        "Roth = BFDR \u2264 0.05 across 17 regulators (N2a mouse, uppercased)  |  "
        "Mabin = both replicates in any of MAGOH/CASC3/RNPS1 (HEK293 human)"
    )
    ax.text(
        0.5, -0.06, footer,
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

def _fmt_list(genes: list[str], per_line: int = 10) -> str:
    if not genes:
        return "_(none)_"
    chunks = []
    for i in range(0, len(genes), per_line):
        chunks.append(" ".join(f"`{g}`" for g in genes[i : i + per_line]))
    return "<br>".join(chunks)


def write_summary(out_dir: Path, matrices: dict[str, pd.DataFrame], folder_name: str) -> None:
    lines: list[str] = []
    lines.append("# Cross-paper evidence summary — Roth (2023) vs Mabin (2018)\n")
    lines.append(f"Folder: `{folder_name}`\n")
    lines.append(
        "Each Ex15 gene is classified into one of four evidence tiers based on "
        "whether it appears as a high-confidence interactor in Roth's "
        "17-regulator AP-MS (BFDR \u2264 0.05), Mabin's EJC IPs (both replicates), "
        "both, or neither. The two papers ask different biological questions "
        "(broad gene-expression regulators vs alternate-EJC composition), so "
        "the tier assignment is more informative than any single overlap count.\n"
    )

    lines.append("## Tier counts per Ex15 list\n")
    lines.append("| List | Size | A: both | B: Mabin only | C: Roth only | D: neither |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ("common", "POS_specific", "NEG_specific"):
        df = matrices[key]
        total = len(df)
        counts = df["tier"].value_counts().to_dict()
        row = (
            f"| {LIST_DISPLAY[key]} | {total} | "
            f"{counts.get('A', 0)} | {counts.get('B', 0)} | "
            f"{counts.get('C', 0)} | {counts.get('D', 0)} |"
        )
        lines.append(row)
    lines.append("")

    lines.append("## Tier A — supported by BOTH Roth and Mabin\n")
    lines.append(
        "Strongest literature backing: these Ex15 interactors are recovered as "
        "high-confidence preys in two independent proteomic datasets across "
        "different bait scopes and cell types. Very likely real.\n"
    )
    for key in ("common", "POS_specific", "NEG_specific"):
        df = matrices[key]
        tier_a = sorted(df[df["tier"] == "A"]["gene"].tolist())
        lines.append(f"### {LIST_DISPLAY[key]} ({len(tier_a)} genes)\n")
        lines.append(_fmt_list(tier_a))
        lines.append("")

    lines.append("## Tier B — Mabin only (EJC/splicing/NMD-flavored)\n")
    lines.append(
        "Recovered in Mabin's EJC IPs but not significant in any Roth bait. "
        "Consistent with an interaction that lives in the post-splicing / "
        "mRNA-export / NMD machinery — the biology Roth's general-regulator "
        "panel doesn't directly sample.\n"
    )
    for key in ("common", "POS_specific", "NEG_specific"):
        df = matrices[key]
        tier_b = sorted(df[df["tier"] == "B"]["gene"].tolist())
        lines.append(f"### {LIST_DISPLAY[key]} ({len(tier_b)} genes)\n")
        lines.append(_fmt_list(tier_b))
        lines.append("")

    lines.append("## Tier C — Roth only (transcription/chromatin-flavored)\n")
    lines.append(
        "Recovered in Roth's general-regulator AP-MS but not in any Mabin EJC "
        "IP. Suggests co-transcriptional or chromatin-coupled interaction — "
        "not a stable post-splicing EJC partner.\n"
    )
    for key in ("common", "POS_specific", "NEG_specific"):
        df = matrices[key]
        tier_c = sorted(df[df["tier"] == "C"]["gene"].tolist())
        lines.append(f"### {LIST_DISPLAY[key]} ({len(tier_c)} genes)\n")
        lines.append(_fmt_list(tier_c))
        lines.append("")

    lines.append("## Tier D — neither\n")
    lines.append(
        "Not recovered in either literature dataset. Candidates for genuinely "
        "novel Ex15-specific interactions, HEK293-specific biology, or "
        "interactions that would require a different bait panel to detect. "
        "Numeric counts only (lists are large); full per-gene assignments are "
        "in the TSV files.\n"
    )
    lines.append("| List | Tier D count |")
    lines.append("|---|---:|")
    for key in ("common", "POS_specific", "NEG_specific"):
        df = matrices[key]
        n = int((df["tier"] == "D").sum())
        lines.append(f"| {LIST_DISPLAY[key]} | {n} |")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--folder", default="isoforms_distinct",
                    help="subfolder containing the _clean_genes.txt files and "
                         "the roth_comparison/ and mabin_comparison/ outputs "
                         "(default: isoforms_distinct)")
    args = ap.parse_args()

    folder = HERE / args.folder
    if not folder.is_dir():
        sys.exit(f"ERROR: folder not found: {folder}")

    print(f"Loading reference sets from {folder.relative_to(HERE)}/ ...")
    refs = load_reference_sets(folder)
    for name, s in refs.items():
        print(f"  {name:28}  {len(s):4} genes")

    our_lists: dict[str, set[str]] = {}
    for key, fname in LISTS.items():
        path = folder / fname
        if not path.exists():
            sys.exit(f"ERROR: missing input gene list: {path}")
        our_lists[key] = load_gene_list(path)
        print(f"  {LIST_DISPLAY[key]:20}  {len(our_lists[key]):4} genes")

    out_dir = folder / "cross_paper_comparison"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    matrices: dict[str, pd.DataFrame] = {}
    for key, genes in our_lists.items():
        df = build_matrix(genes, refs, key)
        matrices[key] = df

        tsv_path = out_dir / f"evidence_matrix_{key}.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)

        counts = df["tier"].value_counts().reindex(["A", "B", "C", "D"]).fillna(0).astype(int)
        print(
            f"\n[{LIST_DISPLAY[key]}]  total={len(df)}  "
            f"A={counts['A']}  B={counts['B']}  C={counts['C']}  D={counts['D']}"
        )
        print(f"  -> {tsv_path.relative_to(HERE)}")

        png = out_dir / "figures" / f"venn_{key}_tripartite.png"
        render_tripartite_venn(
            our_genes=genes,
            roth_genes=refs["in_roth_all_baits"],
            mabin_genes=refs["in_mabin_all_ejcs"],
            list_key=key,
            out_path=png,
        )
        print(f"  -> {png.relative_to(HERE)}")

    combined = pd.concat(matrices.values(), ignore_index=True)
    combined_path = out_dir / "evidence_matrix_all.tsv"
    combined.to_csv(combined_path, sep="\t", index=False)
    print(f"\nwrote combined matrix: {combined_path.relative_to(HERE)}  "
          f"({len(combined)} rows)")

    write_summary(out_dir, matrices, args.folder)
    print(f"wrote {(out_dir / 'summary.md').relative_to(HERE)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
