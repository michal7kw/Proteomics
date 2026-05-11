# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "matplotlib-venn",
# ]
# ///
"""Venn diagram of SRRM3 Ex15 POS vs Ex15 NEG interactor lists.

Produces two output folders:
  - isoforms_distinct/   Q9UKV3-5 and Q9UKV3 treated as separate proteins
  - isoforms_collapsed/  Q9UKV3-5 collapsed to Q9UKV3
Each folder contains the Venn PNG and the three list files
(common / POS-specific / NEG-specific).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles

HERE = Path(__file__).parent
POS_FILE = HERE / "Ex15_POS.txt"
NEG_FILE = HERE / "Ex15_NEG.txt"


def parse_ids(path: Path, collapse_isoforms: bool) -> set[str]:
    """Extract UniProt accessions from a Scaffold-style protein list.

    Each line looks like  "Q9UKV3-5 [2]"  or  "Q93009 (+2)"  or just "Q6P2Q9".
    [n] (peptide count) and (+n) (protein-group extras) are Scaffold metadata
    and always dropped. If collapse_isoforms=True the -N isoform suffix is
    additionally stripped so Q9UKV3-5 -> Q9UKV3.
    """
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        acc = line.split()[0]
        if collapse_isoforms:
            acc = re.sub(r"-\d+$", "", acc)
        ids.add(acc)
    return ids


def render(
    pos: set[str],
    neg: set[str],
    out_dir: Path,
    *,
    collapse_isoforms: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    common = pos & neg
    pos_only = pos - neg
    neg_only = neg - pos

    mode = "collapsed" if collapse_isoforms else "distinct"
    print(f"\n[{mode}]")
    print(f"  Ex15_POS total:  {len(pos)}")
    print(f"  Ex15_NEG total:  {len(neg)}")
    print(f"  Common:          {len(common)}")
    print(f"  POS-specific:    {len(pos_only)}")
    print(f"  NEG-specific:    {len(neg_only)}")

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    fig.patch.set_facecolor("white")

    pos_color = "#E63946"
    neg_color = "#1D7874"

    subsets = (len(pos_only), len(neg_only), len(common))
    v = venn2(
        subsets=subsets,
        set_labels=("Ex15 POS", "Ex15 NEG"),
        set_colors=(pos_color, neg_color),
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
            label.set_fontsize(18)
            label.set_fontweight("bold")
            label.set_color("#111111")

    for sid in ("10", "01", "11"):
        lab = v.get_label_by_id(sid)
        if lab is not None:
            lab.set_fontsize(22)
            lab.set_fontweight("bold")
            lab.set_color("white")

    ax.set_title(
        "SRRM3 interactors \u2014 Ex15 POS vs Ex15 NEG",
        fontsize=17,
        fontweight="bold",
        color="#111111",
        pad=18,
    )

    subtitle = (
        f"POS = {len(pos)} proteins    |    NEG = {len(neg)} proteins    |    "
        f"shared = {len(common)}  "
        f"({len(common) / len(pos | neg):.0%} of union)"
    )
    ax.text(
        0.5, -0.02, subtitle,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11, color="#444444",
    )

    footer = (
        "Isoform suffixes collapsed (e.g. Q9UKV3-5 \u2192 Q9UKV3)"
        if collapse_isoforms else
        "Isoforms kept distinct (e.g. Q9UKV3-5 \u2260 Q9UKV3)"
    )
    ax.text(
        0.5, -0.07, footer,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9, style="italic", color="#777777",
    )

    plt.tight_layout()
    out_png = out_dir / "Ex15_POS_vs_NEG_venn.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (out_dir / "Ex15_common.txt").write_text(
        "\n".join(sorted(common)) + "\n", encoding="utf-8"
    )
    (out_dir / "Ex15_POS_specific.txt").write_text(
        "\n".join(sorted(pos_only)) + "\n", encoding="utf-8"
    )
    (out_dir / "Ex15_NEG_specific.txt").write_text(
        "\n".join(sorted(neg_only)) + "\n", encoding="utf-8"
    )
    print(f"  -> {out_dir.relative_to(HERE)}/  (PNG + 3 lists)")


def main() -> None:
    for collapse, folder in [
        (False, "isoforms_distinct"),
        (True,  "isoforms_collapsed"),
    ]:
        pos = parse_ids(POS_FILE, collapse_isoforms=collapse)
        neg = parse_ids(NEG_FILE, collapse_isoforms=collapse)
        render(pos, neg, HERE / folder, collapse_isoforms=collapse)


if __name__ == "__main__":
    main()
