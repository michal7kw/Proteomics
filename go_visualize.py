# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "matplotlib-venn>=1.1",
# ]
# ///
"""Publication-quality figures from go_enrichment.py output.

Reads <folder>/go_enrichment/gprofiler_raw.tsv and writes 4 PNGs to
<folder>/go_enrichment/figures/:

  01_dotplot_comparative.png   top terms x 3 lists, size=genes, color=-log10 p
  02_bars_GOBP_by_list.png     top-15 GO:BP per list (1x3 subplots)
  03_bars_CORUM_by_list.png    top-12 CORUM complexes per list
  04_upset_terms.png           UpSet of significant term sets across lists

Usage:
  uv run go_visualize.py
  uv run go_visualize.py --folder isoforms_collapsed
  uv run go_visualize.py --only dotplot bars_gobp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib_venn import venn3, venn3_circles

HERE = Path(__file__).parent

# Palette - mirrors venn_diagram.py
POS_COLOR = "#E63946"
NEG_COLOR = "#1D7874"
COMMON_COLOR = "#5A5A5A"

LIST_ORDER = ["Ex15_common", "Ex15_POS_specific", "Ex15_NEG_specific"]
LIST_LABEL = {
    "Ex15_common":       "Common",
    "Ex15_POS_specific": "POS-specific",
    "Ex15_NEG_specific": "NEG-specific",
}
LIST_COLOR = {
    "Ex15_common":       COMMON_COLOR,
    "Ex15_POS_specific": POS_COLOR,
    "Ex15_NEG_specific": NEG_COLOR,
}

# Publication-ready rcParams
DPI = 300
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize":   18,
    "axes.labelsize":   14,
    "axes.labelweight": "semibold",
    "xtick.labelsize":  13,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 20,
    "figure.titleweight": "bold",
    "axes.linewidth":   1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "savefig.facecolor": "white",
    "figure.facecolor":  "white",
    "savefig.dpi":       DPI,
})


def load_results(folder: Path) -> pd.DataFrame:
    path = folder / "go_enrichment" / "gprofiler_raw.tsv"
    if not path.exists():
        sys.exit(
            f"ERROR: {path} not found.\n"
            f"Run:  uv run go_enrichment.py --folder {folder.name}  first."
        )
    df = pd.read_csv(path, sep="\t")
    # restore list from '|'-joined intersections
    if "intersections" in df.columns:
        df["intersections"] = df["intersections"].fillna("").apply(
            lambda s: [g for g in str(s).split("|") if g]
        )
    # numeric columns
    for c in ("p_value", "intersection_size", "term_size", "query_size"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))
    # Shorten very long term names for plotting
    df["short_name"] = df["name"].astype(str).apply(
        lambda s: (s[:60] + "\u2026") if len(s) > 60 else s
    )
    return df


# ---------------------------------------------------------------------------
# 1) Comparative dot plot (combined + per-source standalones)
# ---------------------------------------------------------------------------

def _select_top_terms(
    df: pd.DataFrame, source: str, top_n: int
) -> pd.DataFrame:
    src_df = df[df["source"] == source]
    top_terms: set[str] = set()
    for q in LIST_ORDER:
        q_df = src_df[src_df["query"] == q].nsmallest(top_n, "p_value")
        top_terms.update(q_df["name"].astype(str))
    if not top_terms:
        return src_df.iloc[0:0]
    return src_df[src_df["name"].astype(str).isin(top_terms)].copy()


def _term_order(sub: pd.DataFrame) -> list[str]:
    return (
        sub.groupby("name")["neg_log10_p"].mean()
        .sort_values(ascending=True)  # strongest on top (inverted y)
        .index.tolist()
    )


def _draw_dot_panel(
    ax,
    panel_df: pd.DataFrame,
    terms: list[str],
    norm: Normalize,
    cmap,
    size_fn,
    label_max: int = 62,
) -> None:
    y_pos = {t: i for i, t in enumerate(terms)}
    for q_i, q in enumerate(LIST_ORDER):
        q_sub = panel_df[panel_df["query"] == q]
        if q_sub.empty:
            continue
        xs = np.full(len(q_sub), q_i, dtype=float)
        ys = np.array([y_pos[t] for t in q_sub["name"].astype(str)])
        colors = cmap(norm(q_sub["neg_log10_p"].values))
        sizes = [size_fn(n) for n in q_sub["intersection_size"].values]
        ax.scatter(
            xs, ys, s=sizes, c=colors,
            edgecolor="#222", linewidth=0.7, alpha=0.95, zorder=3,
        )
    short = [(t[:label_max] + "\u2026") if len(t) > label_max else t for t in terms]
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(short)
    ax.set_xticks(range(len(LIST_ORDER)))
    ax.set_xticklabels([LIST_LABEL[q] for q in LIST_ORDER], fontweight="bold")
    ax.set_xlim(-0.5, len(LIST_ORDER) - 0.5)
    ax.set_ylim(-0.5, max(0, len(terms) - 0.5))
    ax.grid(True, axis="both", color="#DDDDDD", linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_comparative_dotplot(
    df: pd.DataFrame,
    out_path: Path,
    sources: tuple[str, ...] = ("GO:BP", "CORUM", "REAC"),
    top_n: int = 10,
) -> None:
    """Combined multi-panel dot plot + one standalone PNG per source."""
    sources = tuple(s for s in sources if (df["source"] == s).any())
    if not sources:
        print("  dotplot: none of the requested sources present; skipping")
        return

    picked = {src: _select_top_terms(df, src, top_n) for src in sources}
    picked = {s: d for s, d in picked.items() if not d.empty}
    if not picked:
        print("  dotplot: no data to plot; skipping")
        return

    # Shared color/size scales across ALL panels so standalones match the combined
    plot_df = pd.concat(picked.values(), ignore_index=True)
    vmax = float(plot_df["neg_log10_p"].max()) if len(plot_df) else 1.0
    norm = Normalize(vmin=0, vmax=max(vmax, 2.0))
    cmap = plt.get_cmap("magma_r")
    size_min, size_max = 60, 650
    max_hits = max(1, int(plot_df["intersection_size"].max()))

    def size_fn(n: float) -> float:
        return size_min + (size_max - size_min) * (n / max_hits)

    legend_sizes = sorted({1, max(1, max_hits // 4), max(1, max_hits // 2), max_hits})

    # ---------- combined figure ----------
    _render_combined(
        out_path, picked, norm, cmap, size_fn, legend_sizes, max_hits, top_n,
    )

    # ---------- standalones, one per source ----------
    out_dir = out_path.parent
    for src, sub in picked.items():
        slug = src.replace(":", "").replace("/", "_")
        standalone = out_dir / f"01_dotplot_{slug}.png"
        _render_standalone(
            standalone, src, sub, norm, cmap, size_fn, legend_sizes, top_n,
        )


def _render_combined(
    out_path: Path,
    panels: dict[str, pd.DataFrame],
    norm: Normalize,
    cmap,
    size_fn,
    legend_sizes: list[int],
    max_hits: int,
    top_n: int,
) -> None:
    n = len(panels)
    fig_w = 7.0 * n + 2.2
    fig_h = max(9.0, 0.58 * top_n * 3) + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(
        1, n + 1,
        width_ratios=[10] * n + [0.6],
        left=0.07, right=0.96, top=0.90, bottom=0.14, wspace=0.85,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cbar_ax = fig.add_subplot(gs[0, -1])

    for ax, (src, sub) in zip(axes, panels.items()):
        terms = _term_order(sub)
        _draw_dot_panel(ax, sub, terms, norm, cmap, size_fn)
        ax.set_title(src, pad=14)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(r"$-\log_{10}$ adj $p$")
    cbar.ax.tick_params(labelsize=12)

    handles = [
        plt.scatter([], [], s=size_fn(nn), c="#aaa", edgecolor="#222",
                    linewidth=0.7, label=str(nn))
        for nn in legend_sizes
    ]
    fig.legend(
        handles=handles,
        title="intersection size (gene hits)",
        loc="lower center",
        frameon=False,
        ncol=len(legend_sizes),
        bbox_to_anchor=(0.5, 0.01),
        handletextpad=0.7,
        columnspacing=2.2,
    )
    fig.suptitle(
        "SRRM3 Ex15 interactome \u2014 comparative enrichment",
        y=0.975,
    )
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def _render_standalone(
    out_path: Path,
    source: str,
    sub: pd.DataFrame,
    norm: Normalize,
    cmap,
    size_fn,
    legend_sizes: list[int],
    top_n: int,
) -> None:
    terms = _term_order(sub)
    fig_h = max(7.5, 0.55 * len(terms)) + 1.2
    fig = plt.figure(figsize=(10.5, fig_h), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[14, 0.5],
        left=0.28, right=0.90, top=0.90, bottom=0.15, wspace=0.18,
    )
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])

    _draw_dot_panel(ax, sub, terms, norm, cmap, size_fn, label_max=70)
    ax.set_title(f"SRRM3 Ex15 interactome \u2014 {source}", pad=16)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(r"$-\log_{10}$ adj $p$")
    cbar.ax.tick_params(labelsize=12)

    handles = [
        plt.scatter([], [], s=size_fn(nn), c="#aaa", edgecolor="#222",
                    linewidth=0.7, label=str(nn))
        for nn in legend_sizes
    ]
    fig.legend(
        handles=handles,
        title="intersection size (gene hits)",
        loc="lower center",
        frameon=False,
        ncol=len(legend_sizes),
        bbox_to_anchor=(0.55, 0.01),
        handletextpad=0.7,
        columnspacing=2.2,
    )
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# 2 + 3) Faceted horizontal bars
# ---------------------------------------------------------------------------

def _plot_faceted_bars(
    df: pd.DataFrame,
    out_path: Path,
    source: str,
    top_n: int,
    title: str,
) -> None:
    sub = df[df["source"] == source].copy()
    if sub.empty:
        print(f"  bars ({source}): no rows; skipping")
        return

    fig, axes = plt.subplots(
        1, 3, figsize=(22, max(6.5, 0.55 * top_n) + 0.8),
        dpi=DPI, sharex=False,
    )

    for ax, list_name in zip(axes, LIST_ORDER):
        q = sub[sub["query"] == list_name].nsmallest(top_n, "p_value").copy()
        if q.empty:
            ax.text(0.5, 0.5, f"no significant {source} terms",
                    ha="center", va="center", fontsize=14, color="#888",
                    transform=ax.transAxes)
            ax.set_title(LIST_LABEL[list_name], color=LIST_COLOR[list_name])
            ax.axis("off")
            continue
        q = q.sort_values("neg_log10_p", ascending=True)
        labels = [
            (n[:55] + "\u2026") if len(n) > 55 else n
            for n in q["name"].astype(str)
        ]
        ys = np.arange(len(q))
        ax.barh(
            ys, q["neg_log10_p"],
            color=LIST_COLOR[list_name], edgecolor="#222", linewidth=0.8,
            alpha=0.92,
        )
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_xlabel(r"$-\log_{10}$ adj $p$")
        ax.set_title(
            f"{LIST_LABEL[list_name]}  ({len(q)} shown)",
            color=LIST_COLOR[list_name], pad=12,
        )
        # gene-count annotations at end of bars
        xmax = q["neg_log10_p"].max()
        for y, (_, row) in zip(ys, q.iterrows()):
            ax.text(
                row["neg_log10_p"] + xmax * 0.012, y,
                f"{int(row.get('intersection_size', 0))}",
                va="center", fontsize=11, color="#333", fontweight="semibold",
            )
        # Give the annotations some headroom
        ax.set_xlim(right=xmax * 1.10 + 0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="x", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_bars_gobp(df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    _plot_faceted_bars(
        df, out_path, source="GO:BP", top_n=top_n,
        title="Top GO Biological Process terms per list",
    )


def plot_bars_corum(df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    _plot_faceted_bars(
        df, out_path, source="CORUM", top_n=top_n,
        title="Top CORUM protein complexes per list",
    )


# ---------------------------------------------------------------------------
# 4) Term-level Venn3 (mirrors the gene-level Venn but on enriched terms)
# ---------------------------------------------------------------------------

def plot_term_venn(df: pd.DataFrame, out_path: Path, adj_p_cutoff: float = 0.05) -> None:
    sig = df[df["p_value"] <= adj_p_cutoff]
    term_sets: dict[str, set[str]] = {}
    for q in LIST_ORDER:
        term_sets[q] = set(sig[sig["query"] == q]["native"].astype(str))
        print(f"    {LIST_LABEL[q]}: {len(term_sets[q])} significant terms")

    if not any(term_sets.values()):
        print("  venn: nothing significant; skipping")
        return

    sets = [term_sets[q] for q in LIST_ORDER]
    labels = tuple(LIST_LABEL[q] for q in LIST_ORDER)
    colors = (COMMON_COLOR, POS_COLOR, NEG_COLOR)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=DPI, facecolor="white")

    v = venn3(
        subsets=sets, set_labels=labels, set_colors=colors, alpha=0.55, ax=ax,
    )
    venn3_circles(
        subsets=sets, linewidth=2.0, linestyle="solid", color="#222", ax=ax,
    )

    for lbl in v.set_labels:
        if lbl is not None:
            lbl.set_fontsize(18)
            lbl.set_fontweight("bold")
            lbl.set_color("#111")

    for sid in ("100", "010", "001", "110", "101", "011", "111"):
        sub_lbl = v.get_label_by_id(sid)
        if sub_lbl is not None:
            sub_lbl.set_fontsize(16)
            sub_lbl.set_fontweight("bold")
            sub_lbl.set_color("white")

    ax.set_title(
        f"Enriched terms shared / unique across lists  (adj $p$ < {adj_p_cutoff})",
        pad=18,
    )

    subtitle = (
        f"Common: {len(term_sets['Ex15_common'])}   |   "
        f"POS-specific: {len(term_sets['Ex15_POS_specific'])}   |   "
        f"NEG-specific: {len(term_sets['Ex15_NEG_specific'])}   "
        f"(union: {len(set().union(*sets))})"
    )
    ax.text(
        0.5, -0.02, subtitle,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=13, color="#333",
    )
    ax.text(
        0.5, -0.06,
        "Term IDs pooled across GO:BP/MF/CC, KEGG, Reactome, WikiPathways, CORUM",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11, style="italic", color="#777",
    )

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

PLOTS = {
    "dotplot":    ("01_dotplot_comparative.png",
                   lambda df, p: plot_comparative_dotplot(df, p)),
    "bars_gobp":  ("02_bars_GOBP_by_list.png",
                   lambda df, p: plot_bars_gobp(df, p)),
    "bars_corum": ("03_bars_CORUM_by_list.png",
                   lambda df, p: plot_bars_corum(df, p)),
    "venn":       ("04_venn_terms.png",
                   lambda df, p: plot_term_venn(df, p)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="isoforms_distinct")
    ap.add_argument("--only", nargs="+", choices=list(PLOTS.keys()), default=None,
                    help="regenerate only selected plots")
    args = ap.parse_args()

    folder = HERE / args.folder
    if not folder.is_dir():
        sys.exit(f"ERROR: folder not found: {folder}")

    df = load_results(folder)
    print(f"loaded {len(df)} rows from {folder.name}/go_enrichment/gprofiler_raw.tsv")

    out_dir = folder / "go_enrichment" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = args.only or list(PLOTS.keys())
    for key in selected:
        fname, fn = PLOTS[key]
        print(f"\n[{key}]")
        try:
            fn(df, out_dir / fname)
        except Exception as exc:  # keep going on plot-specific failures
            print(f"  {key} FAILED: {exc.__class__.__name__}: {exc}")

    print(f"\nAll figures in {out_dir}")


if __name__ == "__main__":
    main()
