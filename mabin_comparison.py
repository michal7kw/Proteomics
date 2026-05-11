# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "openpyxl",
#   "matplotlib",
#   "matplotlib-venn",
#   "httpx",
# ]
# ///
"""Compare Ex15 POS/NEG/common interactor lists against the Mabin et al. 2018 alternate-EJC proteome.

Source: Mabin et al., Cell Reports 2018 (PMID 30485799), supplementary Table S1
(`mmc2.xlsx`). The table reports proteins co-purified with three EJC bait
IPs — FLAG-MAGOH (core EJC), FLAG-RNPS1 (alternate EJC flavor A), FLAG-CASC3
(alternate EJC flavor B) — across two biological replicates, as log2(NSAF)
values with zero = not detected in that IP. The paper pre-filters to >2-fold
enrichment over FLAG-only; each row is a real hit in at least one bait.

This script treats a protein as a high-confidence bait interactor when it is
detected (log2(NSAF) > 0) in that bait's column in BOTH replicates — the
standard publication-quality two-replicate intersection.

Gene-symbol mapping. Accessions are UniProt *entry names* (e.g. `SAP18_HUMAN`,
`IF4A3_HUMAN`, `ACINU_HUMAN`). The prefix before `_HUMAN` is NOT reliably the
HGNC symbol (`IF4A3_HUMAN` -> `EIF4A3`, `PININ_HUMAN` -> `PNN`, `ACINU_HUMAN`
-> `ACIN1`, `MGN_HUMAN` -> `MAGOHB`). We therefore resolve each entry name to
its primary gene symbol via the UniProt REST API and cache the result in
`.mabin_uniprot_cache.tsv` so re-runs are offline.

Outputs (under `<folder>/mabin_comparison/`):
  all_ejcs/           union of MAGOH + CASC3 + RNPS1 high-confidence interactors
  alternate_only/     union of CASC3 + RNPS1 (excludes the core-EJC MAGOH column)
  bait_scope_comparison.md   diff between the two scopes

Usage:
  uv run mabin_comparison.py
  uv run mabin_comparison.py --bait-scope all_ejcs
  uv run mabin_comparison.py --bait-scope alternate_only
  uv run mabin_comparison.py --replicate-rule union     # present in EITHER rep (less strict)
  uv run mabin_comparison.py --folder isoforms_collapsed
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2, venn2_circles

HERE = Path(__file__).parent
MMC2_PATH = HERE / "mmc2.xlsx"
UNIPROT_CACHE = HERE / ".mabin_uniprot_cache.tsv"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"

SHEET_REP1 = "replicate 1"
SHEET_REP2 = "replicate 2"
SKIP_META_ROWS = 7                    # metadata block above the header row
ACCESSION_COL = "Accession Number"
BAIT_COLS = ("MAGOH", "CASC3", "RNPS1")

LISTS = {
    "common":       "Ex15_common_clean_genes.txt",
    "POS_specific": "Ex15_POS_specific_clean_genes.txt",
    "NEG_specific": "Ex15_NEG_specific_clean_genes.txt",
}

POS_COLOR = "#E63946"
NEG_COLOR = "#1D7874"
COMMON_COLOR = "#6A4C93"
MABIN_COLOR = "#D08C34"               # warm ochre — distinct from Roth's blue

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

BAIT_SCOPES = {
    "all_ejcs":       ("MAGOH", "CASC3", "RNPS1"),
    "alternate_only": ("CASC3", "RNPS1"),
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gene_list(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def load_mabin_sheet(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    """Load one replicate sheet, strip protein-group suffixes, validate columns."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet, skiprows=SKIP_META_ROWS)
    missing = {ACCESSION_COL, *BAIT_COLS} - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: sheet '{sheet}' missing columns {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=[ACCESSION_COL]).copy()
    # Accession cells look like 'SAP18_HUMAN' or 'H2A1D_HUMAN (+6)'.
    # Drop the '(+N)' protein-group suffix — same pattern as clean_lists.py.
    df["entry_name"] = (
        df[ACCESSION_COL].astype(str).str.strip().str.split(r"\s+", n=1).str[0]
    )
    # Coerce bait columns to numeric; the log2(NSAF) cells are numeric already
    # but occasional '0' strings appear in some rows.
    for bait in BAIT_COLS:
        df[bait] = pd.to_numeric(df[bait], errors="coerce").fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# UniProt entry name -> gene symbol lookup (with disk cache)
# ---------------------------------------------------------------------------

def _read_cache() -> dict[str, str]:
    if not UNIPROT_CACHE.exists():
        return {}
    cache: dict[str, str] = {}
    for line in UNIPROT_CACHE.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            cache[k.strip()] = v.strip()
    return cache


def _write_cache(cache: dict[str, str]) -> None:
    body = "\n".join(f"{k}\t{v}" for k, v in sorted(cache.items()))
    UNIPROT_CACHE.write_text(body + "\n", encoding="utf-8")


def map_entry_names_to_genes(entry_names: list[str]) -> dict[str, str]:
    """Return {entry_name: primary gene symbol}, using cached UniProt lookups."""
    cache = _read_cache()
    to_fetch = [n for n in entry_names if n not in cache]
    if not to_fetch:
        print(f"  UniProt cache hit: all {len(entry_names)} entry names")
        return {n: cache[n] for n in entry_names}

    print(f"  UniProt lookup: {len(to_fetch)} new entry names "
          f"(cached: {len(entry_names) - len(to_fetch)})")

    BATCH = 20   # keep URL length reasonable with OR-joined queries
    with httpx.Client(timeout=60.0) as client:
        for i in range(0, len(to_fetch), BATCH):
            batch = to_fetch[i : i + BATCH]
            query = " OR ".join(f"id:{n}" for n in batch)
            params = {
                "query": query,
                "fields": "id,gene_primary",
                "format": "tsv",
                "size": len(batch) + 5,
            }
            try:
                r = client.get(UNIPROT_URL, params=params)
                r.raise_for_status()
            except httpx.HTTPError as e:
                print(f"    batch {i // BATCH + 1} failed: {e}", file=sys.stderr)
                # Fallback: use the prefix of the entry name (best-effort, imperfect).
                for n in batch:
                    cache.setdefault(n, n.replace("_HUMAN", ""))
                continue
            lines = r.text.strip().splitlines()
            # Header: "Entry Name\tGene Names (primary)"
            for row in lines[1:]:
                parts = row.split("\t")
                if len(parts) < 2:
                    continue
                entry, gene = parts[0].strip(), parts[1].strip()
                if entry:
                    # UniProt may return multiple symbols separated by whitespace
                    # ("SRRM3 SRM3") or semicolons ("H4C1; H4C2; ..."). Take the
                    # first token and strip any trailing punctuation.
                    if gene:
                        first = re.split(r"[\s;,]+", gene.strip(), maxsplit=1)[0]
                        cache[entry] = first.strip(";, ")
                    else:
                        cache[entry] = entry.replace("_HUMAN", "")
            print(f"    batch {i // BATCH + 1}: {len(batch)} queried "
                  f"-> {sum(1 for n in batch if n in cache)} hits")
            time.sleep(0.2)

    # Guarantee every input is resolvable (fall back to stripped prefix).
    for n in entry_names:
        cache.setdefault(n, n.replace("_HUMAN", ""))

    _write_cache(cache)
    return {n: cache[n] for n in entry_names}


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def high_confidence_per_bait(
    rep1: pd.DataFrame,
    rep2: pd.DataFrame,
    *,
    replicate_rule: str,
) -> dict[str, set[str]]:
    """{bait: set of entry_names high-confidence for that bait}.

    replicate_rule='intersection' (default): present in BOTH rep1 AND rep2.
    replicate_rule='union'                 : present in EITHER replicate.
    """
    out: dict[str, set[str]] = {}
    for bait in BAIT_COLS:
        s1 = set(rep1.loc[rep1[bait] > 0, "entry_name"])
        s2 = set(rep2.loc[rep2[bait] > 0, "entry_name"])
        if replicate_rule == "intersection":
            out[bait] = s1 & s2
        elif replicate_rule == "union":
            out[bait] = s1 | s2
        else:
            raise ValueError(f"unknown replicate_rule: {replicate_rule!r}")
    return out


# ---------------------------------------------------------------------------
# Rendering (mirrors roth_comparison.render_venn style)
# ---------------------------------------------------------------------------

def render_venn(
    our_genes: set[str],
    mabin_genes: set[str],
    list_key: str,
    out_path: Path,
    *,
    bait_scope_label: str,
    replicate_rule: str,
) -> None:
    our_only = our_genes - mabin_genes
    mabin_only = mabin_genes - our_genes
    both = our_genes & mabin_genes

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    fig.patch.set_facecolor("white")

    subsets = (len(our_only), len(mabin_only), len(both))
    our_color = LIST_COLORS[list_key]

    v = venn2(
        subsets=subsets,
        set_labels=(LIST_DISPLAY[list_key], "Mabin et al. EJC"),
        set_colors=(our_color, MABIN_COLOR),
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
        f"{LIST_DISPLAY[list_key]} \u2014 overlap with Mabin et al. (2018)",
        fontsize=16,
        fontweight="bold",
        color="#111111",
        pad=16,
    )
    union = len(our_genes | mabin_genes)
    pct = (len(both) / union) if union else 0.0
    subtitle = (
        f"ours = {len(our_genes)}    |    Mabin = {len(mabin_genes)}    |    "
        f"shared = {len(both)}  ({pct:.0%} of union)"
    )
    ax.text(
        0.5, -0.02, subtitle,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11, color="#444444",
    )
    footer = (
        f"EJC scope: {bait_scope_label}  |  replicate rule: {replicate_rule}  |  "
        f"mapping: UniProt entry name \u2192 HGNC"
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
# Summary writers
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
    scope_baits: tuple[str, ...],
    replicate_rule: str,
    folder: str,
    per_bait_hc: dict[str, set[str]],                   # bait -> set(entry_names)
    entry_to_gene: dict[str, str],
    mabin_genes: set[str],
    our_lists: dict[str, set[str]],
    overlaps: dict[str, set[str]],
    gene_to_baits: dict[str, set[str]],
) -> None:
    lines: list[str] = []
    lines.append(f"# Mabin et al. (2018) comparison \u2014 EJC scope: `{bait_scope}`\n")
    lines.append("## Parameters\n")
    lines.append(f"- Source: `{MMC2_PATH.name}` (Mabin et al., Cell Reports 2018, Table S1)")
    lines.append(f"- EJC baits included: {', '.join(f'`FLAG-{b}`' for b in scope_baits)}")
    lines.append(f"- Replicate rule: **{replicate_rule}** "
                 f"({'present in BOTH' if replicate_rule == 'intersection' else 'present in EITHER'} replicate, per bait)")
    lines.append(f"- Our folder: `{folder}`")
    lines.append(
        "- Gene-symbol mapping: UniProt REST API, cached in "
        "`.mabin_uniprot_cache.tsv`. Entry names like `IF4A3_HUMAN` resolve to "
        "their HGNC primary symbol (`EIF4A3`)."
    )
    lines.append("")

    lines.append("## Per-bait high-confidence counts\n")
    lines.append("| Bait | High-confidence interactors (entry names) |")
    lines.append("|---|---:|")
    for b in scope_baits:
        lines.append(f"| FLAG-{b} | {len(per_bait_hc[b])} |")
    lines.append(f"| **union (this scope)** | **{len(mabin_genes)} unique gene symbols** |")
    lines.append("")

    lines.append("## Overlap with our Ex15 lists\n")
    lines.append("| Our list | Our size | Overlap with Mabin | % of our list |")
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
        lines.append(f"### {LIST_DISPLAY[key]} \u2229 Mabin ({len(genes)} genes)\n")
        lines.append(_fmt_list(genes))
        lines.append("")

    lines.append("## Per-EJC-bait contribution\n")
    lines.append(
        "For each overlapping gene, which EJC bait(s) recovered it as a "
        "high-confidence interactor. Lets you distinguish core-EJC (MAGOH) "
        "vs alternate-EJC (RNPS1 / CASC3) Ex15 partners.\n"
    )
    for key in ("common", "POS_specific", "NEG_specific"):
        genes = sorted(overlaps[key])
        if not genes:
            continue
        lines.append(f"### {LIST_DISPLAY[key]}\n")
        lines.append("| Gene | EJC bait(s) |")
        lines.append("|---|---|")
        for g in genes:
            baits = sorted(gene_to_baits.get(g, set()))
            lines.append(f"| `{g}` | {', '.join(f'FLAG-{b}' for b in baits) if baits else '_(unknown)_'} |")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_bait_scope_comparison(
    comparison_root: Path,
    overlaps_by_scope: dict[str, dict[str, set[str]]],
    *,
    replicate_rule: str,
) -> None:
    lines: list[str] = []
    lines.append("# Mabin comparison: EJC bait-scope differences\n")
    lines.append(
        f"Compares `all_ejcs/` (MAGOH + CASC3 + RNPS1 pooled) against "
        f"`alternate_only/` (CASC3 + RNPS1 only, excluding the core-EJC "
        f"MAGOH column). Both scopes use replicate rule **{replicate_rule}**.\n"
    )
    lines.append(
        "Biological read: genes in `all_ejcs` but not in `alternate_only` are "
        "recovered by MAGOH only. MAGOH marks every EJC, so MAGOH-specific "
        "hits either bind the core EJC directly or are present in a MAGOH-"
        "only-containing complex that neither RNPS1 nor CASC3 stabilize. "
        "Genes present in both scopes are more confidently 'alternate-EJC' "
        "partners.\n"
    )

    if "all_ejcs" not in overlaps_by_scope or "alternate_only" not in overlaps_by_scope:
        lines.append("_Only one scope was run; cross-scope diff unavailable._\n")
        (comparison_root / "bait_scope_comparison.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        return

    all_e = overlaps_by_scope["all_ejcs"]
    alt = overlaps_by_scope["alternate_only"]

    lines.append("## Invariant check\n")
    lines.append(
        "`alternate_only` overlap MUST be a subset of `all_ejcs` overlap "
        "(dropping the MAGOH column can only remove preys, never add them).\n"
    )
    lines.append("| List | Subset? | all_ejcs | alternate_only |")
    lines.append("|---|---|---:|---:|")
    any_violation = False
    for key in ("common", "POS_specific", "NEG_specific"):
        a = all_e[key]
        c = alt[key]
        subset_ok = c.issubset(a)
        any_violation = any_violation or not subset_ok
        lines.append(
            f"| {LIST_DISPLAY[key]} | "
            f"{'OK' if subset_ok else 'VIOLATED'} | {len(a)} | {len(c)} |"
        )
    lines.append("")

    lines.append("## Genes unique to each scope\n")
    for key in ("common", "POS_specific", "NEG_specific"):
        a = all_e[key]
        c = alt[key]
        only_all = sorted(a - c)    # MAGOH-only hits
        only_alt = sorted(c - a)    # should be empty if invariant holds
        shared = sorted(a & c)      # recovered by both core and alternate

        lines.append(f"### {LIST_DISPLAY[key]}\n")
        lines.append(f"- **In both scopes** (core + alternate EJC) ({len(shared)}): "
                     f"{_fmt_list(shared)}")
        lines.append("")
        lines.append(f"- **MAGOH-only** (dropped by `alternate_only`) ({len(only_all)}): "
                     f"{_fmt_list(only_all)}")
        lines.append("")
        if only_alt:
            lines.append(
                f"- **INVARIANT VIOLATION \u2014 in `alternate_only` but NOT "
                f"`all_ejcs`** ({len(only_alt)}): {_fmt_list(only_alt)}"
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
    per_bait_hc: dict[str, set[str]],
    entry_to_gene: dict[str, str],
    bait_scope: str,
    replicate_rule: str,
    our_lists: dict[str, set[str]],
    scope_root: Path,
    folder_arg: str,
) -> dict[str, set[str]]:
    print(f"\n[EJC scope: {bait_scope}]")
    scope_root.mkdir(parents=True, exist_ok=True)
    (scope_root / "figures").mkdir(exist_ok=True)

    scope_baits = BAIT_SCOPES[bait_scope]

    # Build entry-name set for this scope + gene-to-baits map for the summary.
    scope_entries: set[str] = set()
    for b in scope_baits:
        scope_entries |= per_bait_hc[b]

    gene_to_baits: dict[str, set[str]] = {}
    for b in scope_baits:
        for entry in per_bait_hc[b]:
            g = entry_to_gene.get(entry)
            if not g:
                continue
            gene_to_baits.setdefault(g, set()).add(b)

    mabin_genes = {entry_to_gene[e] for e in scope_entries if e in entry_to_gene}
    mabin_genes.discard("")

    print(f"  {len(scope_entries)} unique entry names -> {len(mabin_genes)} unique gene symbols")

    (scope_root / "mabin_interactors_entries.txt").write_text(
        "\n".join(sorted(scope_entries)) + "\n", encoding="utf-8"
    )
    (scope_root / "mabin_interactors_genes.txt").write_text(
        "\n".join(sorted(mabin_genes)) + "\n", encoding="utf-8"
    )

    overlaps: dict[str, set[str]] = {}
    for key, genes in our_lists.items():
        overlap = genes & mabin_genes
        overlaps[key] = overlap
        (scope_root / f"overlap_{key}.txt").write_text(
            "\n".join(sorted(overlap)) + "\n", encoding="utf-8"
        )
        png = scope_root / "figures" / f"venn_{key}_vs_mabin.png"
        render_venn(
            genes, mabin_genes, key, png,
            bait_scope_label=bait_scope, replicate_rule=replicate_rule,
        )
        print(
            f"  {LIST_DISPLAY[key]:20}  ours={len(genes):4}  "
            f"overlap={len(overlap):4}  -> {png.relative_to(HERE)}"
        )

    write_summary(
        scope_root,
        bait_scope=bait_scope,
        scope_baits=scope_baits,
        replicate_rule=replicate_rule,
        folder=folder_arg,
        per_bait_hc={b: per_bait_hc[b] for b in scope_baits},
        entry_to_gene=entry_to_gene,
        mabin_genes=mabin_genes,
        our_lists=our_lists,
        overlaps=overlaps,
        gene_to_baits=gene_to_baits,
    )
    print("  wrote summary.md")
    return overlaps


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--folder", default="isoforms_distinct",
                    help="subfolder containing the _clean_genes.txt files "
                         "(default: isoforms_distinct)")
    ap.add_argument("--bait-scope", choices=[*BAIT_SCOPES.keys(), "both"], default="both",
                    help="which EJC bait scope(s) to run (default: both)")
    ap.add_argument("--replicate-rule", choices=["intersection", "union"], default="intersection",
                    help="rep1/rep2 combination rule per bait (default: intersection — "
                         "the publication-quality two-replicate standard)")
    args = ap.parse_args()

    folder = HERE / args.folder
    if not folder.is_dir():
        sys.exit(f"ERROR: folder not found: {folder}")
    if not MMC2_PATH.exists():
        sys.exit(f"ERROR: Mabin supplementary file not found: {MMC2_PATH}")

    print(f"Loading Mabin data from {MMC2_PATH.name} ...")
    rep1 = load_mabin_sheet(MMC2_PATH, SHEET_REP1)
    rep2 = load_mabin_sheet(MMC2_PATH, SHEET_REP2)
    print(f"  {SHEET_REP1}: {len(rep1)} proteins (at >=1 bait > 0)")
    print(f"  {SHEET_REP2}: {len(rep2)} proteins")

    per_bait_hc = high_confidence_per_bait(rep1, rep2, replicate_rule=args.replicate_rule)
    for b in BAIT_COLS:
        print(f"  high-confidence ({args.replicate_rule}) FLAG-{b}: "
              f"{len(per_bait_hc[b])} entry names")

    # Union of every entry we'll ever need; one UniProt batch resolves them all.
    all_entries = sorted(set.union(*per_bait_hc.values()))
    print(f"\nResolving {len(all_entries)} entry names to HGNC symbols ...")
    entry_to_gene = map_entry_names_to_genes(all_entries)

    our_lists: dict[str, set[str]] = {}
    for key, fname in LISTS.items():
        path = folder / fname
        if not path.exists():
            sys.exit(f"ERROR: missing input gene list: {path}")
        our_lists[key] = load_gene_list(path)
        print(f"  loaded {LIST_DISPLAY[key]:20}  {len(our_lists[key]):4} genes")

    comparison_root = folder / "mabin_comparison"
    comparison_root.mkdir(exist_ok=True)

    scopes_to_run: tuple[str, ...] = (
        tuple(BAIT_SCOPES.keys()) if args.bait_scope == "both" else (args.bait_scope,)
    )
    overlaps_by_scope: dict[str, dict[str, set[str]]] = {}
    for scope in scopes_to_run:
        overlaps_by_scope[scope] = run_one_scope(
            per_bait_hc=per_bait_hc,
            entry_to_gene=entry_to_gene,
            bait_scope=scope,
            replicate_rule=args.replicate_rule,
            our_lists=our_lists,
            scope_root=comparison_root / scope,
            folder_arg=args.folder,
        )

    write_bait_scope_comparison(
        comparison_root, overlaps_by_scope, replicate_rule=args.replicate_rule
    )
    print(f"\nwrote {(comparison_root / 'bait_scope_comparison.md').relative_to(HERE)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
