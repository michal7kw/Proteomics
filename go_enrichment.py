# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gprofiler-official>=1.0",
#   "pandas>=2.0",
# ]
# ///
"""Multi-source GO/pathway enrichment for the three Ex15 gene lists.

Runs enrichment via g:Profiler (online, auto-updated annotations, g:SCS
multiple-testing correction) on:
  - Ex15_common_clean_genes.txt       (shared, isoform-independent core)
  - Ex15_POS_specific_clean_genes.txt (POS-only)
  - Ex15_NEG_specific_clean_genes.txt (NEG-only)

Outputs under <folder>/go_enrichment/:
  gprofiler_raw.tsv           canonical long-format table (all 3 queries)
  Ex15_*_enrichment.tsv       per-list tables, sorted by adj p
  summary.md                  top-5 terms per list per source + sanity checks

Usage:
  uv run go_enrichment.py
  uv run go_enrichment.py --folder isoforms_collapsed
  uv run go_enrichment.py --sources GO:BP GO:MF CORUM
  uv run go_enrichment.py --background path/to/ms_detectable_genes.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from gprofiler import GProfiler

HERE = Path(__file__).parent

DEFAULT_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP", "CORUM"]

LISTS = {
    "Ex15_common":       "Ex15_common_clean_genes.txt",
    "Ex15_POS_specific": "Ex15_POS_specific_clean_genes.txt",
    "Ex15_NEG_specific": "Ex15_NEG_specific_clean_genes.txt",
}

SANITY_CHECKS = [
    # (pattern, sources_allowed, threshold, why)
    (r"RNA splicing|mRNA splicing",         ["GO:BP"],            1e-5,  "SRRM3 is an SR splicing factor"),
    (r"mRNA processing",                     ["GO:BP"],            1e-3,  "Core SR-protein function"),
    (r"spliceosomal complex|spliceosome",    ["GO:CC", "CORUM"],   1e-3,  "Physical pulldown expectation"),
    (r"nuclear speck|nuclear speckle",       ["GO:CC"],            5e-2,  "SR proteins localize here"),
    (r"U1 snRNP|U2 snRNP|U2-type spliceosomal", ["CORUM"],         5e-2,  "Direct complex recovery"),
    (r"poly.A. RNA binding|RNA binding",     ["GO:MF"],            1e-5,  "Biochemical signature"),
]


def load_gene_list(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def run_enrichment(
    queries: dict[str, list[str]],
    sources: list[str],
    organism: str = "hsapiens",
    background: list[str] | None = None,
    user_threshold: float = 0.05,
) -> pd.DataFrame:
    """One batched g:Profiler call for all 3 lists -> long-format DataFrame."""
    gp = GProfiler(return_dataframe=True, user_agent="srrm3-ex15-interactome/1.0")
    kwargs = dict(
        organism=organism,
        query=queries,
        sources=sources,
        user_threshold=user_threshold,
        significance_threshold_method="g_SCS",
        no_evidences=False,  # keep `intersections` (gene hits) for plotting
        all_results=False,   # significant terms only
        ordered=False,
    )
    if background:
        kwargs["background"] = background
        kwargs["domain_scope"] = "custom"
    df = gp.profile(**kwargs)
    # Normalize column naming across gprofiler-official versions
    if "adjusted_p_value" not in df.columns and "p_value" in df.columns:
        df["adjusted_p_value"] = df["p_value"]  # g_SCS-corrected value is reported as p_value
    # Ensure intersections column is always a list for downstream join
    if "intersections" in df.columns:
        df["intersections"] = df["intersections"].apply(
            lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [v])
        )
    return df


def write_per_list_tables(df: pd.DataFrame, out_dir: Path) -> None:
    for list_name in LISTS:
        sub = df[df["query"] == list_name].copy()
        if sub.empty:
            print(f"  (no significant terms for {list_name})")
            continue
        sub = sub.sort_values("p_value").reset_index(drop=True)
        path = out_dir / f"{list_name}_enrichment.tsv"
        # Serialize intersections as '|'-joined for TSV portability
        to_write = sub.copy()
        if "intersections" in to_write.columns:
            to_write["intersections"] = to_write["intersections"].apply(
                lambda v: "|".join(v) if isinstance(v, list) else str(v)
            )
        to_write.to_csv(path, sep="\t", index=False)
        print(f"  {path.name}  ({len(sub)} significant terms)")


def sanity_check(df: pd.DataFrame, list_name: str = "Ex15_common") -> list[dict]:
    sub = df[df["query"] == list_name]
    results: list[dict] = []
    for pattern, sources_allowed, threshold, why in SANITY_CHECKS:
        rx = re.compile(pattern, re.IGNORECASE)
        candidates = sub[sub["source"].isin(sources_allowed)]
        matches = candidates[candidates["name"].astype(str).str.contains(rx, na=False)]
        if matches.empty:
            results.append({
                "pattern": pattern, "sources": sources_allowed, "threshold": threshold,
                "why": why, "status": "FAIL", "best_p": None, "hit": None, "src_hit": None,
            })
            continue
        best = matches.loc[matches["p_value"].idxmin()]
        status = "PASS" if best["p_value"] <= threshold else "WARN"
        results.append({
            "pattern": pattern, "sources": sources_allowed, "threshold": threshold,
            "why": why, "status": status,
            "best_p": float(best["p_value"]), "hit": str(best["name"]),
            "src_hit": str(best["source"]),
        })
    return results


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return f"{p:.2e}"


def write_summary_md(
    df: pd.DataFrame,
    checks: list[dict],
    out_path: Path,
    folder_name: str,
    sources_used: list[str],
) -> None:
    lines: list[str] = []
    lines.append(f"# GO enrichment summary\n")
    lines.append(f"Folder: `{folder_name}`  \n")
    lines.append(f"Sources queried: `{', '.join(sources_used)}`  \n")
    lines.append(f"Significance threshold: g:SCS adj p < 0.05  \n")
    lines.append("")

    # ---- sanity checks ----
    lines.append("## Sanity checks (Ex15_common)\n")
    lines.append("Confirm SRRM3 interactome biology looks right before trusting POS-vs-NEG differences.\n")
    lines.append("")
    lines.append("| Status | Expected term | Best hit | Source | Adj p | Threshold |")
    lines.append("|---|---|---|---|---|---|")
    for c in checks:
        hit = c["hit"] or "(no match)"
        src = c["src_hit"] or "—"
        lines.append(
            f"| `[{c['status']}]` | `{c['pattern']}` | {hit} | {src} | "
            f"{_fmt_p(c['best_p'])} | {c['threshold']:.0e} |"
        )
    lines.append("")

    # ---- top terms per list per source ----
    lines.append("## Top 5 terms per list per source\n")
    for list_name in LISTS:
        lines.append(f"### {list_name}\n")
        sub = df[df["query"] == list_name]
        if sub.empty:
            lines.append("_no significant terms_\n")
            continue
        for src in sources_used:
            src_sub = sub[sub["source"] == src].sort_values("p_value").head(5)
            if src_sub.empty:
                continue
            lines.append(f"**{src}**")
            lines.append("")
            lines.append("| Term | Genes | Adj p |")
            lines.append("|---|---|---|")
            for _, row in src_sub.iterrows():
                lines.append(
                    f"| {row['name']} | "
                    f"{row.get('intersection_size','?')}/{row.get('term_size','?')} | "
                    f"{_fmt_p(float(row['p_value']))} |"
                )
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="isoforms_distinct",
                    help="subfolder containing the _clean_genes.txt files (default: isoforms_distinct)")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                    help=f"enrichment sources to query (default: {DEFAULT_SOURCES})")
    ap.add_argument("--background", type=Path, default=None,
                    help="optional custom background: file of gene symbols, one per line")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="significance threshold after g:SCS correction (default: 0.05)")
    args = ap.parse_args()

    folder = HERE / args.folder
    if not folder.is_dir():
        sys.exit(f"ERROR: folder not found: {folder}")

    queries: dict[str, list[str]] = {}
    for list_name, fname in LISTS.items():
        path = folder / fname
        if not path.exists():
            sys.exit(f"ERROR: missing input file: {path}")
        queries[list_name] = load_gene_list(path)
        print(f"  loaded {list_name}: {len(queries[list_name])} genes")

    background = None
    if args.background is not None:
        if not args.background.exists():
            sys.exit(f"ERROR: background file not found: {args.background}")
        background = load_gene_list(args.background)
        print(f"  custom background: {len(background)} genes")

    print(f"\nQuerying g:Profiler for {len(args.sources)} sources: {args.sources}")
    df = run_enrichment(queries, args.sources, background=background, user_threshold=args.threshold)
    print(f"  -> {len(df)} significant terms total")

    out_dir = folder / "go_enrichment"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    # raw canonical TSV (keeps intersections as '|'-joined for portability)
    raw = df.copy()
    if "intersections" in raw.columns:
        raw["intersections"] = raw["intersections"].apply(
            lambda v: "|".join(v) if isinstance(v, list) else str(v)
        )
    raw_path = out_dir / "gprofiler_raw.tsv"
    raw.to_csv(raw_path, sep="\t", index=False)
    print(f"\nwrote {raw_path.name}  ({len(df)} rows, {len(df.columns)} cols)")

    print("\nper-list tables:")
    write_per_list_tables(df, out_dir)

    print("\nrunning sanity checks on Ex15_common ...")
    checks = sanity_check(df, "Ex15_common")
    for c in checks:
        marker = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[c["status"]]
        hit = c["hit"] or "(no match)"
        print(f"  {marker:6}  {c['pattern']:45}  {hit[:40]:40}  adj p={_fmt_p(c['best_p'])}")

    summary_path = out_dir / "summary.md"
    write_summary_md(df, checks, summary_path, args.folder, args.sources)
    print(f"\nwrote {summary_path.name}")

    failed = [c for c in checks if c["status"] == "FAIL"]
    critical_fails = [c for c in failed if c["pattern"].startswith(("RNA splicing", "spliceosomal"))]
    if critical_fails:
        print("\nWARNING: critical sanity checks failed. Investigate gene lists before interpreting POS/NEG differences.",
              file=sys.stderr)

    print(f"\nDone. Next:  uv run go_visualize.py --folder {args.folder}")


if __name__ == "__main__":
    main()
