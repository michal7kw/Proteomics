# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Produce clean versions of the three Venn lists.

For each of the six existing list files (3 lists x 2 isoform-handling folders)
this script writes two siblings:

  *_clean.txt        only clean UniProt accessions, one per line (e.g. Q22222)
  *_clean_genes.txt  only HGNC gene symbols, one per line       (e.g. SRRM3)

The gene symbols are fetched from the UniProt REST API in batches. An accession
with no primary gene symbol on record (rare) falls back to its UniProt ID so no
protein silently disappears from the enrichment input.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
FOLDERS = ("isoforms_distinct", "isoforms_collapsed")
LISTS = ("Ex15_common", "Ex15_POS_specific", "Ex15_NEG_specific")

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/accessions"
BATCH_SIZE = 400          # /accessions endpoint accepts up to 500
TIMEOUT = 60.0


def clean_accession(line: str) -> str:
    """Defensive: strip whitespace and any trailing Scaffold metadata.

    Accepts a raw line and returns just the UniProt accession (which may
    include an isoform suffix like -5, preserved verbatim). Returns "" for
    empty/comment lines.
    """
    tok = line.strip()
    if not tok or tok.startswith("#"):
        return ""
    # First whitespace-separated chunk; strips " [2]", " (+1)", etc.
    return tok.split()[0]


def base_accession(acc: str) -> str:
    """Strip isoform suffix for gene-name lookup (Q9UKV3-5 -> Q9UKV3)."""
    return re.sub(r"-\d+$", "", acc)


def fetch_gene_symbols(accessions: list[str]) -> dict[str, str]:
    """Return {accession: primary_gene_symbol} via UniProt REST API."""
    out: dict[str, str] = {}
    with httpx.Client(timeout=TIMEOUT, http2=False) as client:
        for i in range(0, len(accessions), BATCH_SIZE):
            batch = accessions[i : i + BATCH_SIZE]
            params = {
                "accessions": ",".join(batch),
                "fields": "accession,gene_primary",
                "format": "tsv",
            }
            r = client.get(UNIPROT_URL, params=params)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            # Header: "Entry\tGene Names (primary)"
            for row in lines[1:]:
                parts = row.split("\t")
                if len(parts) < 2:
                    continue
                acc, gene = parts[0].strip(), parts[1].strip()
                if acc:
                    out[acc] = gene
            print(
                f"    batch {i // BATCH_SIZE + 1}: "
                f"{len(batch)} queried -> {len(out)} cumulative hits"
            )
            time.sleep(0.2)  # be polite
    return out


def read_list(path: Path) -> list[str]:
    """Read list file, return sorted unique clean accessions."""
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        acc = clean_accession(line)
        if acc:
            seen.add(acc)
    return sorted(seen)


def main() -> None:
    # 1) Collect all unique base accessions across every list for one API call
    all_accs: set[str] = set()
    for folder in FOLDERS:
        for lst in LISTS:
            src = HERE / folder / f"{lst}.txt"
            if src.exists():
                all_accs.update(read_list(src))
    print(f"Total unique accessions across all lists: {len(all_accs)}")

    # Query UniProt on base accessions (isoform suffix removed)
    base_accs = sorted({base_accession(a) for a in all_accs})
    print(f"Unique base accessions to query: {len(base_accs)}")
    try:
        gene_map_by_base = fetch_gene_symbols(base_accs)
    except httpx.HTTPError as e:
        print(f"\nUniProt lookup failed: {e}", file=sys.stderr)
        print("Writing _clean.txt only; skipping _clean_genes.txt", file=sys.stderr)
        gene_map_by_base = {}

    missing = [b for b in base_accs if b not in gene_map_by_base]
    if missing:
        print(f"  No record for {len(missing)} accession(s): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    no_symbol = [b for b, g in gene_map_by_base.items() if not g]
    if no_symbol:
        print(f"  No primary gene symbol for {len(no_symbol)} accession(s): {no_symbol[:5]}{'...' if len(no_symbol) > 5 else ''}")

    # 2) Write _clean.txt and _clean_genes.txt for each list
    print()
    for folder in FOLDERS:
        for lst in LISTS:
            src = HERE / folder / f"{lst}.txt"
            if not src.exists():
                continue
            accs = read_list(src)

            clean_path = src.with_name(f"{lst}_clean.txt")
            clean_path.write_text("\n".join(accs) + "\n", encoding="utf-8")

            if gene_map_by_base:
                genes: set[str] = set()
                for a in accs:
                    sym = gene_map_by_base.get(base_accession(a), "").strip()
                    # Some entries carry multiple symbols like "SRRM3 SRM3"; take first.
                    if sym:
                        genes.add(sym.split()[0])
                    else:
                        genes.add(a)  # fallback: keep accession so nothing is lost
                genes_path = src.with_name(f"{lst}_clean_genes.txt")
                genes_path.write_text(
                    "\n".join(sorted(genes)) + "\n", encoding="utf-8"
                )
                print(
                    f"  {folder}/{lst}: "
                    f"{len(accs)} accessions -> {len(genes)} unique gene symbols"
                )
            else:
                print(f"  {folder}/{lst}: {len(accs)} accessions (genes skipped)")


if __name__ == "__main__":
    main()
