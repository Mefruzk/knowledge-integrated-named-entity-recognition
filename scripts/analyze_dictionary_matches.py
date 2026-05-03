"""
analyze_dictionary_matches.py

Analyzes protein dictionary matching quality against gold annotations
across ALL three splits (train, dev, test).

Produces four TSV reports:
  1. fp_by_surface.tsv         — FPs grouped by surface form, ranked by frequency
  2. fn_by_surface.tsv         — FNs grouped by surface form, ranked by frequency
  3. cross_abstract_conflicts.tsv — same surface form, different gold label across abstracts
  4. fn_complex_indicators.tsv — FNs whose surface form contains complex-indicative morphology

Keeps existing abbreviation expansion analysis (Schwartz-Hearst via SciSpacy)
for FP classification, useful for later FP reduction work.

Usage:
    python scripts/analyze_dictionary_matches.py
"""

import csv
import json
import unicodedata
from pathlib import Path
from collections import defaultdict

from ner_core.core.entity import EntitySource, EntityLabel
from ner_core.data.loaders.factory import LoaderFactory

try:
    import spacy
    from scispacy.abbreviation import AbbreviationDetector
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    _nlp.add_pipe("abbreviation_detector")
    _ABBREV_AVAILABLE = True
    print("Abbreviation detector loaded.")
except Exception as e:
    print(f"Warning: abbreviation detection unavailable ({e})")
    _ABBREV_AVAILABLE = False
    _nlp = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_root      = Path(__file__).resolve().parents[1]
DATA_DIR          = project_root / "data" / "processed_with_dicts"
PREPROCESSED_PATH = project_root / "data" / "dictionaries" / "processed" / "protein_preprocessed.json"
OUTPUT_DIR        = project_root / "analysis" / "dictionary_match_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "dev", "test"]

# Complex-indicative morphology for FN flagging.
# Used as a lightweight proxy until the Complex Portal dictionary is built.
COMPLEX_INDICATORS = {
    "complex", "dimer", "trimer", "tetramer", "pentamer", "hexamer",
    "heterodimer", "homodimer", "heterotrimer", "homotrimer",
    "heterotetramer", "homotetramer", "oligomer", "multimer",
    "assembly", "subunit", "complex", "filament", "scaffold",
}

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def contains_complex_indicator(surface: str) -> str:
    """
    Returns the matched indicator token if the surface form contains any
    complex-indicative morphology, else empty string.
    Checks whole-word matches on lowercased surface.
    """
    low = surface.lower()
    for indicator in COMPLEX_INDICATORS:
        # Simple substring check; word boundary not enforced here
        # because indicators like "dimer" reliably don't appear in
        # unrelated protein names.
        if indicator in low:
            return indicator
    return ""


def check_expansion_in_dict(expansion: str, preprocessed: dict) -> str:
    if not expansion:
        return ""
    return "in_dict" if normalize(expansion) in preprocessed else "not_in_dict"

# ---------------------------------------------------------------------------
# Abbreviation detection
# ---------------------------------------------------------------------------

def get_abbreviation_map(text: str) -> dict:
    if not _ABBREV_AVAILABLE:
        return {}
    doc   = _nlp(text)
    pairs = {}
    for abrv in doc._.abbreviations:
        short = normalize(abrv.text)
        long_ = normalize(abrv._.long_form.text)
        pairs[short] = long_
    return pairs

# ---------------------------------------------------------------------------
# Load all splits
# ---------------------------------------------------------------------------

def load_all_examples() -> list:
    loader   = LoaderFactory.create("jsonl")
    examples = []
    for split in SPLITS:
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(loader.load(line))
                    count += 1
        print(f"  Loaded {count:>4} examples from {split}.jsonl")
    print(f"  Total: {len(examples)} examples across all splits")
    return examples

# ---------------------------------------------------------------------------
# Span key
# ---------------------------------------------------------------------------

def span_key(entity) -> tuple:
    return (entity.start_offset, entity.end_offset)

# ---------------------------------------------------------------------------
# Per-document evaluation
# ---------------------------------------------------------------------------

def evaluate_document(gold_entities, dict_entities):
    gold_protein = {span_key(e): e for e in gold_entities
                    if e.label == EntityLabel.PROTEIN}
    dict_protein = {span_key(e): e for e in dict_entities
                    if e.label == EntityLabel.PROTEIN}
    fp = [e for k, e in dict_protein.items() if k not in gold_protein]
    fn = [e for k, e in gold_protein.items() if k not in dict_protein]
    tp = [e for k, e in dict_protein.items() if k in gold_protein]
    return tp, fp, fn

# ---------------------------------------------------------------------------
# FP abbreviation classification
# ---------------------------------------------------------------------------

def classify_fp(fp_entity, dict_entities, abbrev_map: dict) -> tuple:
    meta    = fp_entity.metadata or {}
    pattern = meta.get("name_pattern", "")
    if pattern != "acronym":
        return "not_acronym", ""
    short_norm = normalize(fp_entity.span_text)
    long_form  = abbrev_map.get(short_norm)
    if long_form is None:
        return "no_expansion", ""
    dict_spans = {normalize(e.span_text)
                  for e in dict_entities
                  if e.label == EntityLabel.PROTEIN}
    if long_form in dict_spans:
        return "expansion_matched", long_form
    return "expansion_no_match", long_form

# ---------------------------------------------------------------------------
# Build gold surface-form index for cross-abstract conflict detection
# This is independent of dictionary matching.
# ---------------------------------------------------------------------------

def build_gold_surface_index(examples: list) -> dict:
    """
    Returns:
        surface_index: normalized_surface -> {
            "protein": [(doc_id, original_surface, context), ...],
            "complex": [(doc_id, original_surface, context), ...],
        }
    """
    index = defaultdict(lambda: defaultdict(list))
    for example in examples:
        gold_entities = example.get_all_entities(sources=[EntitySource.GOLD])
        for e in gold_entities:
            if e.label not in (EntityLabel.PROTEIN, EntityLabel.COMPLEX):
                continue
            surface  = e.span_text
            norm     = normalize(surface)
            label    = e.label.name  # "PROTEIN" or "COMPLEX"
            start    = e.start_offset
            end      = e.end_offset
            context  = example.text[max(0, start - 50): end + 50]
            index[norm][label].append((example.example_id, surface, context))
    return index

# ---------------------------------------------------------------------------
# TSV writers
# ---------------------------------------------------------------------------

def write_fp_tsv(all_fp_records: list, path: Path):
    """
    all_fp_records: list of dicts with keys:
        surface, doc_id, abbrev_class, expansion,
        matched_key, name_types, pattern, context
    Groups by normalized surface, sorts by frequency desc.
    """
    by_surface = defaultdict(list)
    for r in all_fp_records:
        by_surface[normalize(r["surface"])].append(r)

    rows = []
    for norm_surface, records in by_surface.items():
        doc_ids    = list({r["doc_id"] for r in records})
        abbrev_cls = records[0]["abbrev_class"]  # same for same surface form
        expansion  = records[0]["expansion"]
        matched_key= records[0]["matched_key"]
        name_types = records[0]["name_types"]
        pattern    = records[0]["pattern"]
        contexts   = " ||| ".join(r["context"] for r in records[:3])  # first 3
        rows.append({
            "surface"       : records[0]["surface"],
            "frequency"     : len(records),
            "doc_count"     : len(doc_ids),
            "abbrev_class"  : abbrev_cls,
            "expansion"     : expansion,
            "matched_key"   : matched_key,
            "name_types"    : name_types,
            "pattern"       : pattern,
            "doc_ids"       : " ".join(str(d) for d in doc_ids[:10]),
            "contexts"      : contexts,
        })
    rows.sort(key=lambda r: -r["frequency"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} FP surface forms → {path.name}")


def write_fn_tsv(all_fn_records: list, path: Path):
    """
    all_fn_records: list of dicts with keys:
        surface, doc_id, context, complex_indicator
    Groups by normalized surface, sorts by frequency desc.
    """
    by_surface = defaultdict(list)
    for r in all_fn_records:
        by_surface[normalize(r["surface"])].append(r)

    rows = []
    for norm_surface, records in by_surface.items():
        doc_ids          = list({r["doc_id"] for r in records})
        complex_indicator = records[0]["complex_indicator"]
        contexts         = " ||| ".join(r["context"] for r in records[:3])
        rows.append({
            "surface"          : records[0]["surface"],
            "frequency"        : len(records),
            "doc_count"        : len(doc_ids),
            "complex_indicator": complex_indicator,
            "doc_ids"          : " ".join(str(d) for d in doc_ids[:10]),
            "contexts"         : contexts,
        })
    rows.sort(key=lambda r: -r["frequency"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} FN surface forms → {path.name}")


def write_conflict_tsv(surface_index: dict, path: Path):
    """
    Writes surface forms that appear with BOTH protein and complex gold labels
    across abstracts, ranked by total occurrence count.
    """
    rows = []
    for norm_surface, label_map in surface_index.items():
        protein_records = label_map.get("PROTEIN", [])
        complex_records = label_map.get("COMPLEX", [])
        if not protein_records or not complex_records:
            continue  # no conflict
        total       = len(protein_records) + len(complex_records)
        p_doc_ids   = list({r[0] for r in protein_records})
        c_doc_ids   = list({r[0] for r in complex_records})
        p_contexts  = " ||| ".join(r[2] for r in protein_records[:2])
        c_contexts  = " ||| ".join(r[2] for r in complex_records[:2])
        rows.append({
            "surface"           : protein_records[0][1],  # original casing
            "total_occurrences" : total,
            "protein_count"     : len(protein_records),
            "complex_count"     : len(complex_records),
            "protein_doc_ids"   : " ".join(str(d) for d in p_doc_ids[:10]),
            "complex_doc_ids"   : " ".join(str(d) for d in c_doc_ids[:10]),
            "protein_contexts"  : p_contexts,
            "complex_contexts"  : c_contexts,
        })
    rows.sort(key=lambda r: -r["total_occurrences"])

    if not rows:
        print("  No cross-abstract label conflicts found.")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} conflict surface forms → {path.name}")


def write_fn_complex_indicators_tsv(all_fn_records: list, path: Path):
    """
    Subset of FNs whose surface form contains complex-indicative morphology.
    These are gold protein annotations that look like complexes — highest review priority.
    """
    flagged = [r for r in all_fn_records if r["complex_indicator"]]
    by_surface = defaultdict(list)
    for r in flagged:
        by_surface[normalize(r["surface"])].append(r)

    rows = []
    for norm_surface, records in by_surface.items():
        doc_ids   = list({r["doc_id"] for r in records})
        indicator = records[0]["complex_indicator"]
        contexts  = " ||| ".join(r["context"] for r in records[:3])
        rows.append({
            "surface"          : records[0]["surface"],
            "frequency"        : len(records),
            "doc_count"        : len(doc_ids),
            "complex_indicator": indicator,
            "doc_ids"          : " ".join(str(d) for d in doc_ids[:10]),
            "contexts"         : contexts,
        })
    rows.sort(key=lambda r: -r["frequency"])

    if not rows:
        print("  No FNs with complex indicators found.")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} FN complex-indicator surface forms → {path.name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading preprocessed dictionary from {PREPROCESSED_PATH.name}...")
    with open(PREPROCESSED_PATH, encoding="utf-8") as f:
        preprocessed = json.load(f)
    print(f"  {len(preprocessed):,} keys loaded.")

    print(f"\nLoading examples from all splits...")
    examples = load_all_examples()

    # Build gold surface index for cross-abstract conflict detection
    # This runs over all gold annotations regardless of dictionary matching.
    print(f"\nBuilding gold surface index for cross-abstract conflict detection...")
    surface_index = build_gold_surface_index(examples)
    print(f"  {len(surface_index):,} unique normalized surface forms indexed.")

    all_fp_records = []
    all_fn_records = []
    all_tp         = []

    total_gold = 0
    total_dict = 0

    for example in examples:
        gold_entities = example.get_all_entities(sources=[EntitySource.GOLD])
        dict_entities = example.get_all_entities(sources=[EntitySource.DICTIONARY])
        abbrev_map    = get_abbreviation_map(example.text)

        tp, fp, fn = evaluate_document(gold_entities, dict_entities)

        total_gold += len([e for e in gold_entities if e.label == EntityLabel.PROTEIN])
        total_dict += len([e for e in dict_entities if e.label == EntityLabel.PROTEIN])

        for e in fp:
            start        = e.start_offset
            end          = e.end_offset
            context      = example.text[max(0, start - 50): end + 50]
            abbrev_class, expansion = classify_fp(e, dict_entities, abbrev_map)
            meta         = e.metadata or {}
            all_fp_records.append({
                "surface"     : e.span_text,
                "doc_id"      : example.example_id,
                "abbrev_class": abbrev_class,
                "expansion"   : expansion,
                "matched_key" : meta.get("matched_key", ""),
                "name_types"  : str(list({r["name_type"] for r in meta.get("match_records", [])})),
                "pattern"     : meta.get("name_pattern", ""),
                "context"     : f"...{context}...",
            })

        for e in fn:
            start     = e.start_offset
            end       = e.end_offset
            context   = example.text[max(0, start - 50): end + 50]
            indicator = contains_complex_indicator(e.span_text)
            all_fn_records.append({
                "surface"          : e.span_text,
                "doc_id"           : example.example_id,
                "context"          : f"...{context}...",
                "complex_indicator": indicator,
            })

        for e in tp:
            abbrev_class, expansion = classify_fp(e, dict_entities, abbrev_map)
            dict_check = check_expansion_in_dict(expansion, preprocessed)
            all_tp.append((e, abbrev_class, expansion, dict_check))

    # ── Summary ──────────────────────────────────────────────────────────────
    tp_count  = total_gold - len(all_fn_records)
    precision = tp_count / total_dict if total_dict else 0.0
    recall    = tp_count / total_gold if total_gold else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall else 0.0)

    print(f"\n{'='*60}")
    print(f"  PROTEIN Dictionary Match Analysis  —  all splits")
    print(f"{'='*60}")
    print(f"  Gold protein entities : {total_gold:,}")
    print(f"  Dict protein entities : {total_dict:,}")
    print(f"  True positives (TP)   : {tp_count:,}")
    print(f"  False positives (FP)  : {len(all_fp_records):,}  "
          f"({len({normalize(r['surface']) for r in all_fp_records}):,} unique surfaces)")
    print(f"  False negatives (FN)  : {len(all_fn_records):,}  "
          f"({len({normalize(r['surface']) for r in all_fn_records}):,} unique surfaces)")
    print(f"  Precision             : {precision:.3f}")
    print(f"  Recall                : {recall:.3f}")
    print(f"  F1                    : {f1:.3f}")

    # Cross-abstract conflict summary
    conflict_count = sum(
        1 for label_map in surface_index.values()
        if label_map.get("PROTEIN") and label_map.get("COMPLEX")
    )
    print(f"\n  Cross-abstract label conflicts: {conflict_count:,} surface forms "
          f"labeled both protein and complex across abstracts")

    fn_with_indicator = len({normalize(r["surface"]) for r in all_fn_records
                              if r["complex_indicator"]})
    print(f"  FN complex indicators : {fn_with_indicator:,} unique FN surfaces "
          f"with complex-indicative morphology")

    # Abbreviation breakdown
    if _ABBREV_AVAILABLE:
        from collections import Counter
        fp_cls_counts = Counter(r["abbrev_class"] for r in all_fp_records)
        print(f"\n  FP abbreviation breakdown:")
        for cls, cnt in fp_cls_counts.most_common():
            pct = cnt / len(all_fp_records) * 100 if all_fp_records else 0
            print(f"    {cls:<25} : {cnt:,}  ({pct:.1f}%)")

    print(f"{'='*60}")

    # ── Write TSV reports ─────────────────────────────────────────────────────
    print(f"\nWriting TSV reports to {OUTPUT_DIR}/...")

    write_fp_tsv(
        all_fp_records,
        OUTPUT_DIR / "fp_by_surface.tsv"
    )
    write_fn_tsv(
        all_fn_records,
        OUTPUT_DIR / "fn_by_surface.tsv"
    )
    write_conflict_tsv(
        surface_index,
        OUTPUT_DIR / "cross_abstract_conflicts.tsv"
    )
    write_fn_complex_indicators_tsv(
        all_fn_records,
        OUTPUT_DIR / "fn_complex_indicators.tsv"
    )

    print(f"\nDone. Review priority order:")
    print(f"  1. cross_abstract_conflicts.tsv  — same surface, different label")
    print(f"  2. fn_complex_indicators.tsv     — gold protein that looks like a complex")
    print(f"  3. fp_by_surface.tsv             — dict fires but not annotated (sorted by freq)")
    print(f"  4. fn_by_surface.tsv             — annotated but dict misses (sorted by freq)")


if __name__ == "__main__":
    main()