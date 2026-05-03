"""
build_dictionary.py

Parses uniprot_sprot.xml and produces two JSON outputs:

1. protein_name_index.json
   Surface form (raw, no normalization) -> list of match records
   Each match record: { accession, name_type, organism_id }

   name_type values:
     recommended_full, recommended_short,
     alternative_full, alternative_short,
     submitted_full, submitted_short,
     gene_primary, gene_synonym

2. protein_entries.json
   Accession -> entry metadata (joined at analysis time, not stored in index)
   Fields: entry_name, organism_id, organism_name, pe_level,
           has_subunit, subunit_text, hgnc_id, dataset

Usage:
    python build_dictionary.py
"""

import re
import json
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import os, sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/dictionaries/raw")
OUT_DIR = Path("data/dictionaries/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

XML_PATH = RAW_DIR / "uniprot_sprot.xml"
INDEX_PATH = OUT_DIR / "protein_name_index.json"
ENTRIES_PATH = OUT_DIR / "protein_entries.json"

# ---------------------------------------------------------------------------
# UniProt XML namespace
# ---------------------------------------------------------------------------
NS = "https://uniprot.org/uniprot"
T = lambda tag: f"{{{NS}}}{tag}"  # namespace-qualified tag helper

# ---------------------------------------------------------------------------
# Protein existence type string -> PE level int
# ---------------------------------------------------------------------------
PE_MAP = {
    "evidence at protein level":    1,
    "evidence at transcript level": 2,
    "inferred from homology":       3,
    "predicted":                    4,
    "uncertain":                    5,
}

# ---------------------------------------------------------------------------
# Target organisms — entries outside this set are skipped entirely
# ---------------------------------------------------------------------------
TARGET_ORGANISMS = {9606, 10090, 10116}  # human, mouse, rat

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_entry(entry_elem):
    """
    Extract all relevant fields from a single <entry> element.
    Returns (None, [], {}) for entries outside TARGET_ORGANISMS.

    Returns:
        accession   : str
        names       : list of (surface_form, name_type)
        metadata    : dict  (stored in entries table)
    """
    # -- Accession (first <accession> child is the primary AC) -------------
    accession = entry_elem.findtext(T("accession"), default="").strip()
    if not accession:
        return None, [], {}

    # -- Entry name --------------------------------------------------------
    entry_name = entry_elem.findtext(T("name"), default="").strip()

    # -- Dataset attribute (Swiss-Prot / TrEMBL) ---------------------------
    dataset = entry_elem.get("dataset", "")

    # -- Protein existence level -------------------------------------------
    pe_elem = entry_elem.find(T("proteinExistence"))
    pe_raw = pe_elem.get("type", "").lower() if pe_elem is not None else ""
    pe_level = PE_MAP.get(pe_raw, None)

    # -- Organism — skip early if not a target organism --------------------
    organism_id = None
    organism_name = ""
    organism_elem = entry_elem.find(T("organism"))
    if organism_elem is not None:
        sci_name = organism_elem.find(f"{T('name')}[@type='scientific']")
        if sci_name is not None:
            organism_name = sci_name.text or ""
        tax_ref = organism_elem.find(f"{T('dbReference')}[@type='NCBI Taxonomy']")
        if tax_ref is not None:
            try:
                organism_id = int(tax_ref.get("id", ""))
            except ValueError:
                organism_id = None

    if organism_id not in TARGET_ORGANISMS:
        return None, [], {}

    # -- HGNC cross-reference ----------------------------------------------
    hgnc_id = None
    for dbref in entry_elem.findall(T("dbReference")):
        if dbref.get("type") == "HGNC":
            hgnc_id = dbref.get("id")
            break

    # -- Subunit comment ---------------------------------------------------
    has_subunit = False
    subunit_text = ""

    for comment in entry_elem.findall(T("comment")):
        if comment.get("type") == "subunit":
            text_elem = comment.find(T("text"))
            if text_elem is not None and text_elem.text:
                subunit_text = text_elem.text.strip()
                has_subunit = True
            break  # only one subunit comment per entry

    # -- Names -------------------------------------------------------------
    names = []  # list of (surface_form, name_type)

    def add(surface_form, name_type):
        """Add non-empty surface forms only."""
        if surface_form and surface_form.strip():
            names.append((surface_form.strip(), name_type))

    # Protein names
    prot_elem = entry_elem.find(T("protein"))
    if prot_elem is not None:
        # Recommended name (most entries)
        rec = prot_elem.find(T("recommendedName"))
        if rec is not None:
            add(rec.findtext(T("fullName")),  "recommended_full")
            add(rec.findtext(T("shortName")), "recommended_short")

        # Alternative names (may be multiple)
        for alt in prot_elem.findall(T("alternativeName")):
            add(alt.findtext(T("fullName")), "alternative_full")
            for short in alt.findall(T("shortName")):
                add(short.text, "alternative_short")

        # Submitted name — used when no recommended name exists
        for sub in prot_elem.findall(T("submittedName")):
            add(sub.findtext(T("fullName")),  "submitted_full")
            add(sub.findtext(T("shortName")), "submitted_short")

    # Gene names — primary and synonym only; ORF and ordered locus excluded
    gene_type_map = {
        "primary": "gene_primary",
        "synonym": "gene_synonym",
    }
    gene_elem = entry_elem.find(T("gene"))
    if gene_elem is not None:
        for gname in gene_elem.findall(T("name")):
            gtype = gname.get("type", "")
            mapped = gene_type_map.get(gtype)
            if mapped:
                add(gname.text, mapped)

    # -- Metadata dict -----------------------------------------------------
    metadata = {
        "entry_name":    entry_name,
        "dataset":       dataset,
        "organism_id":   organism_id,
        "organism_name": organism_name,
        "pe_level":      pe_level,
        "has_subunit":   has_subunit,
        "subunit_text":  subunit_text,   # raw text — homomer/heteromer flags deferred to analysis
        "hgnc_id":       hgnc_id,
    }

    return accession, names, metadata


def build_dictionary(xml_path: Path):
    """
    Stream-parse the XML and build:
      name_index : dict[surface_form -> list of match records]
      entries    : dict[accession -> metadata]
    """
    name_index = defaultdict(list)
    entries = {}

    print(f"Parsing {xml_path} ...")

    # iterparse streams the file — memory-safe for the full Swiss-Prot XML
    context = ET.iterparse(str(xml_path), events=("end",))

    count = 0
    for event, elem in context:
        # Process only top-level <entry> elements
        if elem.tag != T("entry"):
            continue

        accession, names, metadata = parse_entry(elem)


        if accession:
            entries[accession] = metadata

            for surface_form, name_type in names:
                name_index[surface_form].append({
                    "accession":   accession,
                    "name_type":   name_type,
                    "organism_id": metadata["organism_id"],
                    "surface_form": surface_form,
                })

            count += 1
            if count % 100_000 == 0:
                print(f"  {count:,} entries processed ...")

        # Free memory — critical for large XML files
        elem.clear()

    print(f"Done. {count:,} entries parsed.")
    print(f"  Unique surface forms : {len(name_index):,}")
    print(f"  Total match records  : {sum(len(v) for v in name_index.values()):,}")

    return dict(name_index), entries


def save(name_index, entries):
    print(f"\nSaving name index  -> {INDEX_PATH}")
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(name_index, f, ensure_ascii=False, indent=2)

    print(f"Saving entry table -> {ENTRIES_PATH}")
    with open(ENTRIES_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


# ===========================================================================
# STAGE 2 — DictionaryPreprocessor
# ===========================================================================
 
class DictionaryPreprocessor:
    """
    Normalizes the raw name_index and classifies each entry.
 
    Hyper-parameters:
        min_length        (int)  — names shorter than this flagged too_short.
                                   Not removed. Default: 3.
        strip_whitespace  (bool) — remove all whitespace from norm key vs
                                   collapse to single space. Default: False.
        n_pattern_samples (int)  — (original, norm) pairs shown per name
                                   pattern class in the report. Default: 5.
        n_merge_samples   (int)  — rows shown for merges/multi-acc/too-short.
                                   Default: 20.
 
    Normalization pipeline (fixed order):
        1. Unicode NFC
        2. Greek letter expansion      (α → alpha, Β → beta, ...)
        3. Hyphen-digit rule           IL-6 → IL6
           Pattern: ([A-Za-z])-(\\d)
           Sub:     \\1\\2
           \\1 = letter capture group, \\2 = digit capture group.
           The hyphen between them is discarded; both sides are kept.
        4. Slash → space               serine/threonine → serine threonine
        5. Punctuation removal         ( ) [ ] { } , ; : . " ' `
        6. Lowercase
        7. Whitespace: collapse to space (default) or remove entirely
 
    Name pattern classification (raw form, whitespace-tokenized):
        acronym       single token, all uppercase + optional trailing digits
        single_token  single token, not acronym  (e.g. "laminin", "p53", "c-Myc")
        multi_token   two or more whitespace-separated tokens
 
    Family candidate detection: COMMENTED OUT.
    Too many legitimate protein names contain terms like "type" or "family".
    Use n_pattern_samples / n_merge_samples to inspect raw names manually first.
    """
 
    GREEK = {
        "α": "alpha",   "β": "beta",    "γ": "gamma",   "δ": "delta",
        "ε": "epsilon", "ζ": "zeta",    "η": "eta",     "θ": "theta",
        "ι": "iota",    "κ": "kappa",   "λ": "lambda",  "μ": "mu",
        "ν": "nu",      "ξ": "xi",      "ο": "omicron", "π": "pi",
        "ρ": "rho",     "σ": "sigma",   "τ": "tau",     "υ": "upsilon",
        "φ": "phi",     "χ": "chi",     "ψ": "psi",     "ω": "omega",
        "Α": "alpha",   "Β": "beta",    "Γ": "gamma",   "Δ": "delta",
        "Ε": "epsilon", "Ζ": "zeta",    "Η": "eta",     "Θ": "theta",
        "Ι": "iota",    "Κ": "kappa",   "Λ": "lambda",  "Μ": "mu",
        "Ν": "nu",      "Ξ": "xi",      "Ο": "omicron", "Π": "pi",
        "Ρ": "rho",     "Σ": "sigma",   "Τ": "tau",     "Υ": "upsilon",
        "Φ": "phi",     "Χ": "chi",     "Ψ": "psi",     "Ω": "omega",
    }
    # Set of Greek characters for fast membership testing
    _GREEK_CHARS = frozenset(GREEK.keys())
    # Hyphen followed by digit, Greek char, or Roman numeral suffix char.
    # Captures the suffix character for use in variant generation.
    _HYPHEN_SUFFIX = re.compile(
        r"-([0-9IVXLCivxlcα-ωΑ-Ω])"
    )
 
    _PUNCT_RE     = re.compile(r"[()[\]{},;:.\"'`]")
    _HYPHEN_DIGIT = re.compile(r"([A-Za-z])-(\d)")   # \1 = letter, \2 = digit
    _SLASH_RE     = re.compile(r"/")
    # Greek characters that can appear as name suffixes (e.g. laminin-α, TNF-α)
    _GREEK_CHARS  = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    # Splits base from suffix: hyphen followed by digit or Greek letter
    # laminin-10 → base=laminin   laminin-α → base=laminin   TNF-α → base=TNF
    _BASE_SPLIT   = re.compile(
        "[-](?=[0-9αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ])"
    )
 
    def __init__(
        self,
        min_length: int        = 3,
        strip_whitespace: bool = False,
        n_pattern_samples: int = 5,
        n_merge_samples: int   = 20,
        compact_min_length: int = 5,  # single tokens shorter than this → compact
    ):
        self.min_length         = min_length
        self.strip_whitespace   = strip_whitespace
        self.n_pattern_samples  = n_pattern_samples
        self.n_merge_samples    = n_merge_samples
        self.compact_min_length = compact_min_length
 
    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
 
    def run(self, name_index: dict, verbose: bool = True) -> tuple:
        print(f"\n[Stage 2] Preprocessing  "
              f"strip_whitespace={self.strip_whitespace}  "
              f"min_length={self.min_length}")
 
        preprocessed = {}
        report       = self._empty_report()
 
        for raw_form, match_records in name_index.items():
            norm_key = self.normalize(raw_form)
 
            if not norm_key:
                report["skipped_empty_after_norm"] += 1
                continue
 
            # Generate all surface variants for this raw form.
            # Each variant becomes its own key in the preprocessed dict,
            # all pointing to the same canonical entry data.
            # norm_key is always one of the variants (the normalized form).
            variants = self.generate_variants(raw_form)
 
            for variant in variants:
                if variant not in preprocessed:
                    preprocessed[variant] = {
                        "norm_key":       norm_key,   # canonical key for grouping
                        "original_forms": [],
                        "match_records":  [],
                        "name_pattern":   self._classify(raw_form),
                        "token_count":    self._token_count(raw_form),
                        "flags": {
                            "too_short": len(variant) < self.min_length,
                            # family_candidate: commented out — inspect names first
                        },
                    }
                    report["unique_norm_keys"] += 1
                else:
                    report["normalization_merges"] += 1
 
                entry = preprocessed[variant]
 
                if raw_form not in entry["original_forms"]:
                    entry["original_forms"].append(raw_form)
 
                # surface_form inside each record keeps (accession, name_type)
                # traceable even after multiple originals merge onto same key
                entry["match_records"].extend(match_records)
 
        report = self._compute_report(preprocessed, report)
 
        if verbose:
            self._print_report(report)
 
        return preprocessed, report
 
    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
 
    def normalize(self, text: str) -> str:
        """
        Canonical form used purely for grouping/deduplication (norm_key).
 
        Only applies Unicode NFC and lowercase — no Greek expansion, no
        hyphen handling, no punctuation removal. Those transformations
        are handled explicitly in generate_variants() which produces the
        actual automaton keys. This keeps the two concerns separate:
          normalize()         -> deduplication key (are two raw forms the same?)
          generate_variants() -> matching keys (what forms appear in text?)
        """
        text = unicodedata.normalize("NFC", text)
        return text.lower()
 
    # ------------------------------------------------------------------
    # Variant generation  (for Aho-Corasick automaton keys)
    # ------------------------------------------------------------------
 
    def generate_variants(self, raw_form: str) -> list:
        """
        Generate all surface form variants to store as automaton keys.
        All variants are lowercased — match against lowercased text only.
 
        Transformations applied to hyphens before digit, Greek, or Roman
        numeral suffix (I, V, X, L, C):
          with hyphen    : il-6   tnf-α   tnf-alpha   phas-i
          hyphen→space   : il 6   tnf α   tnf alpha   phas i
          hyphen removed : il6    tnfα    tnfalpha    phasi
 
        Greek characters additionally produce both unicode and spelled-out
        forms for each hyphen variant.
 
        Names with no qualifying suffix produce one variant (lowercased).
        Variants are deduplicated and sorted.
 
        To add new variation rules (e.g. slash handling, Roman numeral
        expansion), add them here without touching the matcher.
        """
        # Always include the original raw form as a key for traceability
        # normalize() provides the lowercased base for variant generation
        base       = self.normalize(raw_form)
        has_suffix = bool(self._HYPHEN_SUFFIX.search(base))
        has_greek  = any(c in self._GREEK_CHARS for c in base)
 
        variants = set()
        #variants.add(raw_form)   # original form (preserves case)
        variants.add(base)       # lowercased normalized form
 
        # Greek expansion of base (no hyphen change)
        if has_greek:
            variants.add(self._expand_greek(base))
 
        if has_suffix:
            # hyphen → space
            space_form = self._HYPHEN_SUFFIX.sub(r" \1", base)
            variants.add(space_form)
            if any(c in self._GREEK_CHARS for c in space_form):
                variants.add(self._expand_greek(space_form))
 
            # hyphen removed
            no_hyph = self._HYPHEN_SUFFIX.sub(r"\1", base)
            variants.add(no_hyph)
            if any(c in self._GREEK_CHARS for c in no_hyph):
                variants.add(self._expand_greek(no_hyph))
 
        return sorted(variants)
 
    def _expand_greek(self, text: str) -> str:
        """Replace Greek unicode characters with spelled-out equivalents."""
        for ch, exp in self.GREEK.items():
            text = text.replace(ch, exp)
        return text
 
    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
 
    def _classify(self, raw_form: str) -> str:
        """
        Three classes based on opacity vs transparency of the name base.
 
        multi_token        : two or more whitespace-separated tokens.
 
        acronym            : single token whose base (part before any
                             hyphen-digit or hyphen-Greek suffix) is opaque.
                             Opaque = contains uppercase, digit, Greek char,
                             or is shorter than compact_min_length (default 5).
                             Examples: BRCA1, p53, hsp, Hsp10, TNF, TNF-α,
                                       c-Myc, IL-6, CARP-2
 
        descriptive_single : single token whose base is a transparent readable
                             word — all lowercase alpha, no digits, no Greek,
                             no uppercase, length >= compact_min_length.
                             A hyphen-digit or hyphen-Greek suffix is allowed
                             and does not change the class.
                             Examples: laminin, laminin-10, laminin-α,
                                       integrin-β1, caspase, ubiquitin
 
        Tuning: adjust compact_min_length (default 5) to shift the boundary
        for short lowercase base words.
        """
        tokens = raw_form.strip().split()
        if len(tokens) > 1:
            return "multi_token"
        token = tokens[0]
        # Isolate base — strip any hyphen-digit or hyphen-Greek suffix
        base = self._BASE_SPLIT.split(token)[0]
        has_upper  = any(c.isupper() for c in base[1:])
        has_digit  = any(c.isdigit() for c in base)
        has_greek  = any(c in self._GREEK_CHARS for c in base)
        base_clean = base.replace("-", "")
        pure_lower = base_clean.isalpha() and base_clean.islower()
        short      = len(base) < self.compact_min_length
        if has_upper or has_digit or has_greek or short:
            return "acronym"
        return "descriptive_single"
 
    def _token_count(self, raw_form: str) -> int:
        return len(raw_form.strip().split())
 
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
 
    def _empty_report(self) -> dict:
        return {
            "unique_norm_keys":            0,
            "normalization_merges":        0,
            "skipped_empty_after_norm":    0,
            "multi_accession_keys":        0,
            "too_short_count":             0,
            "pattern_counts": {
                "acronym":           0,
                "descriptive_single": 0,
                "multi_token":       0,
            },
            "token_count_distribution": {},
            "sample_normalization_merges": [],
            "sample_multi_accession":      [],
            "sample_too_short":            [],
            "sample_by_pattern": {
                "acronym":           [],
                "descriptive_single": [],
                "multi_token":       [],
            },
        }
 
    def _compute_report(self, preprocessed: dict, report: dict) -> dict:
        N = self.n_merge_samples
        P = self.n_pattern_samples
 
        for norm_key, entry in preprocessed.items():
            pattern = entry["name_pattern"]
            report["pattern_counts"][pattern] += 1
 
            tc = str(entry["token_count"])
            report["token_count_distribution"][tc] = \
                report["token_count_distribution"].get(tc, 0) + 1
 
            unique_acc = {r["accession"] for r in entry["match_records"]}
            if len(unique_acc) > 1:
                report["multi_accession_keys"] += 1
                if len(report["sample_multi_accession"]) < N:
                    report["sample_multi_accession"].append({
                        "norm_key":     norm_key,
                        "originals":    entry["original_forms"],
                        "n_accessions": len(unique_acc),
                    })
 
            if entry["flags"]["too_short"]:
                report["too_short_count"] += 1
                if len(report["sample_too_short"]) < N:
                    report["sample_too_short"].append({
                        "norm_key":  norm_key,
                        "originals": entry["original_forms"],
                    })
 
            if len(entry["original_forms"]) > 1:
                if len(report["sample_normalization_merges"]) < N:
                    report["sample_normalization_merges"].append({
                        "norm_key":  norm_key,
                        "originals": entry["original_forms"],
                    })
 
            # Per-pattern samples: (original, norm_key) for manual inspection
            pat_samples = report["sample_by_pattern"][pattern]
            if len(pat_samples) < P:
                pat_samples.append({
                    "norm_key":  norm_key,
                    "originals": entry["original_forms"],
                })
 
        return report
 
    def _print_report(self, report: dict):
        sep = "─" * 60
        print(f"\n  ── Preprocessing Report  "
              f"(strip_whitespace={self.strip_whitespace})")
        print(f"  {sep}")
        print(f"  Unique normalized keys      : {report['unique_norm_keys']:>8,}")
        print(f"  Normalization merges        : {report['normalization_merges']:>8,}")
        print(f"  Multi-accession keys        : {report['multi_accession_keys']:>8,}")
        print(f"  Too-short flags (<{self.min_length})        : {report['too_short_count']:>8,}")
        print(f"  Skipped (empty after norm)  : {report['skipped_empty_after_norm']:>8,}")
 
        print(f"\n  Name pattern distribution:")
        for pat, cnt in report["pattern_counts"].items():
            print(f"    {pat:<15} : {cnt:,}")
 
        print(f"\n  Token count distribution (top 8):")
        dist = sorted(report["token_count_distribution"].items(),
                      key=lambda x: int(x[0]))
        for tc, cnt in dist[:8]:
            print(f"    {tc} token(s) : {cnt:,}")
        if len(dist) > 8:
            print(f"    ... ({len(dist) - 8} further levels)")
 
        print(f"\n  Name pattern examples "
              f"(n_pattern_samples={self.n_pattern_samples}):")
        for pat, samples in report["sample_by_pattern"].items():
            print(f"\n    [{pat}]")
            for s in samples:
                for orig in s["originals"]:
                    print(f"      {orig!r:50s}  ->  {s['norm_key']!r}")
 
        print(f"\n  Sample — normalization merges "
              f"(n={self.n_merge_samples}):")
        for s in report["sample_normalization_merges"]:
            print(f"    [{s['norm_key']}]  ←  {s['originals']}")
 
        print(f"\n  Sample — multi-accession keys "
              f"(n={self.n_merge_samples}):")
        for s in report["sample_multi_accession"]:
            print(f"    [{s['norm_key']}]  "
                  f"({s['n_accessions']} accessions)  "
                  f"originals: {s['originals']}")
 
        print(f"\n  Sample — too-short entries "
              f"(n={self.n_merge_samples}):")
        for s in report["sample_too_short"]:
            print(f"    [{s['norm_key']}]  ←  {s['originals']}")
 
        print(f"  {sep}\n")
 
 
def save_stage2(preprocessed: dict, report: dict, suffix: str = ""):
    prep_path   = OUT_DIR / f"protein_preprocessed{suffix}.json"
    report_path = OUT_DIR / f"preprocessing_report{suffix}.json"
    print(f"  Saving -> {prep_path}")
    with open(prep_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed, f, ensure_ascii=False, indent=2)
    print(f"  Saving -> {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
 
 
# ===========================================================================
# Entry point
# ===========================================================================
 
if __name__ == "__main__":
    args           = set(sys.argv[1:])
    run_parse      = "--preprocess" not in args
    run_preprocess = "--parse"      not in args
 
    if run_parse:
        name_index, entries = build_dictionary(XML_PATH)
        save_stage1(name_index, entries)
    else:
        print("\n[Stage 1] Skipped — loading from disk ...")
        with open(INDEX_PATH, encoding="utf-8") as f:
            name_index = json.load(f)
        print(f"  Loaded {len(name_index):,} surface forms from {INDEX_PATH}")
 
    if run_preprocess:
        for strip_ws in [False, True]:
            suffix = "_ws_stripped" if strip_ws else ""
            preprocessor = DictionaryPreprocessor(
                min_length=3,
                strip_whitespace=strip_ws,
                n_pattern_samples=100,   # adjust to see more/fewer examples per class
                n_merge_samples=20,    # adjust for merges / multi-acc / too-short
            )
            preprocessed, report = preprocessor.run(name_index, verbose=True)
            save_stage2(preprocessed, report, suffix=suffix)
 
    print("\nAll done.")
