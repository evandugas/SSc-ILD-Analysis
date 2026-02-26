"""
Core analysis functions for SSc-ILD gene expression pipeline.

Functions are extracted from analysis.ipynb so they can be independently
tested and reused.
"""

import gzip
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_multi_sample_series_matrix(path, study_id):
    """
    Parse GEO series_matrix with multi-sample columns.

    Returns a DataFrame indexed by GSM (sample_id) with columns:
      study, raw_text, condition

    Condition assignment rules (applied to lower-cased concatenated text):
      - "Control"  : contains "normal" or "control", AND does not contain "ssc"
      - "SSc-ILD"  : contains "ssc" AND at least one of "lung", "fibrosis", "ild"
      - "Other"    : everything else
    """
    col_data = {}

    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("!Sample_"):
                continue
            parts = line.strip().split("\t")
            field = parts[0].replace("!Sample_", "")
            values = [v.strip('"') for v in parts[1:]]
            col_data[field] = values

    df = pd.DataFrame(col_data)
    df = df.rename(columns={"geo_accession": "sample_id"})
    df["study"] = study_id

    text_fields = [c for c in df.columns if c not in ["sample_id", "study"]]
    df["raw_text"] = df[text_fields].astype(str).agg(" ; ".join, axis=1).str.lower()

    cond = []
    for txt in df["raw_text"]:
        if ("normal" in txt or "control" in txt) and "ssc" not in txt:
            cond.append("Control")
        elif "ssc" in txt and ("lung" in txt or "fibrosis" in txt or "ild" in txt):
            cond.append("SSc-ILD")
        else:
            cond.append("Other")
    df["condition"] = cond

    df = df.set_index("sample_id")
    return df


def load_gpl(gpl_path):
    """
    Parse a GPL annotation file (gzipped) and return a probe→gene dict.

    Raises ValueError if no probe column or no gene symbol column is found.
    """
    rows = []
    inside = False
    with gzip.open(gpl_path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                inside = True
                continue
            if line.startswith("!platform_table_end"):
                inside = False
                continue
            if inside:
                rows.append(line.rstrip("\n"))

    df = pd.DataFrame(
        [r.split("\t") for r in rows[1:]],
        columns=rows[0].split("\t"),
    )
    df.columns = df.columns.str.lower()

    probe_col = None
    for c in ["id", "id_ref", "probe_id"]:
        if c in df.columns:
            probe_col = c
            break
    if probe_col is None:
        raise ValueError("No probe column in GPL")

    gene_col = None
    for c in df.columns:
        if "symbol" in c or c == "gene":
            gene_col = c
            break
    if gene_col is None:
        raise ValueError("No gene symbol column in GPL")

    df = df[[probe_col, gene_col]].copy()
    df.columns = ["probe", "gene"]
    df["gene"] = df["gene"].str.upper().str.replace("///.*", "", regex=True).str.strip()

    mapping = df.set_index("probe")["gene"].to_dict()
    return mapping


def load_series_matrix(path, gpl_map):
    """
    Load gene-level expression matrix from a GEO series_matrix file.

    Steps:
      1. Skip metadata lines (starting with "!")
      2. Map probe IDs → gene symbols using gpl_map
      3. Drop probes with no gene mapping
      4. Collapse duplicate gene symbols by mean
    """
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
        header = next(f).strip().split("\t")

    df = pd.read_csv(path, sep="\t", comment="!", header=None, low_memory=False)
    df.columns = header

    df = df.rename(columns={df.columns[0]: "probe"})
    df = df.set_index("probe")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all")

    df["gene"] = df.index.map(gpl_map).fillna("")
    df = df[df["gene"] != ""]

    gene_df = df.groupby("gene").mean(numeric_only=True)
    return gene_df


# ---------------------------------------------------------------------------
# Multi-dataset integration
# ---------------------------------------------------------------------------

def find_common_genes(*gene_sets):
    """Return a sorted list of genes present in all provided sets/iterables."""
    result = set(gene_sets[0])
    for gs in gene_sets[1:]:
        result &= set(gs)
    return sorted(result)


def clean_index(index):
    """Strip quotes and surrounding whitespace from a pandas Index."""
    return index.str.replace('"', "", regex=False).str.replace("'", "", regex=False).str.strip()


# ---------------------------------------------------------------------------
# Differential expression
# ---------------------------------------------------------------------------

def run_differential_expression(expr, meta, condition_col="condition",
                                 case_label="SSc-ILD"):
    """
    Run per-gene OLS regression comparing case_label vs all others in meta.

    Parameters
    ----------
    expr : pd.DataFrame
        Gene × sample expression matrix (already batch-corrected).
    meta : pd.DataFrame
        Sample metadata; must have index aligned with expr columns.
    condition_col : str
        Column in meta that holds condition labels.
    case_label : str
        The positive class (e.g. "SSc-ILD").

    Returns
    -------
    pd.DataFrame with columns: logFC, pval, FDR  (sorted by FDR ascending)
    """
    meta = meta.copy()
    meta["condition_bin"] = (meta[condition_col] == case_label).astype(int)
    X = sm.add_constant(meta["condition_bin"]).astype(float)

    samples = expr.columns.intersection(meta.index)
    X = X.loc[samples]

    logFC_list, pval_list = [], []
    for gene in expr.index:
        y = expr.loc[gene, samples].astype(float).values
        model = sm.OLS(y, X).fit()
        logFC_list.append(model.params["condition_bin"])
        pval_list.append(model.pvalues["condition_bin"])

    de = pd.DataFrame({"logFC": logFC_list, "pval": pval_list}, index=expr.index)
    de["FDR"] = multipletests(de["pval"], method="fdr_bh")[1]
    return de.sort_values("FDR")


def filter_significant_genes(de, fdr_threshold=0.05, logfc_threshold=1.0):
    """
    Return the subset of DE results that pass both FDR and |logFC| thresholds.
    """
    mask = (de["FDR"] < fdr_threshold) & (de["logFC"].abs() > logfc_threshold)
    return de.loc[mask]


def parse_gene_ratio(overlap_str):
    """
    Convert a GO enrichment Overlap string like '12/310' to a float ratio.
    """
    parts = overlap_str.split("/")
    return int(parts[0]) / int(parts[1])
