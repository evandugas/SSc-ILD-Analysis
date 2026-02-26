"""
Test suite for src/analysis.py

Coverage areas
--------------
1. parse_multi_sample_series_matrix – condition classification heuristic
2. load_gpl                         – probe-column detection, gene-symbol cleaning,
                                      ValueError guards
3. load_series_matrix               – numeric coercion, unmapped-probe removal,
                                      duplicate-gene collapsing
4. find_common_genes / clean_index  – multi-dataset integration helpers
5. run_differential_expression      – OLS regression, FDR correction
6. filter_significant_genes         – significance threshold logic
7. parse_gene_ratio                 – GO enrichment overlap string parsing
"""

import io
import textwrap
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import (
    parse_multi_sample_series_matrix,
    load_gpl,
    load_series_matrix,
    find_common_genes,
    clean_index,
    run_differential_expression,
    filter_significant_genes,
    parse_gene_ratio,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _series_matrix_text(samples: list[dict]) -> str:
    """
    Build a GEO series_matrix text fixture (not gzip-compressed).

    Each sample dict maps field_name → value.  The first field must be
    geo_accession.
    """
    lines = []
    fields = list(samples[0].keys())
    for field in fields:
        values = "\t".join(f'"{s[field]}"' for s in samples)
        lines.append(f"!Sample_{field}\t{values}")
    lines.append("# comment – should be skipped")
    return "\n".join(lines) + "\n"


def _gpl_text(columns: list[str], rows: list[list[str]]) -> str:
    """Build a GPL annotation text fixture (not compressed)."""
    lines = ["!platform_table_begin", "\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row))
    lines.append("!platform_table_end")
    return "\n".join(lines) + "\n"


def _series_expr_text(
    probes: list[str],
    samples: list[str],
    values: list[list],
) -> str:
    """Build a GEO series_matrix expression-table text fixture (not compressed)."""
    header = "\t".join(["\"ID_REF\""] + [f'"{s}"' for s in samples])
    lines = ["!series_matrix_table_begin", header]
    for probe, row in zip(probes, values):
        lines.append("\t".join([probe] + [str(v) for v in row]))
    lines.append("!series_matrix_table_end")
    return "\n".join(lines) + "\n"


# ===========================================================================
# 1. parse_multi_sample_series_matrix
# ===========================================================================

class TestParseMultiSampleSeriesMatrix:
    """Tests for the condition-classification heuristic."""

    def _run(self, samples: list[dict], study_id: str = "GSE_TEST"):
        text = _series_matrix_text(samples)
        with patch("src.analysis.gzip.open", return_value=io.StringIO(text)):
            return parse_multi_sample_series_matrix("fake.gz", study_id)

    # --- Classification happy paths ---

    def test_control_via_normal_keyword(self):
        """'normal' in text without 'ssc' → Control."""
        df = self._run([{"geo_accession": "GSM001",
                         "title": "Normal lung tissue",
                         "characteristics_ch1": "tissue: lung"}])
        assert df.loc["GSM001", "condition"] == "Control"

    def test_control_via_control_keyword(self):
        """'control' in text without 'ssc' → Control."""
        df = self._run([{"geo_accession": "GSM002",
                         "title": "Healthy control",
                         "characteristics_ch1": "subject: healthy"}])
        assert df.loc["GSM002", "condition"] == "Control"

    def test_ssc_ild_via_lung_keyword(self):
        """'ssc' + 'lung' → SSc-ILD."""
        df = self._run([{"geo_accession": "GSM003",
                         "title": "SSc patient lung biopsy",
                         "characteristics_ch1": "disease: systemic sclerosis"}])
        assert df.loc["GSM003", "condition"] == "SSc-ILD"

    def test_ssc_ild_via_fibrosis_keyword(self):
        """'ssc' + 'fibrosis' → SSc-ILD."""
        df = self._run([{"geo_accession": "GSM004",
                         "title": "SSc pulmonary fibrosis sample",
                         "characteristics_ch1": "status: fibrosis"}])
        assert df.loc["GSM004", "condition"] == "SSc-ILD"

    def test_ssc_ild_via_ild_keyword(self):
        """'ssc' + 'ild' → SSc-ILD."""
        df = self._run([{"geo_accession": "GSM005",
                         "title": "SSc-ILD patient",
                         "characteristics_ch1": "disease: ILD"}])
        assert df.loc["GSM005", "condition"] == "SSc-ILD"

    # --- Ambiguous / edge cases that yield "Other" ---

    def test_ssc_without_lung_fibrosis_ild_is_other(self):
        """
        'ssc' without lung/fibrosis/ild → Other.

        This is a known classification gap: SSc patients without lung/ILD
        keywords in their metadata fall through to "Other" even though they
        have SSc. A future improvement would add a dedicated SSc-non-ILD rule.
        """
        df = self._run([{"geo_accession": "GSM006",
                         "title": "SSc skin biopsy",
                         "characteristics_ch1": "tissue: skin"}])
        assert df.loc["GSM006", "condition"] == "Other"

    def test_ssc_and_normal_is_other(self):
        """
        Text containing both 'ssc' and 'normal' → Other.

        The Control branch requires 'ssc' NOT be present.  Once 'ssc' appears,
        the Control branch fails.  Without lung/fibrosis/ild, the sample falls
        through to Other.  This edge case deserves a dedicated rule or a warning.
        """
        df = self._run([{"geo_accession": "GSM007",
                         "title": "SSc patient with normal spirometry",
                         "characteristics_ch1": "subject: SSc"}])
        assert df.loc["GSM007", "condition"] == "Other"

    def test_completely_unrelated_text_is_other(self):
        """Unrelated metadata → Other."""
        df = self._run([{"geo_accession": "GSM008",
                         "title": "Cancer cell line",
                         "characteristics_ch1": "cell_line: HeLa"}])
        assert df.loc["GSM008", "condition"] == "Other"

    # --- Multi-sample batch ---

    def test_mixed_batch_classifies_all_correctly(self):
        """Multiple samples in one file are all classified independently."""
        samples = [
            {"geo_accession": "GSM010", "title": "Normal donor",
             "characteristics_ch1": "healthy control"},
            {"geo_accession": "GSM011", "title": "SSc ILD lung",
             "characteristics_ch1": "SSc patient lung fibrosis"},
            {"geo_accession": "GSM012", "title": "Skin SSc",
             "characteristics_ch1": "SSc biopsy"},
        ]
        df = self._run(samples)
        assert df.loc["GSM010", "condition"] == "Control"
        assert df.loc["GSM011", "condition"] == "SSc-ILD"
        assert df.loc["GSM012", "condition"] == "Other"

    # --- Metadata fields ---

    def test_study_id_assigned(self):
        """study column reflects the study_id argument."""
        df = self._run([{"geo_accession": "GSM020", "title": "Normal",
                         "characteristics_ch1": "control"}], study_id="GSE_XYZ")
        assert df.loc["GSM020", "study"] == "GSE_XYZ"

    def test_sample_id_is_index(self):
        """geo_accession values become the DataFrame index."""
        df = self._run([{"geo_accession": "GSM030", "title": "Normal",
                         "characteristics_ch1": "control"}])
        assert "GSM030" in df.index

    def test_case_insensitive_classification(self):
        """Condition classification is case-insensitive (raw_text is lowercased)."""
        df = self._run([{"geo_accession": "GSM040", "title": "NORMAL LUNG DONOR",
                         "characteristics_ch1": "CONTROL"}])
        assert df.loc["GSM040", "condition"] == "Control"

    def test_non_sample_lines_are_ignored(self):
        """Lines not starting with '!Sample_' are skipped without error."""
        text = (
            "# This is a comment\n"
            "!Series_title\tSome Study\n"
            '!Sample_geo_accession\t"GSM050"\n'
            '!Sample_title\t"Normal donor"\n'
            '!Sample_characteristics_ch1\t"healthy control"\n'
        )
        with patch("src.analysis.gzip.open", return_value=io.StringIO(text)):
            df = parse_multi_sample_series_matrix("fake.gz", "GSE_TEST")
        assert "GSM050" in df.index
        assert df.loc["GSM050", "condition"] == "Control"


# ===========================================================================
# 2. load_gpl
# ===========================================================================

class TestLoadGpl:

    def _load(self, columns, rows):
        text = _gpl_text(columns, rows)
        with patch("src.analysis.gzip.open", return_value=io.StringIO(text)):
            return load_gpl("fake.gz")

    def test_basic_id_column(self):
        """Parses correctly when probe column is named 'ID'."""
        mapping = self._load(["ID", "Gene Symbol"],
                             [["probe_1", "Gapdh"], ["probe_2", "Actb"]])
        assert mapping["probe_1"] == "GAPDH"
        assert mapping["probe_2"] == "ACTB"

    def test_id_ref_probe_column(self):
        """Probe column 'ID_REF' is recognised."""
        mapping = self._load(["ID_REF", "Gene Symbol"], [["p1", "Tp53"]])
        assert "p1" in mapping

    def test_probe_id_column(self):
        """Probe column 'PROBE_ID' is recognised."""
        mapping = self._load(["PROBE_ID", "symbol"], [["p1", "Myc"]])
        assert mapping["p1"] == "MYC"

    def test_gene_symbol_uppercased(self):
        """Gene symbols are normalised to uppercase."""
        mapping = self._load(["ID", "symbol"], [["p1", "gapdh"]])
        assert mapping["p1"] == "GAPDH"

    def test_triple_slash_suffix_stripped(self):
        """'GENE_A /// GENE_B' is collapsed to 'GENE_A'."""
        mapping = self._load(["ID", "symbol"], [["p1", "Gene_A /// Gene_B"]])
        assert mapping["p1"] == "GENE_A"

    def test_raises_if_no_probe_column(self):
        """ValueError when no recognised probe column is found."""
        with pytest.raises(ValueError, match="No probe column"):
            self._load(["accession", "symbol"], [["p1", "GAPDH"]])

    def test_raises_if_no_gene_symbol_column(self):
        """ValueError when no gene symbol column is found."""
        with pytest.raises(ValueError, match="No gene symbol column"):
            self._load(["ID", "description", "platform"],
                       [["p1", "desc", "GPL"]])

    def test_mapping_length(self):
        """Returned dict has one entry per data row."""
        rows = [[f"p{i}", f"GENE{i}"] for i in range(5)]
        mapping = self._load(["ID", "symbol"], rows)
        assert len(mapping) == 5

    def test_lines_outside_table_ignored(self):
        """Metadata lines before !platform_table_begin are silently skipped."""
        text = (
            "!Platform_title\tHuman Genome\n"
            "!platform_table_begin\n"
            "ID\tsymbol\n"
            "p1\tGAPDH\n"
            "!platform_table_end\n"
        )
        with patch("src.analysis.gzip.open", return_value=io.StringIO(text)):
            mapping = load_gpl("fake.gz")
        assert mapping == {"p1": "GAPDH"}

    def test_column_matching_is_case_insensitive(self):
        """Column names are lower-cased before matching."""
        mapping = self._load(["ID", "Gene_Symbol"], [["p1", "Actb"]])
        # 'gene_symbol' contains 'symbol' → accepted
        assert mapping["p1"] == "ACTB"


# ===========================================================================
# 3. load_series_matrix
# ===========================================================================

class TestLoadSeriesMatrix:
    """
    load_series_matrix opens the file with gzip.open to read the header row,
    then delegates the rest to pd.read_csv.  We patch both.
    """

    def _load(self, probes, samples, values, gpl_map):
        # Use unquoted names so that after `df.columns = header` the sample
        # columns are clean strings like "S1" rather than '"S1"'.
        header_text = (
            "!series_matrix_table_begin\n"
            + "\t".join(["ID_REF"] + samples)
            + "\n"
        )

        # Build the DataFrame that pd.read_csv would normally return.
        # It has no header (header=None), so columns are positional ints.
        rows = [[probe] + [str(v) for v in row]
                for probe, row in zip(probes, values)]
        header_row = ["ID_REF"] + samples
        raw_df = pd.DataFrame([header_row] + rows)

        with patch("src.analysis.gzip.open",
                   return_value=io.StringIO(header_text)), \
             patch("pandas.read_csv", return_value=raw_df):
            return load_series_matrix("fake.gz", gpl_map)

    def test_unmapped_probes_are_dropped(self):
        """Probes absent from gpl_map should not appear in the output."""
        gpl_map = {"probe_A": "GAPDH"}  # probe_B intentionally absent
        gene_df = self._load(
            probes=["probe_A", "probe_B"],
            samples=["S1", "S2"],
            values=[[1.0, 2.0], [3.0, 4.0]],
            gpl_map=gpl_map,
        )
        assert "GAPDH" in gene_df.index
        assert gene_df.shape[0] == 1

    def test_duplicate_genes_averaged(self):
        """Multiple probes mapping to the same gene are collapsed by mean."""
        gpl_map = {"probe_A": "GAPDH", "probe_B": "GAPDH"}
        gene_df = self._load(
            probes=["probe_A", "probe_B"],
            samples=["S1"],
            values=[[2.0], [4.0]],
            gpl_map=gpl_map,
        )
        assert gene_df.loc["GAPDH", "S1"] == pytest.approx(3.0)

    def test_non_numeric_values_become_nan(self):
        """String probe values are coerced to NaN rather than raising.

        A second gene (probe_B → ACTB) ensures column S1 is not all-NaN after
        coercion, which would otherwise trigger the all-NaN column-drop logic.
        """
        gpl_map = {"probe_A": "GAPDH", "probe_B": "ACTB"}
        gene_df = self._load(
            probes=["probe_A", "probe_B"],
            samples=["S1", "S2"],
            # probe_A: S1 is "null" (→ NaN), S2 is 5.0
            # probe_B: both samples have valid numeric values
            values=[["null", 5.0], [3.0, 4.0]],
            gpl_map=gpl_map,
        )
        assert np.isnan(gene_df.loc["GAPDH", "S1"])
        assert gene_df.loc["GAPDH", "S2"] == pytest.approx(5.0)

    def test_all_nan_column_is_dropped(self):
        """A sample column whose every value is NaN is removed from output."""
        gpl_map = {"probe_A": "GAPDH", "probe_B": "ACTB"}
        gene_df = self._load(
            probes=["probe_A", "probe_B"],
            samples=["S1", "S2"],
            values=[[1.0, float("nan")], [2.0, float("nan")]],
            gpl_map=gpl_map,
        )
        assert "S2" not in gene_df.columns

    def test_multiple_genes_returned(self):
        """Distinct probes mapping to distinct genes all appear in output."""
        gpl_map = {"p1": "GAPDH", "p2": "ACTB", "p3": "TP53"}
        gene_df = self._load(
            probes=["p1", "p2", "p3"],
            samples=["S1"],
            values=[[1.0], [2.0], [3.0]],
            gpl_map=gpl_map,
        )
        assert set(gene_df.index) == {"GAPDH", "ACTB", "TP53"}


# ===========================================================================
# 4. find_common_genes / clean_index
# ===========================================================================

class TestFindCommonGenes:

    def test_two_overlapping_sets(self):
        common = find_common_genes(["A", "B", "C"], ["B", "C", "D"])
        assert common == ["B", "C"]

    def test_three_overlapping_sets(self):
        common = find_common_genes(["A", "B", "C"], ["B", "C", "D"], ["C", "D", "E"])
        assert common == ["C"]

    def test_disjoint_sets_return_empty(self):
        common = find_common_genes(["A", "B"], ["C", "D"])
        assert common == []

    def test_result_is_sorted(self):
        common = find_common_genes(["Z", "A", "M"], ["M", "Z", "A"])
        assert common == sorted(common)

    def test_single_set_returns_sorted_copy(self):
        common = find_common_genes(["C", "A", "B"])
        assert common == ["A", "B", "C"]

    def test_preserves_all_when_identical(self):
        genes = ["GAPDH", "ACTB", "TP53"]
        assert set(find_common_genes(genes, genes)) == set(genes)


class TestCleanIndex:

    def test_strips_double_quotes(self):
        idx = pd.Index(['"GSM001"', '"GSM002"'])
        assert list(clean_index(idx)) == ["GSM001", "GSM002"]

    def test_strips_single_quotes(self):
        idx = pd.Index(["'GSM001'"])
        assert list(clean_index(idx)) == ["GSM001"]

    def test_strips_surrounding_whitespace(self):
        idx = pd.Index(["  GSM001  "])
        assert list(clean_index(idx)) == ["GSM001"]

    def test_no_change_for_clean_index(self):
        idx = pd.Index(["GSM001", "GSM002"])
        assert list(clean_index(idx)) == ["GSM001", "GSM002"]

    def test_strips_mixed_quotes_and_spaces(self):
        idx = pd.Index([' "GSM001" '])
        assert list(clean_index(idx)) == ["GSM001"]


# ===========================================================================
# 5. run_differential_expression
# ===========================================================================

def _make_de_inputs(n_control=5, n_case=5, seed=42):
    """
    Construct a small synthetic gene × sample expression matrix + metadata.

    - GENE_UP   : higher in cases  (mean +4 vs +8  → logFC ≈ +4)
    - GENE_DOWN : lower in cases   (mean +8 vs +4  → logFC ≈ -4)
    - GENE_FLAT : no difference    (same distribution in both groups)
    """
    rng = np.random.default_rng(seed)
    samples = [f"ctrl_{i}" for i in range(n_control)] + [f"case_{i}" for i in range(n_case)]
    meta = pd.DataFrame(
        {"condition": ["Control"] * n_control + ["SSc-ILD"] * n_case},
        index=samples,
    )

    data: dict[str, dict[str, float]] = {s: {} for s in samples}

    base = rng.normal(5, 0.2, n_control + n_case)
    for i, s in enumerate(samples):
        data[s]["GENE_FLAT"] = float(base[i])

    for i, s in enumerate(samples):
        if i < n_control:
            data[s]["GENE_UP"] = float(rng.normal(4, 0.2))
        else:
            data[s]["GENE_UP"] = float(rng.normal(8, 0.2))

    for i, s in enumerate(samples):
        if i < n_control:
            data[s]["GENE_DOWN"] = float(rng.normal(8, 0.2))
        else:
            data[s]["GENE_DOWN"] = float(rng.normal(4, 0.2))

    # pd.DataFrame(data): columns=samples, index=genes → gene × sample (correct)
    expr = pd.DataFrame(data)
    return expr, meta


class TestRunDifferentialExpression:

    def test_returns_expected_columns(self):
        expr, meta = _make_de_inputs()
        de = run_differential_expression(expr, meta)
        assert {"logFC", "pval", "FDR"}.issubset(de.columns)

    def test_upregulated_gene_has_positive_logfc(self):
        expr, meta = _make_de_inputs()
        de = run_differential_expression(expr, meta)
        assert de.loc["GENE_UP", "logFC"] > 1.0

    def test_downregulated_gene_has_negative_logfc(self):
        expr, meta = _make_de_inputs()
        de = run_differential_expression(expr, meta)
        assert de.loc["GENE_DOWN", "logFC"] < -1.0

    def test_flat_gene_has_nonsignificant_pvalue(self):
        expr, meta = _make_de_inputs(n_control=10, n_case=10)
        de = run_differential_expression(expr, meta)
        assert de.loc["GENE_FLAT", "pval"] > 0.05

    def test_de_genes_have_significant_fdr(self):
        """Strongly separated genes should survive FDR correction."""
        expr, meta = _make_de_inputs(n_control=10, n_case=10)
        de = run_differential_expression(expr, meta)
        assert de.loc["GENE_UP", "FDR"] < 0.05
        assert de.loc["GENE_DOWN", "FDR"] < 0.05

    def test_sorted_by_fdr_ascending(self):
        expr, meta = _make_de_inputs()
        de = run_differential_expression(expr, meta)
        assert list(de["FDR"]) == sorted(de["FDR"])

    def test_all_genes_have_index_in_output(self):
        expr, meta = _make_de_inputs()
        de = run_differential_expression(expr, meta)
        assert set(expr.index).issubset(de.index)

    def test_fdr_values_between_0_and_1(self):
        expr, meta = _make_de_inputs()
        de = run_differential_expression(expr, meta)
        assert (de["FDR"] >= 0).all() and (de["FDR"] <= 1).all()

    def test_pval_values_between_0_and_1(self):
        expr, meta = _make_de_inputs()
        de = run_differential_expression(expr, meta)
        assert (de["pval"] >= 0).all() and (de["pval"] <= 1).all()

    def test_custom_case_label(self):
        """The case_label parameter is respected; flipping it inverts logFC signs."""
        expr, meta = _make_de_inputs()
        de_case = run_differential_expression(expr, meta, case_label="SSc-ILD")
        de_ctrl = run_differential_expression(expr, meta, case_label="Control")
        assert np.sign(de_case.loc["GENE_UP", "logFC"]) != np.sign(
            de_ctrl.loc["GENE_UP", "logFC"]
        )

    def test_samples_not_in_meta_are_excluded(self):
        """Samples present in expr but absent from meta are silently dropped."""
        expr, meta = _make_de_inputs(n_control=5, n_case=5)
        # Add an extra sample column to expr that has no metadata row
        expr["orphan_sample"] = 5.0
        de = run_differential_expression(expr, meta)
        # Should still produce results for all genes; orphan sample is ignored
        assert set(expr.index).issubset(de.index)


# ===========================================================================
# 6. filter_significant_genes
# ===========================================================================

class TestFilterSignificantGenes:

    def _make_de(self):
        return pd.DataFrame(
            {
                "logFC": [2.5, -1.5, 0.3, 2.0, -0.8],
                "pval":  [0.001, 0.002, 0.9, 0.04, 0.001],
                "FDR":   [0.01, 0.02, 0.95, 0.06, 0.04],
            },
            index=["UP_SIG", "DOWN_SIG", "FLAT_NS", "UP_NS_FDR", "DOWN_SMALL_FC"],
        )

    def test_both_thresholds_applied(self):
        de = self._make_de()
        sig = filter_significant_genes(de)
        assert set(sig.index) == {"UP_SIG", "DOWN_SIG"}

    def test_fdr_threshold_respected(self):
        """Gene with FDR just above threshold is excluded."""
        sig = filter_significant_genes(self._make_de(), fdr_threshold=0.05)
        assert "UP_NS_FDR" not in sig.index  # FDR=0.06 > 0.05

    def test_logfc_threshold_respected(self):
        """Gene with |logFC| just below threshold is excluded."""
        sig = filter_significant_genes(self._make_de(), logfc_threshold=1.0)
        assert "DOWN_SMALL_FC" not in sig.index  # |logFC|=0.8 < 1.0

    def test_empty_when_nothing_passes(self):
        de = pd.DataFrame(
            {"logFC": [0.1, -0.2], "pval": [0.5, 0.6], "FDR": [0.8, 0.9]},
            index=["G1", "G2"],
        )
        assert filter_significant_genes(de).empty

    def test_all_pass_with_lenient_thresholds(self):
        de = self._make_de()
        sig = filter_significant_genes(de, fdr_threshold=1.1, logfc_threshold=0.0)
        assert len(sig) == len(de)

    def test_custom_thresholds(self):
        # Only UP_SIG passes FDR<0.015 AND |logFC|>2.0
        sig = filter_significant_genes(self._make_de(), fdr_threshold=0.015, logfc_threshold=2.0)
        assert list(sig.index) == ["UP_SIG"]

    def test_returns_dataframe(self):
        assert isinstance(filter_significant_genes(self._make_de()), pd.DataFrame)

    def test_output_columns_preserved(self):
        de = self._make_de()
        sig = filter_significant_genes(de)
        assert set(de.columns) == set(sig.columns)

    def test_boundary_fdr_excluded(self):
        """FDR exactly equal to threshold should be excluded (strict <)."""
        de = pd.DataFrame(
            {"logFC": [2.0], "pval": [0.01], "FDR": [0.05]},
            index=["BOUNDARY"],
        )
        sig = filter_significant_genes(de, fdr_threshold=0.05)
        assert "BOUNDARY" not in sig.index

    def test_boundary_logfc_excluded(self):
        """|logFC| exactly equal to threshold should be excluded (strict >)."""
        de = pd.DataFrame(
            {"logFC": [1.0], "pval": [0.001], "FDR": [0.01]},
            index=["BOUNDARY"],
        )
        sig = filter_significant_genes(de, logfc_threshold=1.0)
        assert "BOUNDARY" not in sig.index


# ===========================================================================
# 7. parse_gene_ratio
# ===========================================================================

class TestParseGeneRatio:

    def test_basic_ratio(self):
        assert parse_gene_ratio("12/310") == pytest.approx(12 / 310)

    def test_ratio_of_one(self):
        assert parse_gene_ratio("1/1") == pytest.approx(1.0)

    def test_ratio_of_zero_numerator(self):
        assert parse_gene_ratio("0/100") == pytest.approx(0.0)

    def test_large_denominator(self):
        assert parse_gene_ratio("50/10000") == pytest.approx(0.005)

    def test_returns_float(self):
        assert isinstance(parse_gene_ratio("3/4"), float)

    def test_equal_numerator_denominator(self):
        assert parse_gene_ratio("7/7") == pytest.approx(1.0)

    def test_malformed_string_raises(self):
        """Non-fraction strings should raise an exception."""
        with pytest.raises(Exception):
            parse_gene_ratio("not_a_ratio")
