import pytest
import pandas as pd
import json
from unittest.mock import patch
from src.cme_evaluator import CMEEvaluator
from config.validation_config import (
    none_valid_status,
    validation_failed_message,
    valid_status,
    partial_valid_status
)

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"status": "valid", "field1": "A", "field1_pred": "A", "field2": "X", "field2_pred": "Y"},
        {"status": "valid", "field1": "B", "field1_pred": "B", "field2": "Y", "field2_pred": "Y"},
    ])

@pytest.fixture
def sample_comparison_dict():
    return {
        "field1": "field1_pred",
        "field2": "field2_pred"
    }

@pytest.fixture
def sample_accepted_values():
    accepted_values = {
        "field1_pred": ["A", "B", "C"],
        "field2_pred": ["T", "X", "V", "U", "W", "Y", "Z"]
    }
    return accepted_values

@pytest.fixture
def sample_final_df():
    sample_final_df = pd.DataFrame([
        {
            "status": "valid", "field1": "A", "field1_pred": "A",
            "field2": "X", "field2_pred": "Y",
            "field1_V_field1_pred": 1,
            "field2_V_field2_pred": 0
        },
        {
            "status": "valid", "field1": "B", "field1_pred": "B",
            "field2": "Y", "field2_pred": "Y",
            "field1_V_field1_pred": 1,
            "field2_V_field2_pred": 1
        }
    ])
    return sample_final_df
    

@pytest.fixture
def evaluator_numeric_correctness():
    df = pd.DataFrame([
        {"status": "valid", "field1": "A", "field1_pred": "A", "field2": "X", "field2_pred": "X"},
        {"status": "valid", "field1": "A", "field1_pred": "B", "field2": "Y", "field2_pred": "Y"},
        {"status": "valid", "field1": "B", "field1_pred": "B", "field2": "Z", "field2_pred": "Y"},
        {"status": "valid", "field1": "B", "field1_pred": None, "field2": None, "field2_pred": "Y"},
        {"status": "valid", "field1": None, "field1_pred": "B", "field2": "W", "field2_pred": "W"},
        {"status": "valid", "field1": None, "field1_pred": None, "field2": None, "field2_pred": None},
        {"status": "valid", "field1": "C", "field1_pred": "C", "field2": "V", "field2_pred": "V"},
        {"status": "valid", "field1": "C", "field1_pred": "A", "field2": "U", "field2_pred": "T"},
    ])

    comparison_dict = {
        "field1": "field1_pred",
        "field2": "field2_pred"
    }

    accepted_values = {
        "field1_pred": ["A", "B", "C"],
        "field2_pred": ["T", "X", "V", "U", "W", "Y", "Z"]
    }

    evaluator = CMEEvaluator(df=df,
                             comparison_dict=comparison_dict,
                             accepted_values=accepted_values,
                             id_col="id_col")
    
    return evaluator

@pytest.fixture
def evaluator_statuses(sample_accepted_values):
    df = pd.DataFrame([
        {"id_col": 1, "status": validation_failed_message, "field1": "A", "field1_pred": None,
         "model_output": "m1", "parsed_output": "p1", "validated_output": "v1"},
        {"id_col": 2, "status": valid_status, "field1": "B", "field1_pred": "B",
         "model_output": "m2", "parsed_output": "p2", "validated_output": "v2"},
        {"id_col": 3, "status": none_valid_status, "field1": "B", "field1_pred": "Z",
         "model_output": "m3", "parsed_output": "p3", "validated_output": "v3"},
        {"id_col": 4, "status": partial_valid_status, "field1": None, "field1_pred": None,
         "model_output": "m4", "parsed_output": "p4", "validated_output": "v4"},
        {"id_col": 5, "status": valid_status, "field1": "B", "field1_pred": "X",
         "model_output": "m5", "parsed_output": "p5", "validated_output": "v5"},
    ])

    comparison_dict = {
        "field1": "field1_pred"
    }
    
    evaluator = CMEEvaluator(
        df=df,
        comparison_dict=comparison_dict,
        accepted_values=sample_accepted_values,
        id_col="id_col"
    )
    return evaluator

@pytest.fixture
def mock_load_dataframe_from_s3(sample_final_df):
    def _mock(bucket_name, object_key):
        return sample_final_df.copy()
    return _mock

@pytest.fixture
def mock_load_from_s3_prompt(sample_final_df):
    def _mock(self, bucket_name, folder):
        return sample_final_df.copy()
    return _mock

class TestCMEEvaluator:

    def test_init_from_df_only(self,
                               sample_df,
                               sample_comparison_dict,
                               sample_accepted_values):
        evaluator = CMEEvaluator(
            comparison_dict=sample_comparison_dict,
            df=sample_df,
            accepted_values=sample_accepted_values,
            id_col="id_col",
        )
        
        assert isinstance(evaluator.df, pd.DataFrame)
        assert evaluator.comparison_dict == sample_comparison_dict
        assert all(col in evaluator.df.columns for col in evaluator.match_cols)
        assert evaluator.rowwise_distribution is not None
        assert evaluator.correctness_summary_df is not None
        assert evaluator.confusion_matrices.keys() == sample_comparison_dict.keys()
        assert evaluator.per_value_metrics.keys() == sample_comparison_dict.keys()

    def test_init_from_s3_direct(self,
                                 sample_comparison_dict,
                                 sample_accepted_values,
                                 mock_load_dataframe_from_s3):
        with patch("src.cme_evaluator.load_dataframe_from_s3", new=mock_load_dataframe_from_s3):
            evaluator = CMEEvaluator(
                comparison_dict=sample_comparison_dict,
                accepted_values=sample_accepted_values,
                id_col="id_col",
                bucket_name="my-bucket",
                folder="folder_name",
                object_key="my-key.csv"
            )
    
        assert isinstance(evaluator.df, pd.DataFrame)
        assert evaluator.comparison_dict == sample_comparison_dict
        assert all(col in evaluator.df.columns for col in evaluator.match_cols)
        assert evaluator.rowwise_distribution is not None
        assert evaluator.correctness_summary_df is not None
        assert evaluator.confusion_matrices.keys() == sample_comparison_dict.keys()
        assert evaluator.per_value_metrics.keys() == sample_comparison_dict.keys()

    def test_init_from_s3_prompt(self,
                                 sample_comparison_dict,
                                 sample_accepted_values,
                                 mock_load_from_s3_prompt):
        with patch.object(CMEEvaluator, "_load_from_s3_prompt", new=mock_load_from_s3_prompt):
            evaluator = CMEEvaluator(
                comparison_dict=sample_comparison_dict,
                bucket_name="my-bucket",
                accepted_values=sample_accepted_values,
                id_col="id_col",
                list_saved=True,
                folder="folder_name"
            )
    
        assert isinstance(evaluator.df, pd.DataFrame)
        assert evaluator.comparison_dict == sample_comparison_dict
        assert all(col in evaluator.df.columns for col in evaluator.match_cols)
        assert evaluator.rowwise_distribution is not None
        assert evaluator.correctness_summary_df is not None
        assert evaluator.confusion_matrices.keys() == sample_comparison_dict.keys()
        assert evaluator.per_value_metrics.keys() == sample_comparison_dict.keys()


    def test_init_invalid_params(self,
                                 sample_df,
                                 sample_comparison_dict,
                                 sample_accepted_values):
        # No params at all
        with pytest.raises(TypeError):
            CMEEvaluator()
            
        with pytest.raises(ValueError):
            CMEEvaluator(
                comparison_dict=None,
                df=None,
                accepted_values=sample_accepted_values,
                id_col="id_col",
                bucket_name=None,
                folder=None,
                object_key=None,
                list_saved=False
            )

        # Incomplete S3 info: only object_key
        with pytest.raises(ValueError):
            CMEEvaluator(
                object_key='some_key.csv',
                comparison_dict=sample_comparison_dict,
                accepted_values=sample_accepted_values,
                id_col="id_col",
            )

        # df + list_saved=True is invalid
        with pytest.raises(ValueError):
            CMEEvaluator(
                df=sample_df,
                list_saved=True,
                comparison_dict=sample_comparison_dict,
                accepted_values=sample_accepted_values,
                id_col="id_col",
            )

        # All params given – too many options provided to the class.
        with pytest.raises(ValueError):
            CMEEvaluator(
                df=sample_df,
                bucket_name='bucket',
                folder='folder',
                accepted_values=sample_accepted_values,
                id_col="id_col",
                object_key='some_key.csv',
                list_saved=True,
                comparison_dict=sample_comparison_dict
            )

    def test_compute_rowwise_distribution_numeric_correctness(self,
                                                              evaluator_numeric_correctness):
        result = evaluator_numeric_correctness._compute_rowwise_distribution(df=evaluator_numeric_correctness.df)
    
        expected = [
            {'_num_correct': 0, 'count': 2, 'percent': 25.0, 'label': 'None correct'},
            {'_num_correct': 1, 'count': 3, 'percent': 37.5, 'label': '1 correct'},
            {'_num_correct': 2, 'count': 3, 'percent': 37.5, 'label': 'All correct'},
        ]
    
        result_sorted = sorted(result, key=lambda x: x['_num_correct'])
        expected_sorted = sorted(expected, key=lambda x: x['_num_correct'])
    
        assert result_sorted == expected_sorted

    
    def test_compute_correctness_summary_numeric_correctness(self,
                                                             evaluator_numeric_correctness):
        summary_df = evaluator_numeric_correctness._compute_correctness_summary_df()
    
        expected = [
            {
                "comparison_index": 1,
                "original_col": "field1",
                "compare_col": "field1_pred",
                "match_col": "field1_V_field1_pred",
                "correct_percent": 50.0,
            },
            {
                "comparison_index": 2,
                "original_col": "field2",
                "compare_col": "field2_pred",
                "match_col": "field2_V_field2_pred",
                "correct_percent": 62.5,
            }
        ]
    
        for expected_row in expected:
            actual_row = summary_df.loc[
                summary_df["match_col"] == expected_row["match_col"]
            ].iloc[0]
    
            assert round(actual_row["correct_percent"], 2) == expected_row["correct_percent"]
            assert actual_row["original_col"] == expected_row["original_col"]
            assert actual_row["compare_col"] == expected_row["compare_col"]
            assert actual_row["comparison_index"] == expected_row["comparison_index"]

    
    def test_compute_confusion_matrix_data_numeric_correctness(self,
                                                               evaluator_numeric_correctness):
        cm_df = evaluator_numeric_correctness._compute_confusion_matrix_data("field1")
    
        labels = ["A", "B", "C", "true_na"]
    
        expected_cm_df = pd.DataFrame(
            data=[
                [1, 1, 0, 0],  # true = A
                [0, 1, 0, 1],  # true = B
                [1, 0, 1, 0],  # true = C
                [0, 1, 0, 1],  # true = "True NA"
            ],
            index=labels,
            columns=labels
        ).astype(int)
    
        expected_cm_df.index.name = 'y_true'
        expected_cm_df.columns.name = 'y_pred'
    
        # Reindex both DataFrames to ensure exact matching order and labels
        cm_df = cm_df.reindex(index=labels,
                              columns=labels)
        
        expected_cm_df = expected_cm_df.reindex(index=labels,
                                                columns=labels)
    
        pd.testing.assert_frame_equal(cm_df, expected_cm_df)

    def test_confusion_matrix_handles_validation_failed(self):
        df = pd.DataFrame([
            {"status": "validation_failed", "field1": "A", "field1_pred": None},
            {"status": "valid", "field1": "B", "field1_pred": "B"},
        ])
        comparison_dict = {
            "field1": "field1_pred"
        }

        accepted_values = {
        "field1_pred": ["A", "B", "C"],
        }
        
        evaluator = CMEEvaluator(
            df=df,
            comparison_dict=comparison_dict,
            accepted_values=accepted_values,
            id_col="id_col"
        )
    
        cm_df = evaluator._compute_confusion_matrix_data("field1")
    
        labels = sorted(set(cm_df.index) | set(cm_df.columns))
        cm_df = cm_df.reindex(index=labels, columns=labels, fill_value=0)
    
        expected = pd.DataFrame(
            data=[
                [0, 0, 0, 1],  # A predicted as validation_failed
                [0, 1, 0, 0],  # B predicted as B
                [0, 0, 0, 0],  # No true na values
                [0, 0, 0, 0],  # No validation failed compared.
            ],
            index=["A", "B", "true_na", validation_failed_message],
            columns=["A", "B", "true_na", validation_failed_message]
        ).astype(int)

        expected.index.name = 'y_true'
        expected.columns.name = 'y_pred'
    
        expected = expected.reindex(index=labels, columns=labels, fill_value=0)
    
        pd.testing.assert_frame_equal(cm_df, expected)

    def test_confusion_matrix_handles_non_accepted_values(self):
        df = pd.DataFrame([
            {"status": "valid", "field1": "A", "field1_pred": "Z"},   # Z not accepted
            {"status": "valid", "field1": "B", "field1_pred": "B"},   # accepted
            {"status": "valid", "field1": None, "field1_pred": "X"},  # X is non accepted
        ])
        comparison_dict = {
            "field1": "field1_pred"
        }
        accepted_values = {
        "field1_pred": ["A", "B", "C"],
        }
        evaluator = CMEEvaluator(
            df=df,
            comparison_dict=comparison_dict,
            accepted_values=accepted_values,
            id_col="id_col"
        )
    
        cm_df = evaluator._compute_confusion_matrix_data("field1")
    
        labels = sorted(set(cm_df.index) | set(cm_df.columns))
        cm_df = cm_df.reindex(index=labels, columns=labels, fill_value=0)
    
        expected = pd.DataFrame(
            data=[
                [0, 0, 0, 1],  # A predicted as non-accepted values
                [0, 1, 0, 0],  # B predicted as B
                [0, 0, 0, 1],  # true na predicted as non-accepted values
                [0, 0, 0, 0],  # non-accepted values row (no true values with this label)
            ],
            index=["A", "B", "true_na", "non_accepted"],
            columns=["A", "B", "true_na", "non_accepted"]
        ).astype(int)

        expected.index.name = 'y_true'
        expected.columns.name = 'y_pred'
        expected = expected.reindex(index=labels, columns=labels, fill_value=0)

        print(cm_df, expected)
    
        pd.testing.assert_frame_equal(cm_df, expected)

    def test_compute_per_value_metrics_numeric_correctness(self, evaluator_numeric_correctness):
    
        results = evaluator_numeric_correctness._compute_per_value_metrics("field1")
        metrics = {r["value"]: r for r in results}
    
        expected_labels = set(["A", "B", "C", "true_na"])
        assert set(metrics.keys()) == expected_labels
    
        # Check "A"
        assert metrics["A"]["precision"] == 50.0
        assert metrics["A"]["recall"] == 50.0
    
        # Check "B"
        assert abs(metrics["B"]["precision"] - 33.33) < 0.1
        assert metrics["B"]["recall"] == 50.0
    
        # Check "C"
        assert metrics["C"]["precision"] == 100.0
        assert metrics["C"]["recall"] == 50.0
    
        # Check "TRUE NA"
        assert metrics["true_na"]["precision"] == 50.0
        assert metrics["true_na"]["recall"] == 50.0

    def test_get_status_summary(self, evaluator_statuses):
        summary = evaluator_statuses.get_status_summary("status")
        assert set(summary.index) == {valid_status, validation_failed_message, none_valid_status, partial_valid_status}
        assert summary.loc[valid_status].values[0] == 2
        assert summary.loc[validation_failed_message].values[0] == 1
        assert summary.loc[partial_valid_status].values[0] == 1
        assert summary.loc[none_valid_status].values[0] == 1


    def test_get_validation_failed(self, evaluator_statuses):
        vf_df = evaluator_statuses.get_validation_failed("status")
        assert all(vf_df["status"] == validation_failed_message)
        assert len(vf_df) == 1
        assert vf_df.iloc[0]["id_col"] == 1
    
    
    def test_get_invalid(self, evaluator_statuses):
        inv_df = evaluator_statuses.get_invalid("status")
        assert all(inv_df["status"] == none_valid_status)
        assert len(inv_df) == 1
        assert inv_df.iloc[0]["id_col"] == 3
    
    
    def test_get_partial(self, evaluator_statuses):
        part_df = evaluator_statuses.get_partial("status")
        assert all(part_df["status"] == partial_valid_status)
        assert len(part_df) == 1
        assert part_df.iloc[0]["id_col"] == 4
    
    
    def test_print_text_outputs_correct_value(self, evaluator_statuses, capsys):
        evaluator_statuses.print_text(id_val=1, text_col="field1")
        captured = capsys.readouterr()
        assert captured.out.strip() == "A"
    
    
    def test_get_non_accepted_summary_filters_correctly(self, evaluator_statuses):
        summary_df = evaluator_statuses.get_non_accepted_summary("field1")
        extracted_values = summary_df["extracted_value"].unique().tolist()
        assert "X" in extracted_values
        assert "Z" in extracted_values
        assert all(val not in evaluator_statuses.accepted_values["field1_pred"] for val in extracted_values)
    
    
    def test_get_non_accepted_summary_all_combines_results(self, evaluator_statuses):
        all_df = evaluator_statuses.get_non_accepted_summary_all()
        assert "actual_col" in all_df.columns
        assert set(all_df["actual_col"]) == {"field1"}
        assert all_df["count"].sum() > 0






