import pytest
import numpy as np
import pandas as pd

from src.post_processor import PostProcessor

@pytest.fixture
def standard_post_processor():
    df = pd.DataFrame([
            {"id": "1", "report": "test one","er_status": "a","er_score": "1+2","pr_status": "b","pr_score": "2+2","her2_status": "c","status":"partial",},
            {"id": "2", "report": "test two","er_status": "c","er_score": "4","pr_status": "b","pr_score": np.nan,"her2_status": "a","status":"valid",},
        ])
    cols_of_interest = ["er_status","er_score","pr_status","pr_score","her2_status"]
    processor = PostProcessor(df, cols_of_interest)

    yield processor

    del processor

@pytest.fixture
def mock_accepted_values(monkeypatch):
    monkeypatch.setattr(
        "config.pipeline_config.accepted_values",
        {
            "er_status": ['a','b','c',None],
            "er_score": ['2','3','4',None],
            "pr_status": ['a','b','c',None],
            "pr_score": ['2','3','4',None],
            "her2_status": ['a','b','c',None]
        }
    )
    yield

class TestPostProcessor:

    @pytest.mark.parametrize(
        "original_values, new_values",
        [({"er_status": "positive","er_score": "8","pr_status": "negative","pr_score": "2","her2_status": None},
          {"er_status": "positive","er_score": "8","pr_status": "negative","pr_score": "2","her2_status": None}),
         ({"er_status": "positive","er_score": "5+3","pr_status": "negative","pr_score": "2+2","her2_status": "unknown"},
          {"er_status": "positive","er_score": "8","pr_status": "negative","pr_score": "4","her2_status": "unknown"}),
         ({"er_status": "positive","er_score": "3 + 3","pr_status": "positive","pr_score": np.nan,"her2_status": "borderline"},
          {"er_status": "positive","er_score": "6","pr_status": "positive","pr_score": np.nan,"her2_status": "borderline"}),
         ({"er_status": "positive","er_score": "95+1","pr_status": "negative","pr_score": "pending","her2_status": "negative"},
          {"er_status": "positive","er_score": "95+1","pr_status": "negative","pr_score": "pending","her2_status": "negative"}),
        ]
    )
    def test_map_two_part_scores(self, standard_post_processor, original_values, new_values):
        cols_to_check = ["er_score","pr_score"]
        mapped_values = standard_post_processor.map_two_part_scores(original_values, cols_to_check)
        assert mapped_values == new_values


    @pytest.mark.parametrize(
        "original_values, new_values",
        [
            ({"er_status": "positive","er_score": "5/8","pr_status": "negative","pr_score": "2/8","her2_status": "borderline"},{"er_status": "positive","er_score": "5","pr_status": "negative","pr_score": "2","her2_status": "borderline"}),
            ({"er_status": "positive","er_score": "7","pr_status": "negative","pr_score": "4/8","her2_status": None}, {"er_status": "positive","er_score": "7","pr_status": "negative","pr_score": "4","her2_status": None}),
            ({"er_status": "positive","er_score": "1","pr_status": "negative","pr_score": "2","her2_status": None}, {"er_status": "positive","er_score": "1","pr_status": "negative","pr_score": "2","her2_status": None})
        ]
    )
    def test_map_score(self, standard_post_processor, original_values, new_values):

        output = standard_post_processor.map_score(original_values, ["er_score", "pr_score"])

        assert output == new_values

    @pytest.mark.parametrize(
        "original_values, new_values",
        [
            ({"er_status": "positive","er_score": "7","pr_status": "negative","pr_score": "4","her2_status": "3+"}, {"er_status": "positive","er_score": "7","pr_status": "negative","pr_score": "4","her2_status": "positive"}),
            ({"er_status": "positive","er_score": "5","pr_status": "negative","pr_score": "2","her2_status": "negative"},{"er_status": "positive","er_score": "5","pr_status": "negative","pr_score": "2","her2_status": "negative"}),
            ({"er_status": "positive","er_score": "1","pr_status": "negative","pr_score": "2","her2_status": None}, {"er_status": "positive","er_score": "1","pr_status": "negative","pr_score": "2","her2_status": None})
        ]
    )
    def test_apply_general_mapping(self, standard_post_processor, original_values, new_values):
        mapping = {
            "0": "negative",
            "1+": "negative",
            "2+": "borderline",
            "3+": "positive"}

        output = standard_post_processor.apply_general_mapping(original_values, mapping, ["her2_status"])     

        assert output == new_values

    def test_mappable_values_are_mapped_corectly(self,standard_post_processor):
        pairs = [("er_score","er_status"),("pr_score","pr_status")]
        row  ={"er_score":"0","er_status": "old_er","pr_score":"8","pr_status":"old_pr"}
    
    
        output = standard_post_processor.score_to_status(row,pairs=pairs)
    
        assert output["er_status"]== "negative"
        assert output["pr_status"]== "positive"
    
    def test_unmapped_values_unchanged(self,standard_post_processor):
        pairs = [("er_score","er_status"),("pr_score","pr_status")]
        row  ={"er_score":"5","er_status": "keep_er","pr_score":"6","pr_status":"keep_pr"}
    
        output = standard_post_processor.score_to_status(row,pairs=pairs)
    
        assert output["er_status"]== "keep_er"
        assert output["pr_status"]== "keep_pr"
    
    def test_missing_status_column_only(self,standard_post_processor):
        pairs = [("er_score","er_status"),("pr_score","pr_status")]
        row  ={"er_score":"0","er_status": None,"pr_score":"8"}
    
        output = standard_post_processor.score_to_status(row,pairs=pairs)
        assert output["er_status"] == "negative"
        assert set(output.keys())== set(row.keys())

    def test_process_row(self, standard_post_processor):
        data = {"id": "2", "report": "test two","er_status": "positive","er_score": "6","pr_status": "positive","pr_score": np.nan,"her2_status": "borderline", "status":"valid"}
        row = pd.Series(data)

        result = standard_post_processor.process_row(row)
        expected_cols = list(data.keys()) + [col + "_p" for col in standard_post_processor.cols_to_process] + ["status_processed"]
        assert isinstance(result, dict)
        assert all(col in result.keys() for col in expected_cols)

    def test_process_row_val_failed(self, standard_post_processor):
        data = {"id": "2", "report": "test two","er_status": np.nan,"er_score": np.nan,"pr_status": np.nan,"pr_score": np.nan,"her2_status": np.nan, "status":"validation_failed"}
        row = pd.Series(data)

        result = standard_post_processor.process_row(row)
        expected_cols = list(data.keys()) + ["status_processed"]
        assert isinstance(result, dict)
        assert all(col in result.keys() for col in expected_cols)
        assert result["status_processed"] == "validation_failed"

    def test_run(self, standard_post_processor, mock_accepted_values):
        new_df = standard_post_processor.run()
        expected_df = pd.DataFrame([
            {"id": "1", "report": "test one","er_status": "a","er_score": "1+2","pr_status": "b","pr_score": "2+2","her2_status": "c","status":"partial","er_status_p": "a","er_score_p": "3","pr_status_p": "b","pr_score_p": "4","her2_status_p": "c",  "status_processed":"valid"},
            {"id": "2", "report": "test two","er_status": "c","er_score": "4","pr_status": "b","pr_score": np.nan,"her2_status": "a","status":"valid","er_status_p": "c","er_score_p": "4","pr_status_p": "b","pr_score_p": None,"her2_status_p": "a","status_processed":"valid"},
        ])

        pd.testing.assert_frame_equal(new_df, expected_df)