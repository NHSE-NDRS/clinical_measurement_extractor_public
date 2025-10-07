import pandas as pd
import numpy as np
import pytest
from src.evaluation_helpers import (
    add_complete_match,
    calculate_column_difference,
    add_columns
    )


class TestAddColumns:
    def test_addition(self):
        """Test addition of two numeric columns"""
        df = pd.DataFrame({'col1': [10,20,30],
                           'col2': [15,25,35]
                           })
        result = add_columns(df, 'col1','col2','total')
        expected = pd.Series( [25,45,65],name = 'total')
        pd.testing.assert_series_equal(result['total'],expected)
        assert 'total' in result.columns
        
    def test_float_columns(self):
        """Test addition with float columns"""
        df = pd.DataFrame({
            'col1': [10.5,20.35],
            'col2': [1.25,2.035]
            })
        result = add_columns(df,'col1','col2','total')
        assert result.loc[0,'total'] == 11.75
        assert result.loc[1,'total'] == 22.385

    def test_missing_column_error(self):
        """Test that KeyError is raised when columns are missing"""
        df = pd. DataFrame({'A':[1,2], 'B':[3,4]})
        with pytest.raises(KeyError):
            add_columns(df,'A','C','total')
            with pytest.raises(KeyError):
                add_columns(df,'D','B','total')

    def test_mixed_int_float_columns(self):
        """Test addition of int and float columns"""
        df = pd.DataFrame({
            'int_col': [1,2,3],
            'float_col' : [0.5,1.5,2.5]
        })
        result = add_columns(df,'int_col','float_col','total')
        expected = pd.Series([1.5,3.5,5.5],name = 'total')
        pd.testing.assert_series_equal(result['total'],expected)
    


class TestAddCompleteMatch:

    def test_identical_values(self):
        df = pd.DataFrame({'A': ['test'], 'B': ['test']})
        result = add_complete_match(df, 'A', 'B', 'match')
        assert result['match'].iloc[0] == True

    def test_different_values(self):
        df = pd.DataFrame({'A': ['test'], 'B': ['fail']})
        result = add_complete_match(df, 'A', 'B', 'match')
        assert result['match'].iloc[0] == False

    def test_nan_values_equal(self):
        df = pd.DataFrame({'A': [np.nan], 'B': [np.nan]})
        result = add_complete_match(df, 'A', 'B', 'match')
        assert result['match'].iloc[0] == True 

    def test_mixed_types_raises(self):
        df = pd.DataFrame({'A': ['string'], 'B': [123]})
        with pytest.raises(TypeError):
            add_complete_match(df, 'A', 'B', 'match')


class TestCalculateColumnDifference:

    def test_difference_correct(self):
        evaluation_test_df = pd.DataFrame({
            'column_1':[10, 20, 30],
            'column_2': [15, 5, 25]
        })
        result = calculate_column_difference(evaluation_test_df, 'column_1', 'column_2','difference')
        expected = pd.Series([-5, 15, 5],name = 'difference')
        pd.testing.assert_series_equal(result['difference'], expected)

    def test_difference_with_nans(self):
        evaluation_test_df = pd.DataFrame({
            'column_1': [np.nan, 20],
            'column_2': [15, np.nan]
        })
        result = calculate_column_difference(evaluation_test_df, 'column_1', 'column_2','difference')
        expected = pd.Series([np.nan, np.nan],name = 'difference')
        pd.testing.assert_series_equal(result['difference'], expected)

    def test_column_missing_raises(self):
        evaluation_test_df = pd.DataFrame({
            'column_1': [10],
            'column_2': [5]
        })
        with pytest.raises(KeyError):
            calculate_column_difference(evaluation_test_df, 'nonexistent_col', 'column_2','difference')

    def test_empty_dataframe(self):
        evaluation_test_df = pd.DataFrame(columns=['column_1', 'column_2'])
        result = calculate_column_difference(evaluation_test_df, 'column_1', 'column_2','difference')
        expected = pd.Series(name= 'difference')#name to match column name
        pd.testing.assert_series_equal(result['difference'], expected)


        


