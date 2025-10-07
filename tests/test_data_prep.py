import pytest
import pandas as pd
from src.data_prep import remove_gap, remove_placeholders

@pytest.mark.parametrize(
        "input_row, expected",
        [(
            pd.Series([1,'ORG','This text is fine'], index = ['id','org','text']),
            pd.Series([1,'ORG','This text is fine'], index = ['id','org','text'])
        ),
         (
             pd.Series([1,'ORG','This is the <GAP> <GAP> text that needs sorting out <GAP> <GAP> <GAP>'], index = ['id','org','text']),
             pd.Series([1,'ORG','This is the   text that needs sorting out   '], index = ['id','org','text'])
         )]
    )
def test_remove_gap(input_row, expected):
    output = remove_gap(input_row, 'text')
    pd.testing.assert_series_equal(output, expected)

@pytest.mark.parametrize(
        "input_text, by_list, expected_text",
        [('This text has no placeholder', True,'This text has no placeholder'),
        ('This text has multiple {EMAIL} placeholders {PHONE NUMBER} including some with a space in {DATE}', True,'This text has multiple  placeholders  including some with a space in '),
        ('This text has multiple {EMAIL} placeholders {PHONE NUMBER} including one not from the list {EXTRA}', True,'This text has multiple  placeholders  including one not from the list {EXTRA}'),
        ('This text has multiple {EMAIL} placeholders {PHONE NUMBER} including one not from the list {EXTRA}', False,'This text has multiple  placeholders  including one not from the list '),
        ]
    )
def test_remove_placeholders(input_text, by_list, expected_text):
    input_row = pd.Series([1,'ORG',input_text], index = ['id','org','text'])
    expected_row = pd.Series([1,'ORG',expected_text], index = ['id','org','text'])
    output = remove_placeholders(input_row, 'text', by_list = by_list)
    pd.testing.assert_series_equal(output, expected_row)