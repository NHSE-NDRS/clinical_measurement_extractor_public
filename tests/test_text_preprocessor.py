import pytest
from src.text_preprocessor import TextPreprocessor

class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    @pytest.mark.parametrize(
        "input_text, expected_output",
        [('This  is a test string with an extra space','This is a test string with an extra space'),# Double space in text
         ('   Leading spaces','Leading spaces'),# Leading spaces
         ('Trailing spaces   ','Trailing spaces'),# Trailing spaces
         ('  Mixed   bag  ','Mixed bag'),# Leading + trailing + internal
        ]
    )
    def test_remove_spaces(self, input_text, expected_output):
        result = self.preprocessor.extra_space_removal(input_text)
        assert result == expected_output
    
    def test_process(self):
        input_text = "  This is  a test string.    "
        expected_output = "This is a test string."
        result = self.preprocessor.process(input_text, 1)
        assert result == expected_output
