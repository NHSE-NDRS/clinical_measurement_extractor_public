import logging

class TextPreprocessor:
    """
    Take in text and preprocess the text to be inserted into the prompt.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extra_space_removal(self,text: str) -> str:
        """
        Remove extra spaces from a string by converting multiple spaces into one,
        and trimming any leading or trailing whitespace.
    
        Args:
            text (str): Input string that contains extra spaces.
    
        Returns:
            str: A cleaned string with only single spaces and no leading/trailing whitespace.
        """
        text = text.strip()  # Remove leading and trailing spaces
        
        while "  " in text:
            text = text.replace("  ", " ")
        return text
    
    def process(self, text: str, doc_id:int) -> str:
        """
        Preprocesses Text for LLM extraction Pipeline.

        Args:
            text (str): Text to be proprocessed for entity extraction.
            doc_id (int): Document ID associated with the free-text data.
        Returns:
            text (str): Returns preprocessed text.
        """
        
        self.logger.info(f"Document ID: {doc_id} | Preprocessing text")
        text = self.extra_space_removal(text)
        
        return text