from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional
from config.pipeline_config import status_values, score_values

status_values_lower = [v.lower() for v in status_values] + [None]
score_values_str = list(map(str,score_values)) + [None]

key_missing_str = "key_missing"
validation_failed_message = "validation_failed"
valid_status = "valid"
partial_valid_status = "partial"
none_valid_status = "invalid"

class ValidSchema(BaseModel):
    """
    The schema to validate a parsed JSON dictionary against.
    
    'Optional' means 'don't throw an error if this key is missing'
    The key value is populated with a placeholder if it was missing 
    and an alias added to refer to the key using an alternative name, 
    in this case keys with spaces in them
    """
    er_status: Optional[str] = Field(key_missing_str, alias = "er status")
    er_score: Optional[str] = Field(key_missing_str, alias = "er score")
    pr_status: Optional[str] = Field(key_missing_str, alias = "pr status")
    pr_score: Optional[str] = Field(key_missing_str, alias = "pr score")
    her2_status: Optional[str] = Field(key_missing_str, alias = "her2 status")

    model_config = {
        "validate_by_name" : True,
        "extra" : "ignore"  # Discards unknown keys (default)
    }
    # Make sure score fields are strings
    @field_validator('er_score', 'pr_score', mode='before')
    @classmethod
    def convert_to_str(cls, value: Any) -> Any:
        if value == None:
            return value
        else:
            return str(value)