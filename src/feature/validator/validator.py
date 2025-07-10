import pandas as pd
from typing import List, Union
from pandera import DataFrameSchema
from pandera.errors import SchemaErrors
import logging

class Validator:
    """
    Validator that accepts one or multiple pandera schemas at initialization.
    The validate method applies them sequentially to the provided DataFrame.
    """

    def __init__(self, schemas: Union[DataFrameSchema, List[DataFrameSchema]]):
        # Always store schemas as a list
        self.schemas: List[DataFrameSchema] = (
            schemas if isinstance(schemas, list) else [schemas]
        )

    def validate(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Applies all provided schemas sequentially to the DataFrame.
        Raises ValueError if any schema fails, with detailed error information.
        Returns the validated DataFrame after the last schema.
        """
        current = df
        for idx, schema in enumerate(self.schemas):
            try:
                current = schema.validate(current, lazy=True)
                logging.info(f"[SUCCESS] {dataset_name} - schema #{idx} passed.")
            except SchemaErrors as e:
                # Collect all validation errors
                errs = e.failure_cases.to_dict(orient="records")
                msg = (
                    f"[FAILURE] {dataset_name} - schema #{idx} failed. "
                    f"Errors: {errs}"
                )
                logging.error(msg)
                raise ValueError(msg)
        return current
