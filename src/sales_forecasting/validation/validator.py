import pandas as pd
from typing import List, Union
from pandera import DataFrameSchema
from pandera.errors import SchemaErrors
from sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)

class Validator:
    def __init__(self):
        pass

    def validate(
        self,
        df: pd.DataFrame,
        schema: Union[DataFrameSchema, List[DataFrameSchema]],
        dataset_name: str = "dataset"
    ) -> pd.DataFrame:
        schemas = schema if isinstance(schema, list) else [schema]
        current = df
        for idx, sch in enumerate(schemas):
            try:
                current = sch.validate(current, lazy=True)
                logger.info(f"[SUCCESS] {dataset_name} - schema #{idx} passed.")
            except SchemaErrors as e:
                errs = e.failure_cases.to_dict(orient="records")
                msg = (
                    f"[FAILURE] {dataset_name} - schema #{idx} failed. "
                    f"Errors: {errs}"
                )
                logger.error(msg)
                raise ValueError(msg)
        return current
