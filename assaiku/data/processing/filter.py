import pandas as pd

AGE_COL = "age"


def filter_on_age(
    data: pd.DataFrame, age_col: str = "age", age_limit: int = 16
) -> pd.DataFrame:
    return data[data[age_col] >= age_limit]
