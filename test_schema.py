
import pandas as pd
import pytest

from src import schema


@pytest.mark.parametrize("filename", ["churn_v1.csv", "churn_v2.csv"])
def test_dataset_schema_is_valid(filename):
    df = pd.read_csv(filename)

    validated_df = schema.validate(df)

    assert list(validated_df[schema.FEATURES].columns) == schema.FEATURES
    assert len(validated_df) > 0
