# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pandas as pd
import pytest

from anemoi.transform.filters import create_filter_by_name as create_filter


def test_overwrite_target_where_present_default():
    df = pd.DataFrame(
        {
            "source": [1.0, np.nan, 3.0, np.nan],
            "target": [10.0, 20.0, 30.0, np.nan],
        }
    )
    copy_column_values = create_filter("copy_column_values", source="source", target="target")
    result = copy_column_values(df.copy())

    assert isinstance(result, pd.DataFrame)
    # source overwrites target where source is non-missing; target kept where source is missing
    expected = pd.Series([1.0, 20.0, 3.0, np.nan], name="target")
    pd.testing.assert_series_equal(result["target"], expected)
    # source column untouched
    pd.testing.assert_series_equal(result["source"], df["source"])


def test_overwrite_target():
    df = pd.DataFrame(
        {
            "source": [1.0, np.nan, 3.0, np.nan],
            "target": [10.0, 20.0, 30.0, np.nan],
        }
    )
    copy_column_values = create_filter(
        "copy_column_values",
        source="source",
        target="target",
        operation="overwrite_target",
    )
    result = copy_column_values(df.copy())

    pd.testing.assert_series_equal(result["target"], df["source"], check_names=False)


def test_missing_source_raises_by_default():
    df = pd.DataFrame({"target": [1.0, 2.0]})
    copy_column_values = create_filter("copy_column_values", source="source", target="target")
    with pytest.raises(ValueError):
        _ = copy_column_values(df.copy())


def test_missing_target_raises_by_default():
    df = pd.DataFrame({"source": [1.0, 2.0]})
    copy_column_values = create_filter("copy_column_values", source="source", target="target")
    with pytest.raises(ValueError):
        _ = copy_column_values(df.copy())


def test_missing_source_allowed_is_noop():
    df = pd.DataFrame({"target": [1.0, 2.0]})
    copy_column_values = create_filter(
        "copy_column_values",
        source="source",
        target="target",
        allow_missing_columns=True,
    )
    result = copy_column_values(df.copy())

    pd.testing.assert_frame_equal(result, df)


@pytest.mark.parametrize("operation", ["overwrite_target_where_present", "overwrite_target"])
def test_missing_target_allowed_creates_target(operation):
    df = pd.DataFrame({"source": [1.0, np.nan, 3.0]})
    copy_column_values = create_filter(
        "copy_column_values",
        source="source",
        target="target",
        operation=operation,
        allow_missing_columns=True,
    )
    result = copy_column_values(df.copy())

    assert "target" in result.columns
    pd.testing.assert_series_equal(result["target"], df["source"], check_names=False)


def test_invalid_operation_raises():
    with pytest.raises(ValueError):
        _ = create_filter(
            "copy_column_values",
            source="source",
            target="target",
            operation="not_a_real_operation",
        )
