# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pandas as pd
import pytest

from anemoi.transform.filters import create_filter_by_name as create_filter


# For testing one positional argument and one keyword argument
def add_offset(df: pd.DataFrame, offset: float, *, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column] + offset
    return df


# For testing two positional-or-keyword arguments
def scale_two_columns(df: pd.DataFrame, factor: float, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = df[column] * factor
    return df


# For testing no arguments
def identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def test_apply_custom_dataframe_function_with_args_and_kwargs():
    df = pd.DataFrame({"pressure": [1000.0, 2000.0, 3000.0]})
    apply_custom_dataframe_function = create_filter(
        "apply_custom_dataframe_function",
        function="tests.tabular_filters.test_apply_custom_dataframe_function.add_offset",
        args=[13.25],
        kwargs={"column": "pressure"},
    )
    result = apply_custom_dataframe_function(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result["pressure"].tolist() == [1013.25, 2013.25, 3013.25]


def test_apply_custom_dataframe_function_with_positional_args_only():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    apply_custom_dataframe_function = create_filter(
        "apply_custom_dataframe_function",
        function="tests.tabular_filters.test_apply_custom_dataframe_function.scale_two_columns",
        args=[2.0, ["a", "b"]],
    )
    result = apply_custom_dataframe_function(df.copy())

    assert result["a"].tolist() == [2.0, 4.0]
    assert result["b"].tolist() == [6.0, 8.0]


def test_apply_custom_dataframe_function_defaults():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    apply_custom_dataframe_function = create_filter(
        "apply_custom_dataframe_function",
        function="tests.tabular_filters.test_apply_custom_dataframe_function.identity",
    )
    result = apply_custom_dataframe_function(df.copy())

    assert result.equals(df)


def test_apply_custom_dataframe_function_invalid_function_type():
    with pytest.raises(ValueError):
        create_filter("apply_custom_dataframe_function", function=123)


def test_apply_custom_dataframe_function_invalid_args_type():
    with pytest.raises(ValueError):
        create_filter(
            "apply_custom_dataframe_function",
            function="tests.tabular_filters.test_apply_custom_dataframe_function.identity",
            args="not-a-list",
        )


def test_apply_custom_dataframe_function_invalid_kwargs_type():
    with pytest.raises(ValueError):
        create_filter(
            "apply_custom_dataframe_function",
            function="tests.tabular_filters.test_apply_custom_dataframe_function.identity",
            kwargs="not-a-dict",
        )


def test_apply_custom_dataframe_function_unimportable_function():
    with pytest.raises(ValueError):
        create_filter("apply_custom_dataframe_function", function="not.a.real.module.function")
