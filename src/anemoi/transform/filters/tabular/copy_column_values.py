# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols

VALID_OPERATIONS = ("overwrite_target_where_present", "overwrite_target")


@filter_registry.register("copy_column_values")
class CopyColumnValues(Filter):
    """Copy values from one DataFrame column into another.

    The configuration should contain ``source`` and ``target`` column names,
    and optionally an ``operation`` key (default ``overwrite_target_where_present``)
    selecting one of:

    - ``overwrite_target_where_present``: copies only non-missing source values
      into the target column. Existing target values are kept wherever the
      source is missing.
    - ``overwrite_target``: always sets target equal to source, including
      missing values.

    The optional ``allow_missing_columns`` key (default ``False``) controls
    behaviour when ``source`` and/or ``target`` are not present in the
    DataFrame:

    - ``False``: raises if either ``source`` or ``target`` is missing.
    - ``True``:

      - if ``source`` is missing, this filter is a no-op.
      - if ``target`` is missing (but ``source`` is present), ``target`` is
        created as a copy of ``source``.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - copy_column_values:
              source: roadside_wind_direction
              target: wind_direction
              operation: overwrite_target_where_present
              allow_missing_columns: false

    """

    def __init__(
        self,
        *,
        source: str,
        target: str,
        operation: str = "overwrite_target_where_present",
        allow_missing_columns: bool = False,
    ):
        if operation not in VALID_OPERATIONS:
            raise ValueError(f"Invalid operation: {operation!r}. Must be one of {VALID_OPERATIONS}.")
        self.source = source
        self.target = target
        self.operation = operation
        self.allow_missing_columns = allow_missing_columns

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = df.copy()

        if not self.allow_missing_columns:
            raise_if_df_missing_cols(df, [self.source, self.target])
        elif self.source not in df.columns:
            return df

        if self.target not in df.columns:
            df[self.target] = df[self.source]
            return df

        if self.operation == "overwrite_target":
            df[self.target] = df[self.source]
        else:  # overwrite_target_where_present
            df[self.target] = df[self.source].combine_first(df[self.target])

        return df
