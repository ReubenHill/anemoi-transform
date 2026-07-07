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


@filter_registry.register("rename_tabular")
class Rename(Filter):
    """Rename one or more columns in the DataFrame.

    The configuration should be a dictionary with the key ``columns`` containing a
    dictionary which maps old column names to new column names.

    The optional ``allow_missing_columns`` key (default ``False``) can be used
    to skip missing source columns instead of raising an error.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - rename:
              columns:
                from_name: to_name
              allow_missing_columns: false

    """

    def __init__(self, *, columns: dict[str, str], allow_missing_columns: bool = False):
        self.columns = columns
        self.allow_missing_columns = allow_missing_columns

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
      if not self.allow_missing_columns:
        raise_if_df_missing_cols(obs_df, list(self.columns.keys()))
        columns_to_rename = self.columns
      else:
        columns_to_rename = {k: v for k, v in self.columns.items() if k in obs_df.columns}

      obs_df = obs_df.rename(columns=columns_to_rename)
      return obs_df
