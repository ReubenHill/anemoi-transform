# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
from collections.abc import Callable

import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry


@filter_registry.register("apply_custom_dataframe_function")
class ApplyCustomDataframeFunction(Filter):
    """Apply an arbitrary function to the whole DataFrame.

    This filter allows you to apply an arbitrary Python function (imported from
    a module) to the DataFrame. This enables advanced and flexible
    transformations that aren't covered by built-in filters. The function must
    accept a ``pandas.DataFrame`` as its first argument and return a
    ``pandas.DataFrame``.

    Notes
    -----

    This general purpose filter allows users to quickly prototype some data transformation.
    For an operational usage it is recommended to develop dedicated filters,
    that can be contributed to the project or developed
    as :ref:`plugins <anemoi-plugins:index-page>`.

    Examples
    --------

    .. code-block:: yaml

      input:
        pipe:
            - source:
              # mars, odb, csv, etc.
              # source attributes here
              # ...

            - apply_custom_dataframe_function:
                function: "package.module.function"
                args: [1013.25]
                kwargs:
                  column: "pressure"

    """

    def __init__(self, *, function: str, args: list | None = None, kwargs: dict | None = None):
        if not isinstance(function, str):
            raise ValueError(f"Expected 'function' to be a string. Got {function} instead.")

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        if not isinstance(args, list):
            raise ValueError(f"Expected 'args' to be a list. Got {args} instead.")
        if not isinstance(kwargs, dict):
            raise ValueError(f"Expected 'kwargs' to be a dictionary. Got {kwargs} instead.")

        self.function = self._import_function(function)
        self.args = args
        self.kwargs = kwargs

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.function(df, *self.args, **self.kwargs)

    @staticmethod
    def _import_function(function: str) -> Callable[..., pd.DataFrame]:
        """Import a function from a string path.

        Parameters
        ----------
        function : str
            The string path to the function, such as "package.module.function".

        Returns
        -------
        Callable[..., pd.DataFrame]
            The imported function.

        Raises
        ------
        ValueError
            If the function cannot be imported.
        """
        try:
            module_name, function_name = function.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, function_name)
        except Exception as e:
            raise ValueError(f"Could not import function {function}") from e
