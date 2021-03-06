from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from cjwmodule import i18n
from cjwmodule.util.colnames import gen_unique_clean_colnames_and_warn


@dataclass
class MulticolumnOp:
    """
    Multiple-column operations (add, average, ...).
    """

    agg: str
    """Argument to pass to pd.DataFrame.agg()."""

    default_result_column_format: str
    """op.default_result_column_format.format('x, y') => 'Sum of x, y'."""

    def default_result_column_name(self, colnames: List[str]) -> str:
        """op.default_result_column_name(['x', 'y']) => 'Sum of x, y'."""
        if len(colnames) < 4:
            colnames_str = ", ".join(colnames)
        else:
            colnames_str = f"{len(colnames)} columns"

        return self.default_result_column_format.format(cols=colnames_str)

    def _get_single_value(self, table, params):
        """
        Find the single value the user specified (cell value or constant).
        """
        if params["single_value_selector"] == "cell":  # 'Cell value'
            col = params["single_value_col"]
            # go from 1-based in the UI to 0 based in the table
            row = params["single_value_row"] - 1
            if row < 0:
                return i18n.trans(
                    "badParam.single_value_row.tooSmall",
                    "Row number cannot be less than 1",
                )
            elif row >= table.shape[0]:
                return i18n.trans(
                    "badParam.single_value_row.tooBig",
                    "Row number cannot be greater than {limit}",
                    {"limit": table.shape[0]},
                )
            if not col:
                return i18n.trans(
                    "badParam.single_value_col.missing",
                    "Please select the cell value's column",
                )
            value = table[col][row]
            _error_not_a_number = i18n.trans(
                "badParam.single_value_col.notANumber",
                "The chosen cell does not contain a number",
            )
            if pd.isnull(value):
                return _error_not_a_number
            try:
                return float(value)
            except ValueError:
                return _error_not_a_number
        else:
            return params["single_value_constant"]

    def render(self, table, params, input_columns) -> Dict[str, Any]:
        colnames = params["colnames"]
        if not colnames:
            return None, None  # waiting for parameter, do nothing

        columns = [input_columns[c] for c in colnames]
        extra_scalar = (
            self.agg in {"sum", "product"} and params["single_value_selector"] != "none"
        )
        if len(columns) == 1 and not extra_scalar:
            # need at least two columns to operate, unless we are adding
            # another value
            return None, None  # waiting for parameter, do nothing

        series = table[colnames].agg(self.agg, axis=1)
        series.name = self.default_result_column_name(colnames)

        # Optional add/multiply all rows by a scalar
        if extra_scalar:
            val = self._get_single_value(table, params)
            if isinstance(val, i18n.I18nMessage):
                return val, None  # error essage
            if self.agg == "sum":
                series += val
            else:
                series *= val

        return series, columns[0].format


@dataclass
class BinaryOp:
    fn: Callable
    """Function to operate on two Series, plus (optinally) their formats, returning a Series.
    Formats will be passed if this function takes 4 args, not usual 2"""

    default_result_column_name_format: str
    """op.default_result_column_format.format('x', 'y') => 'x minus y'."""

    override_result_column_format: Optional[Callable[[str, str], str]] = None
    """Return result column format, given input colunmn formats (e.g., '{:,.1%}')."""

    def default_result_column_name(self, col1: str, col2: str) -> str:
        """op.default_result_column_name('x', 'y') => 'Sum of x, y'."""
        return self.default_result_column_name_format.format(col1=col1, col2=col2)

    def render(self, table, params, input_columns) -> Dict[str, Any]:
        if not params["col1"] or not params["col2"]:
            return None, None  # waiting for parameter -- no-op

        col1 = input_columns[params["col1"]]
        col2 = input_columns[params["col2"]]

        if len(signature(self.fn).parameters) == 2:
            series = self.fn(table[col1.name], table[col2.name])
        else:
            series = self.fn(
                table[col1.name],
                table[col2.name],
                input_columns[col1.name].format,
                input_columns[col2.name].format,
            )
        series.name = self.default_result_column_name(col1.name, col2.name)

        if self.override_result_column_format:
            newcolformat = self.override_result_column_format(
                input_columns[col1.name].format, input_columns[col2.name].format
            )
        else:
            newcolformat = col1.format

        return series, newcolformat


@dataclass
class UnaryOp:
    """
    Single-column operations (Percent of column sum).
    """

    fn: Callable[[pd.Series], pd.Series]
    """Function to operate on column"""

    default_result_column_name_format: str
    """op.default_result_column_name_format.format('x', 'y') => 'x minus y'."""

    override_result_column_format: Optional[str] = None
    """Python format string to force, if needed (e.g., '{:,.1%}')."""

    def default_result_column_name(self, col1: str) -> str:
        """op.default_result_column_name('x') => 'Percent of x'."""
        return self.default_result_column_name_format.format(col=col1)

    def render(self, table, params, input_columns) -> Dict[str, Any]:
        if not params["col1"]:
            return None, None  # waiting for parameter -- no-op

        col1 = params["col1"]
        series = self.fn(table[col1])

        if isinstance(series, i18n.I18nMessage):
            return series, None  # error message

        series.name = self.default_result_column_name(col1)
        return series, (self.override_result_column_format or col1.format)


PercentFormat = "{:,.1%}"
PercentFormatCallable = lambda x_fmt, y_fmt: PercentFormat

Operations = {
    "add": MulticolumnOp("sum", "Sum of {cols}"),
    "subtract": BinaryOp(lambda x, y: x - y, "{col1} minus {col2}"),
    "multiply": MulticolumnOp("product", "Product of {cols}"),
    "divide": BinaryOp(
        lambda x, y: (x / y).replace([np.inf, -np.inf], np.nan),
        "{col1} divided by {col2}",
    ),
    "mean": MulticolumnOp("mean", "Average of {cols}"),
    "median": MulticolumnOp("median", "Median of {cols}"),
    "minimum": MulticolumnOp("min", "Minimum of {cols}"),
    "maximum": MulticolumnOp("max", "Maximum of {cols}"),
    "percent_change": BinaryOp(
        lambda x, y: ((y - x) / x).replace([np.inf, -np.inf], np.nan),
        "Percent change {col1} to {col2}",
        PercentFormatCallable,
    ),
    "percent_multiply": BinaryOp(
        lambda x, y, x_fmt, y_fmt: x * y if x_fmt == "{:,.1%}" else x * y / 100,
        "{col1} percent of {col2}",
        lambda x_fmt, y_fmt: y_fmt,
    ),
    "percent_divide": BinaryOp(
        lambda x, y: (x / y).replace([np.inf, -np.inf], np.nan),
        "{col1} is this percent of {col2}",
        PercentFormatCallable,
    ),
    "percent_of_column_sum": UnaryOp(
        (
            lambda x: (x / x.sum())
            if (all(x.isna()) or x.sum() != 0)
            else i18n.trans(
                "badData.percent_of_column_sum.sumIsZero", "Column sum is 0."
            )
        ),
        "Percent of {col}",
        PercentFormat,
    ),
}


def render(table, params, *, input_columns, settings):
    operation = Operations[params["operation"]]
    series_or_error, format = operation.render(table, params, input_columns)

    if series_or_error is None:
        return table  # Waiting for parameter -- no-op
    elif isinstance(series_or_error, pd.Series):
        if params["outcolname"]:
            colname = params["outcolname"]
            errors = []
        else:
            colnames, errors = gen_unique_clean_colnames_and_warn(
                [series_or_error.name],
                existing_names=list(input_columns.keys()),
                settings=settings,
            )
            colname = colnames[0]
        table[colname] = series_or_error
        return {
            "dataframe": table,
            "errors": errors,
            "column_formats": {colname: format},
        }
    else:
        return series_or_error


def _migrate_params_v0_to_v1(params):
    """
    v0: statictext had values (!); v1 no statictext values.

    This migration encompasses two time periods: the time before
    multiply_additional (a statictext) was added, and the time after.
    """
    return {
        "operation": params["operation"],
        "colnames": params["colnames"],
        "col1": params["col1"],
        "col2": params["col2"],
        "single_value_selector": params["single_value_selector"],
        "single_value_col": params["single_value_col"],
        "single_value_row": params["single_value_row"],
        "single_value_constant": params["single_value_constant"],
        "outcolname": params.get("outcolname", ""),  # it may not be there
    }


def _migrate_params_v1_to_v2(params):
    """v1: menus are numeric; v2: menus are text."""
    return {
        **params,
        "operation": {
            0: "add",
            1: "subtract",
            2: "multiply",
            3: "divide",
            # separator
            5: "mean",
            6: "median",
            7: "minimum",
            8: "maximum",
            # separator
            10: "percent_change",
            11: "percent_multiply",
            12: "percent_divide",
            # separator
            13: "percent_of_column_sum",
        }.get(params["operation"], "add"),
        "single_value_selector": {0: "none", 1: "cell", 2: "constant"}.get(
            params["single_value_selector"], "none"
        ),
    }


def _migrate_params_v2_to_v3(params):
    """v2: colnames is comma-separated str. v3: it's List[str]."""
    return {**params, "colnames": [c for c in params["colnames"].split(",") if c]}


def migrate_params(params):
    if "xtext" in params or "outcolname" not in params:
        params = _migrate_params_v0_to_v1(params)
    if isinstance(params["operation"], int):
        params = _migrate_params_v1_to_v2(params)
    if isinstance(params["colnames"], str):
        params = _migrate_params_v2_to_v3(params)
    return params
