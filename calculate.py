from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd


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
            colnames_str = ', '.join(colnames)
        else:
            colnames_str = f'{len(colnames)} columns'

        return self.default_result_column_format.format(cols=colnames_str)

    def _get_single_value(self, table, params):
        """
        Find the single value the user specified (cell value or constant).
        """
        if params['single_value_selector'] == 'cell':  # 'Cell value'
            col = params['single_value_col']
            # go from 1-based in the UI to 0 based in the table
            row = params['single_value_row'] - 1
            if row < 0:
                return "Row number cannot be less than 1"
            elif row >= table.shape[0]:
                return "Row number cannot be greater than %d" % table.shape[0]
            value = table[col][row]
            if pd.isnull(value):
                return "The chosen cell does not contain a number"
            try:
                return float(value)
            except ValueError:
                return "The chosen cell does not contain a number"
        else:
            return params['single_value_constant']

    def render(self, table, params, input_columns) -> Dict[str, Any]:
        extra_scalar = (
            self.agg in {'sum', 'product'}
            and params['single_value_selector'] != 'none'
        )

        if not params['colnames']:
            return table  # waiting for paramter, do nothing

        columns = [input_columns[c]
                   for c in params['colnames'].split(',') if c]
        if len(columns) == 1 and not extra_scalar:
            # need at least two columns to operate, unless we are adding
            # another value
            return table

        colnames = [column.name for column in columns]

        if params['outcolname']:
            newcolname = params['outcolname']
        else:
            newcolname = self.default_result_column_name(colnames)
        table[newcolname] = table[colnames].agg(self.agg, axis=1)

        # Optional add/multiply all rows by a scalar
        if extra_scalar:
            val = self._get_single_value(table, params)
            if isinstance(val, str):
                return val  # error essage
            if self.agg == 'sum':
                table[newcolname] += val
            else:
                table[newcolname] *= val

        return {
            'dataframe': table,
            'column_formats': {newcolname: columns[0].format},
        }


@dataclass
class BinaryOp:
    fn: Callable[[pd.Series, pd.Series], pd.Series]
    """Function to operate on two Series, returning a Series."""

    default_result_column_format: str
    """op.default_result_column_format.format('x', 'y') => 'x minus y'."""

    override_result_column_format: Optional[str] = None
    """Python format string to force, if needed (e.g., '{:,.1%}')."""

    def default_result_column_name(self, col1: str, col2: str) -> str:
        """op.default_result_column_name('x', 'y') => 'Sum of x, y'."""
        return self.default_result_column_format.format(col1=col1, col2=col2)

    def render(self, table, params, input_columns) -> Dict[str, Any]:
        if not params['col1'] or not params['col2']:
            return table  # waiting for parameter -- no-op

        col1 = input_columns[params['col1']]
        col2 = input_columns[params['col2']]

        if params['outcolname']:
            newcolname = params['outcolname']
        else:
            newcolname = self.default_result_column_name(col1.name, col2.name)
        table[newcolname] = self.fn(table[col1.name], table[col2.name])

        return {
            'dataframe': table,
            'column_formats': {
                newcolname: self.override_result_column_format or col1.format,
            },
        }


PercentFormat = '{:,.1%}'
Operations = {
    'add': MulticolumnOp('sum', 'Sum of {cols}'),
    'subtract': BinaryOp(lambda x, y: x - y, '{col1} minus {col2}'),
    'multiply': MulticolumnOp('product', 'Product of {cols}'),
    'divide': BinaryOp(
        lambda x, y: (x / y).replace([np.inf, -np.inf], np.nan),
        '{col1} divided by {col2}'
    ),
    'mean': MulticolumnOp('mean', 'Average of {cols}'),
    'median': MulticolumnOp('median', 'Median of {cols}'),
    'minimum': MulticolumnOp('min', 'Minimum of {cols}'),
    'maximum': MulticolumnOp('max', 'Maximum of {cols}'),
    'percent_change': BinaryOp(
        lambda x, y: ((y - x) / x).replace([np.inf, -np.inf], np.nan),
        'Percent change {col1} to {col2}',
        PercentFormat
    ),
    'percent_multiply': BinaryOp(lambda x, y: x * y,
                                 '{col1} percent of {col2}',
                                 PercentFormat),
    'percent_divide': BinaryOp(
        lambda x, y: (x / y).replace([np.inf, -np.inf], np.nan),
        '{col1} is this percent of {col2}',
        PercentFormat
    ),
}


def render(table, params, *, input_columns):
    if not params['operation']:
        return table  # waiting for paramter, do nothing

    operation = Operations[params['operation']]
    return operation.render(table, params, input_columns)
