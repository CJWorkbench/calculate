from dataclasses import dataclass
from typing import Callable, List
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class MulticolumnOp:
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


@dataclass
class BinaryOp:
    fn: Callable[[pd.Series, pd.Series], pd.Series]
    """Function to operate on two Series, returning a Series."""

    default_result_column_format: str
    """op.default_result_column_format.format('x', 'y') => 'x minus y'."""

    def default_result_column_name(self, col1: str, col2: str) -> str:
        """op.default_result_column_name('x', 'y') => 'Sum of x, y'."""
        return self.default_result_column_format.format(col1=col1, col2=col2)


Operations = {
    'add': MulticolumnOp('sum', 'Sum of {cols}'),
    'subtract': BinaryOp(lambda x, y: x - y, '{col1} minus {col2}'),
    'multiply': MulticolumnOp('product', 'Product of {cols}'),
    'divide': BinaryOp(lambda x, y: x / y, '{col1} divided by {col2}'),
    'mean': MulticolumnOp('mean', 'Average of {cols}'),
    'median': MulticolumnOp('median', 'Median of {cols}'),
    'minimum': MulticolumnOp('min', 'Minimum of {cols}'),
    'maximum': MulticolumnOp('max', 'Maximum of {cols}'),
    'percent_change': BinaryOp(lambda x, y: (y - x) / x,
                               'Percent change {col1} to {col2}'),
    'percent_multiply': BinaryOp(lambda x, y: x * y,
                                 '{col1} percent of {col2}'),
    'percent_divide': BinaryOp(lambda x, y: x / y,
                               '{col1} is this percent of {col2}'),
}


def get_single_value(table, params):
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
            return "Row number cannot be greater than " + str(table.shape[0])
        return float(table[col][row])
    else:
        return params['single_value_constant']


def render(table, params):
    if not params['operation']:
        return table  # waiting for paramter, do nothing

    operation = Operations[params['operation']]

    if isinstance(operation, MulticolumnOp):
        # multiple column operations (add, average...)

        extra_scalar = (
            operation.agg in {'sum', 'product'}
            and params['single_value_selector'] != 'none'
        )

        colnames = params['colnames']
        if colnames == '':
            return table  # waiting for paramter, do nothing
        colnames = colnames.split(',')
        if len(colnames) == 1 and not extra_scalar:
            # need at least two columns to operate, unless we are adding
            # another value
            return table

        for name in colnames:
            if not is_numeric_dtype(table[name]):
                return "Column " + name + " is not numbers"

        if params['outcolname']:
            newcolname = params['outcolname']
        else:
            newcolname = operation.default_result_column_name(colnames)
        table[newcolname] = table[colnames].agg(operation.agg, axis=1)

        # Optional add/multiply all rows by a scalar
        if extra_scalar:
            val = get_single_value(table, params)
            if isinstance(val, str):
                return val  # error essage
            if operation.agg == 'sum':
                table[newcolname] += val
            else:
                table[newcolname] *= val
    else:
        # two column operations (subtract, percentage, ...)
        col1 = params['col1']
        col2 = params['col2']

        if col1 == '' or col2 == '':
            return table  # waiting for parameter, do nothing

        # If either column is not a number, return an error message
        # see https://github.com/CJWorkbench/cjworkbench/wiki/Column-Types
        if not is_numeric_dtype(table[col1]):
            return "Column " + col1 + " is not numbers"
        if not is_numeric_dtype(table[col2]):
            return "Column " + col2 + " is not numbers"

        if params['outcolname']:
            newcolname = params['outcolname']
        else:
            newcolname = operation.default_result_column_name(col1, col2)
        table[newcolname] = operation.fn(table[col1], table[col2])
    return table
