from pandas.api.types import is_numeric_dtype

op_functions = {
    'add': 'sum',
    'subtract': lambda x, y: x - y,
    'multiply': 'product',
    'divide': lambda x, y: x / y,
    'mean': 'mean',
    'median': 'median',
    'minimum': 'min',
    'maximum': 'max',
    'percent_change': lambda x, y: (y - x) / x,
    'percent_multiply': lambda x, y: x * y,
    'percent_divide': lambda x, y: x / y,
}

op_result_names = {
    'add': 'Sum of {cols}',
    'subtract': '{col1} minus {col2}',
    'multiply': 'Product of {cols}',
    'divide': '{col1} divided by {col2}',
    'mean': 'Average of {cols}',
    'median': 'Median of {cols}',
    'minimum': 'Minimum of {cols}',
    'maximum': 'Maximum of {cols}',
    'percent_change': 'Percent change {col1} to {col2}',
    'percent_multiply': '{col1} percent of {col2}',
    'percent_divide': '{col1} is this percent of {col2}',
}

# Operations which take an arbitrary number of columns
multicolumn_ops = {'add', 'multiply', 'mean', 'median', 'minimum', 'maximum'}


# Formatters to produce result column names
def format_two_cols(fstring, col1, col2):
    return fstring.format(col1=col1, col2=col2)


def format_multicols(fstring, cols):
    if len(cols) < 4:
        cols_str = ', '.join(cols)
    else:
        cols_str = f'{len(cols)} columns'

    return fstring.format(cols=cols_str)


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
    operation = params['operation']

    if not operation:
        return table  # waiting for paramter, do nothing

    if operation in multicolumn_ops:
        # multiple column operations (add, average...)

        extra_scalar = (
            (operation == 'add' or operation == 'multiply')
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
            newcolname = format_multicols(op_result_names[operation], colnames)
        table[newcolname] = table[colnames].agg(op_functions[operation],
                                                axis=1)

        # Optional add/multiply all rows by a scalar
        if extra_scalar:
            val = get_single_value(table, params)
            if isinstance(val, str):
                return val  # error essage
            if operation == 'add':
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
            newcolname = format_two_cols(op_result_names[operation],
                                         col1, col2)
        table[newcolname] = op_functions[operation](table[col1],
                                                    table[col2])
    return table
