from pandas.api.types import is_numeric_dtype

op_functions = {
    'Sum' : 'sum',
    'Subtract' : lambda x,y: x-y,
    'Multiply' : 'product',
    'Divide' : lambda x,y: x/y,
    'Average' : 'mean',
    'Median' : 'median',
    'Minimum' : 'min',
    'Maximum' : 'max',
    'Percentage change' : lambda x,y: (y-x)/x,
    'X percent of Y' : lambda x,y: x*y,
    'X is what percent of Y' : lambda x,y: x/y
    }

op_result_names = {
    'Sum' : 'Sum of',
    'Subtract' : '{col1} minus {col2}',
    'Multiply' : 'Product of',
    'Divide' : '{col1} divided by {col2}',
    'Average' : 'Average of',
    'Median' : 'Median of',
    'Minimum' : 'Minimum of',
    "Maximum" : 'Minimum of',
    'Percentage change' : 'Percent change {col1} to {col2}',
    'X percent of Y' : '{col1} percent of {col2}',
    'X is what percent of Y' : '{col1} is this percent of {col2}'
    }

# Operations which take an arbitrary number of columns
multicolumn_ops = ['Sum','Multiply','Average','Median','Minimum','Maximum']

# Formatters to produce result column names
def format_two_cols(fstring, col1, col2):
    return fstring.format(col1=col1, col2=col2)

def format_multicols(prefix_string, cols):
    if len(cols) < 4:
        return prefix_string + ' ' + ', '.join(cols)
    else:
        return prefix_string + ' {num} columns'.format(num=len(cols))

# Return the single value the user is specifying, either a cell value or constant
def get_single_value(table, params):

    if params['single_value_selector'] == 1: # 'Cell value'
        col = params['single_value_col']
        row = int(params['single_value_row'])-1 # go from 1-based in the UI to 0 based in the table
        if row<0:
            return "Row number cannot be less than 1"
        elif row >= table.shape[0]:
            return "Row number cannot be greater than " + str(table.shape[0])
        return float(table[col][row])

    else:
        return float(params['single_value_constant'])


def render(table, params):
    operation_strings = 'Sum|Subtract|Multiply|Divide||Average|Median|Minimum|Maximum||Percentage change|X percent of Y|X is what percent of Y'.split('|')
    operation = operation_strings[params['operation']]

    if operation is '':
        return table  # waiting for paramter, do nothing

    if operation in multicolumn_ops:
        # multiple column operations (add, average...)

        extra_scalar = (operation=='Sum' or operation=='Multiply') and params['single_value_selector']!=0

        colnames  = params['colnames']
        if colnames == '':
            return table  # waiting for paramter, do nothing
        colnames = colnames.split(',')
        if len(colnames)==1 and not extra_scalar:
            return table  # need at least two columns to operate, unless we are adding another value

        for name in colnames:
            if not is_numeric_dtype(table[name]):
                return "Column " + name + " is not numbers"

        newcolname = format_multicols(op_result_names[operation], colnames)
        table[newcolname]=(table[colnames]).agg(op_functions[operation], axis=1)

        # Optional add/mulitply all rows by a scalar
        if extra_scalar:
            val = get_single_value(table, params)
            if isinstance(val, str):
                return str # error essage
            if operation=='Sum':
                table[newcolname] += val
            else:
                table[newcolname] *= val


    else:
        # two column operations (subtract, percentage, ...)
        col1  = params['col1']
        col2  = params['col2']

        if col1=='' or col2=='':   
            return table # waiting for paramter, do nothing

        # If either column is not a number, return an error message
        # see https://github.com/CJWorkbench/cjworkbench/wiki/Column-Types
        if not is_numeric_dtype(table[col1]):
            return "Column " + col1 + " is not numbers"
        if not is_numeric_dtype(table[col2]):
            return "Column " + col2 + " is not numbers"

        newcolname = format_two_cols(op_result_names[operation], col1, col2)
        table[newcolname] = op_functions[operation](table[col1], table[col2])
        
    return table

