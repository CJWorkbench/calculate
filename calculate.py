from pandas.api.types import is_numeric_dtype

op_functions = {
    'Add' : 'sum',
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

op_names = {
    'Add' : 'Sum',
    'Subtract' : 'result',
    'Multiply' : 'Product',
    'Divide' : 'result',
    'Average' : 'Average',
    'Median' : 'Median',
    'Minimum' : 'Minimum',
    "Maximum" : 'Minimum',
    'Percentage change' : 'percentage_change',
    'X percent of Y' : 'result',
    'X is what percent of Y' : 'percentage'
    }

# Operations which take an arbitrary number of columns
multicolumn_ops = ['Add','Multiply','Average','Median','Minimum','Maximum']

def render(table, params):
    operation_strings = 'Add|Subtract|Multiply|Divide||Average|Median|Minimum|Maximum||Percentage change|X percent of Y|X is what percent of Y'.split('|')
    operation = operation_strings[params['operation']]

    if operation is '':
        return table  # waiting for paramter, do nothing

    if operation in multicolumn_ops:
        # multiple column operations (add, average...)

        colnames  = params['colnames']
        if colnames == '':
            return table  # waiting for paramter, do nothing
        colnames = colnames.split(',')

        for name in colnames:
            if not is_numeric_dtype(table[name]):
                return "Column " + name + " is not numbers"

        table[op_names[operation]]=(table[colnames]).agg(op_functions[operation], axis=1)

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

        table[op_names[operation]] = op_functions[operation](table[col1], table[col2])
        
    return table

