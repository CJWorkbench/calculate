from pandas.api.types import is_numeric_dtype

def render(table, params):
    operation_strings = 'Add|Subtract|Multiply|Divide||Average|Median|Minimum|Maximum||Percentage change|X percent of Y|X is what percent of Y'.split('|')
    operation = operation_strings[params['operation']]

    if operation is '':
        return table  # waiting for paramter, do nothing

    if operation in ['Add','Multiply','Average','Median','Minimum','Maximum']:
        # multiple column operations (add, average...)

        colnames  = params['colnames']
        if colnames == '':
            return table  # waiting for paramter, do nothing
        colnames = colnames.split(',')

        op_functions = {
            'Add' : 'sum',
            'Multiply' : 'product',
            'Average' : 'mean',
            'Median' : 'median',
            'Minimum' : 'min',
            "Maximum" : 'max'
            }

        op_names = {
            'Add' : 'Sum',
            'Multiply' : 'Product',
            'Average' : 'Average',
            'Median' : 'Median',
            'Minimum' : 'Minimum',
            "Maximum" : 'Minimum'
            }

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

        if operation=='Subtract':
            table['result'] = table[col1]-table[col2]
        elif operation=='Divide':
            table['result'] = table[col1]/table[col2]
        elif operation=='Percentage change':
            table['percentage_change'] = (table[col2]-table[col1])/table[col1]
        elif operation=='X percent of Y':
            table['result'] = table[col1]*table[col2]
        elif operation=='X is what percent of Y':
            table['percentage'] = table[col1]/table[col2]
        

    return table

