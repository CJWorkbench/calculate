import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from calculate import render

# turn menu strings into indices for parameter dictionary
# must be kept in sync with filter.json
menutext = "Sum|Subtract|Multiply|Divide||Average|Median|Minimum|Maximum||Percentage change|X percent of Y|X is what percent of Y"
menu = menutext.split('|')

class TestCalculate(unittest.TestCase):

    def setUp(self):
        # Test data includes some non number columns and some nulls and values to check type handling
        self.table = pd.DataFrame(
            [['fred', 2, 3, 4.5, 1],
             ['frederson', 5, 6, 7.5, 10],
             [np.nan, np.nan, np.nan, np.nan, np.nan],
             ['maggie', 8, 9, 10.5, 1],
             ['Fredrick', 11, 12, 13, 1]],
            columns=['a', 'b', 'c', 'd', 'e'])

        self.defaults = { 'colnames':'', 'col1':'', 'col2':'', 'single_value_selector':0, 'outcolname':'' }

    def test_no_multicolumn(self):
        # Missing columns on a multi-column operation
        params = {**self.defaults, 'operation':menu.index('Sum') }
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

    def test_no_col1(self):
        # Missing first column on a two-column operation
        params = {**self.defaults, 'operation':menu.index('Subtract'), 'col1':'', 'col2':'a'}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

    def test_no_col2(self):
        # Missing second column on a two-column operation
        params = {**self.defaults, 'operation':menu.index('Subtract'), 'col1':'a', 'col2':''}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

    def test_add(self):
        params = {**self.defaults, 'operation':menu.index('Sum'), 'colnames':'b,c,d' }
        newtab = self.table.copy()
        newtab['Sum of b, c, d'] = newtab[['b','c','d']].agg('sum', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_output_name_multicolumn(self):
        params = {**self.defaults, 'operation':menu.index('Sum'), 'colnames':'b,c,d', 'outcolname':'Fish' }
        newtab = self.table.copy()
        newtab['Fish'] = newtab[['b','c','d']].agg('sum', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_empty_output_name(self):
        # empty column name should get default automatic name
        params = {**self.defaults, 'operation':menu.index('Sum'), 'colnames':'b,c,d', 'outcolname':'' }
        newtab = self.table.copy()
        newtab['Sum of b, c, d'] = newtab[['b','c','d']].agg('sum', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_missing_output_name(self):
        # compatibility with prior paramster versions in the database 
        params = {**self.defaults, 'operation':menu.index('Sum'), 'colnames':'b,c,d' }
        del params['outcolname']
        newtab = self.table.copy()
        newtab['Sum of b, c, d'] = newtab[['b','c','d']].agg('sum', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_add_many_cols(self):
        # test with four columns to check column naming case
        params = {**self.defaults,'operation':menu.index('Sum'), 'colnames':'b,c,d,e' }
        newtab = self.table.copy()
        newtab['Sum of 4 columns'] = newtab[['b','c','d','e']].agg('sum', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_add_constant(self):
        params = {  
            **self.defaults, 
            'operation':menu.index('Sum'), 
            'colnames':'b,c,d', 
            'single_value_selector':'Constant value', 
            'single_value_constant':'100'
        }

        newtab = self.table.copy()
        newtab['Sum of b, c, d'] = newtab[['b','c','d']].agg('sum', axis=1) + 100        
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_add_cell(self):
        params = {  
            **self.defaults, 
            'operation':menu.index('Sum'), 
            'colnames':'b,c,d', 
            'single_value_selector':1, # 'Cell value'
            'single_value_col':'e',
            'single_value_row':'2'
        }

        newtab = self.table.copy()
        newtab['Sum of b, c, d'] = newtab[['b','c','d']].agg('sum', axis=1) + newtab['e'][1]  
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_multiply_constant(self):
        params = {  
            **self.defaults, 
            'operation':menu.index('Multiply'), 
            'colnames':'b,c,d', 
            'single_value_selector':'Constant value', 
            'single_value_constant':'100'
        }

        newtab = self.table.copy()
        newtab['Product of b, c, d'] = newtab[['b','c','d']].agg('product', axis=1) * 100        
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_multicolumn_nop(self):
        # if only one column supplied, does nothing
        params = {**self.defaults, 'operation':menu.index('Multiply'), 'colnames':'c' }
        newtab = self.table.copy()
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_average(self):
        params = {**self.defaults, 'operation':menu.index('Average'), 'colnames':'b,c,d' }
        newtab = self.table.copy()
        newtab['Average of b, c, d'] = newtab[['b','c','d']].agg('mean', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_sum_non_numeric(self):
        params = {**self.defaults, 'operation':menu.index("Sum"), 'colnames':'a,b' }
        out = render(self.table, params)
        self.assertTrue(isinstance(out, str))      

    def test_subtract(self):
        params = {**self.defaults, 'operation':menu.index('Subtract'), 'col1':'c', 'col2':'d'}
        out = render(self.table, params)
        newtab = self.table.copy()
        newtab['c minus d'] = newtab['c'] - newtab['d']
        self.assertTrue(out.equals(newtab)) 

    def test_two_column_output_name(self):
        # Different code path for many cols vs. two cols operations, so  test this here too
        params = {**self.defaults, 'operation':menu.index('Subtract'), 'col1':'c', 'col2':'d', 'outcolname':'Fish'}
        out = render(self.table, params)
        newtab = self.table.copy()
        newtab['Fish'] = newtab['c'] - newtab['d']
        self.assertTrue(out.equals(newtab)) 

    def test_percent_change(self):
        params = {**self.defaults, 'operation':menu.index('Percentage change'), 'col1':'e', 'col2':'b'}
        newtab = self.table.copy()
        newtab['Percent change e to b'] = (newtab['b'] - newtab['e']) / newtab['e']
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_percentage_of(self):
        params = {**self.defaults, 'operation':menu.index('X percent of Y'), 'col1':'e', 'col2':'b'}
        newtab = self.table.copy()
        newtab['e percent of b'] = newtab['b'] * newtab['e']
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

if __name__ == '__main__':
    unittest.main()