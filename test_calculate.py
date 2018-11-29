import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from calculate import render

# turn menu strings into indices for parameter dictionary
# must be kept in sync with filter.json
menutext = "Add|Subtract|Multiply|Divide||Average|Median|Minimum|Maximum||Percentage change|X percent of Y|X is what percent of Y"
menu = menutext.split('|')

class TestCalculate(unittest.TestCase):

    def setUp(self):
        # Test data includes some non number columns and some nulls and values to check type handling
        self.table = pd.DataFrame(
            [['fred', 2, 3, 4.5, 1],
             ['frederson', 5, 6, 7.5, 1],
             [np.nan, np.nan, np.nan, np.nan, np.nan],
             ['maggie', 8, 9, 10.5, 1],
             ['Fredrick', 11, 12, 13, 1]],
            columns=['a', 'b', 'c', 'd', 'e'])

    def test_no_multicolumn(self):
        # Missing columns on a multi-column operation
        params = {'operation':menu.index('Add'), 'colnames':'', 'col1':'', 'col2':''}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

    def test_no_col1(self):
        # Missing first column on a two-column operation
        params = {'operation':menu.index('Subtract'), 'colnames':'', 'col1':'', 'col2':'a'}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

    def test_no_col2(self):
        # Missing second column on a two-column operation
        params = {'operation':menu.index('Subtract'), 'colnames':'', 'col1':'a', 'col2':''}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

    def test_multicolumn_add(self):
        params = {'operation':menu.index('Add'), 'colnames':'b,c,d', 'col1':'', 'col2':''}
        newtab = self.table.copy()
        newtab['Sum of b, c, d'] = newtab[['b','c','d']].agg('sum', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_multicolumn_add_many_cols(self):
        # test with four columns to check column naming case
        params = {'operation':menu.index('Add'), 'colnames':'b,c,d,e', 'col1':'', 'col2':''}
        newtab = self.table.copy()
        newtab['Sum of 4 columns'] = newtab[['b','c','d','e']].agg('sum', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_multicolumn_nop(self):
        # if only one column supplied, does nothing
        params = {'operation':menu.index('Multiply'), 'colnames':'c', 'col1':'', 'col2':''}
        newtab = self.table.copy()
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_multicolumn_average(self):
        params = {'operation':menu.index('Average'), 'colnames':'b,c,d', 'col1':'', 'col2':''}
        newtab = self.table.copy()
        newtab['Average of b, c, d'] = newtab[['b','c','d']].agg('mean', axis=1)
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_multicolumn_non_numeric(self):
        params = {'operation':menu.index("Add"), 'colnames':'a,b', 'col1':'', 'col2':''}
        out = render(self.table, params)
        self.assertTrue(isinstance(out, str))      

    def test_two_column_subtract(self):
        params = {'operation':menu.index('Subtract'), 'colnames':'', 'col1':'c', 'col2':'d'}
        out = render(self.table, params)
        newtab = self.table.copy()
        newtab['c minus d'] = newtab['c'] - newtab['d']
        self.assertTrue(out.equals(newtab)) 

    def test_two_column_percent_change(self):
        params = {'operation':menu.index('Percentage change'), 'colnames':'', 'col1':'e', 'col2':'b'}
        newtab = self.table.copy()
        newtab['Percent change e to b'] = (newtab['b'] - newtab['e']) / newtab['e']
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

    def test_two_column_percentage_of(self):
        params = {'operation':menu.index('X percent of Y'), 'colnames':'', 'col1':'e', 'col2':'b'}
        newtab = self.table.copy()
        newtab['e percent of b'] = newtab['b'] * newtab['e']
        out = render(self.table, params)
        self.assertTrue(out.equals(newtab)) 

if __name__ == '__main__':
    unittest.main()