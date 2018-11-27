import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from calculate import render, op_names, op_functions, multicolumn_ops

# turn menu strings into indices for parameter dictionary
# must be kept in sync with filter.json
menutext = "Add|Subtract|Multiply|Divide||Average|Median|Minimum|Maximum||Percentage change|X percent of Y|X is what percent of Y"
menu = menutext.split('|')

two_column_ops = set(menu) - set(multicolumn_ops) - set(['']) # remove empty item from ||

class TestCalculate(unittest.TestCase):

    def setUp(self):
        # Test data includes some non number columns and some nulls and values to check type handling
        self.table = pd.DataFrame(
            [['fred', 2, 3, 4.5],
             ['frederson', 5, 6, 7.5],
             [np.nan, np.nan, np.nan, np.nan],
             ['maggie', 8, 9, 10.5],
             ['Fredrick', 11, 12, 13]],
            columns=['a', 'b', 'c', 'd'])

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

    def test_multicolumn_ops(self):
        for op in multicolumn_ops:
            params = {'operation':menu.index(op), 'colnames':'b,c,d', 'col1':'', 'col2':''}
            out = render(self.table, params)
            newtab = self.table.copy()
            newtab[op_names[op]] = newtab[['b','c','d']].agg(op_functions[op], axis=1)
            self.assertTrue(out.equals(newtab)) 

    def test_multicolumn_non_numeric(self):
        params = {'operation':menu.index("Add"), 'colnames':'a,b', 'col1':'', 'col2':''}
        out = render(self.table, params)
        self.assertTrue(isinstance(out, str))         

    def test_two_column_ops(self):
        print(two_column_ops)
        for op in two_column_ops:
            params = {'operation':menu.index(op), 'colnames':'', 'col1':'c', 'col2':'d'}
            out = render(self.table, params)
            newtab = self.table.copy()
            newtab[op_names[op]] = op_functions[op](newtab['c'], newtab['d'])
            self.assertTrue(out.equals(newtab)) 

if __name__ == '__main__':
    unittest.main()