from collections import namedtuple
import unittest
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal
import calculate


DefaultParams = {
    'operation': 'add',
    'colnames': '',
    'col1': '',
    'col2': '',
    'single_value_selector': 'none',
    'single_value_col': '',
    'single_value_row': 1,
    'single_value_constant': 1.0,
    'outcolname': '',
}


# Quick param (dict) factory
def P(**kwargs):
    assert not (set(kwargs.keys()) - set(DefaultParams.keys()))
    return {
        **DefaultParams,
        **kwargs,
    }


Column = namedtuple('Column', ('name', 'type', 'format'))


def render(table, params, input_columns=None):
    """
    calculate.render() helper that automatically adds input_columns to
    arguments if they aren't specified..

    TODO make a test framework so we don't have to implement this here.
    """
    if input_columns is None:
        def _infer_input_column(series: pd.Series) -> Column:
            if is_numeric_dtype(series):
                return Column(series.name, 'number', '{:,}')
            else:
                return Column(series.name, 'text', None)

        input_columns = {c: _infer_input_column(table[c])
                         for c in table.columns}

    return calculate.render(table, params, input_columns=input_columns)


class TestCalculate(unittest.TestCase):
    def setUp(self):
        # Test data includes some non number columns and some nulls and values
        # to check type handling
        self.table = pd.DataFrame(
            [['fred', 2, 3, 4.5, 1],
             ['frederson', 5, 6, 7.5, 10],
             [np.nan, np.nan, np.nan, np.nan, np.nan],
             ['maggie', 8, 9, 10.5, 1],
             ['Fredrick', 11, 12, 13, 1]],
            columns=['a', 'b', 'c', 'd', 'e'])

    def test_no_multicolumn(self):
        # Missing columns on a multi-column operation
        result = render(self.table, P(operation='add', colnames=''))
        # should NOP when first applied
        assert_frame_equal(result, self.table)

    def test_no_col1(self):
        # Missing first column on a two-column operation
        result = render(self.table,
                        P(operation='subtract', col1='', col2='a'))
        # should NOP when first applied
        assert_frame_equal(result, self.table)

    def test_no_col2(self):
        # Missing second column on a two-column operation
        result = render(self.table,
                        P(operation='subtract', col1='a', col2=''))
        # should NOP when first applied
        assert_frame_equal(result, self.table)

    def test_add(self):
        result = render(pd.DataFrame({
            'b': [1, 2, 3],
            'c': [1.2, 2.3, 3.4],
            'd': [np.nan, 2, 2],
        }), P(operation='add', colnames='b,c,d'))
        expected = pd.DataFrame({
            'b': [1, 2, 3],
            'c': [1.2, 2.3, 3.4],
            'd': [np.nan, 2, 2],
            'Sum of b, c, d': [2.2, 6.3, 8.4],
        })
        assert_frame_equal(result['dataframe'], expected)

    def test_output_name_multicolumn(self):
        result = render(pd.DataFrame({
            'b': [1.0],
            'c': [2.0],
            'd': [3.0],
        }), P(operation='add', colnames='b,c,d', outcolname='X'))
        expected = pd.DataFrame({
            'b': [1.0],
            'c': [2.0],
            'd': [3.0],
            'X': [6.0],
        })
        assert_frame_equal(result['dataframe'], expected)

    def test_output_name_multicolumn_provided(self):
        result = render(pd.DataFrame({'b': [1.0], 'c': [2.0], 'd': [3.0]}),
                        P(operation='add', colnames='b,c,d', outcolname='X'))
        expected = pd.DataFrame(
            {'b': [1.0], 'c': [2.0], 'd': [3.0], 'X': [6.0]}
        )
        assert_frame_equal(result['dataframe'], expected)

    def test_output_name_multicolumn_default(self):
        result = render(pd.DataFrame({'b': [1.0], 'c': [2.0], 'd': [3.0]}),
                        P(operation='add', colnames='b,c,d', outcolname=''))
        expected = pd.DataFrame(
            {'b': [1.0], 'c': [2.0], 'd': [3.0], 'Sum of b, c, d': [6.0]}
        )
        assert_frame_equal(result['dataframe'], expected)

    def test_output_name_multicolumn_default_with_many_cols(self):
        result = render(
            pd.DataFrame({'b': [1.0], 'c': [2.0], 'd': [3.0], 'e': [4.0]}),
            P(operation='add', colnames='b,c,d,e', outcolname=''))
        expected = pd.DataFrame(
            {'b': [1.0], 'c': [2.0], 'd': [3.0], 'e': [4.0],
             'Sum of 4 columns': [10.0]}
        )
        assert_frame_equal(result['dataframe'], expected)

    def test_add_constant(self):
        result = render(
            pd.DataFrame({'b': [1.0], 'c': [2.0]}),
            P(operation='add', colnames='b,c', outcolname='X',
              single_value_selector='constant', single_value_constant=100.0)
        )
        expected = pd.DataFrame({'b': [1.0], 'c': [2.0], 'X': [103.0]})
        assert_frame_equal(result['dataframe'], expected)

    def test_add_cell(self):
        result = render(
            pd.DataFrame({
                'b': [1.0, 1.1],
                'c': [2.0, 2.1],
                'd': [3.0, 3.1],
            }),
            P(operation='add', colnames='b,c,d', outcolname='X',
              single_value_selector='cell', single_value_row=2,
              single_value_col='d'),
            {
                'b': Column('b', 'number', '{:,.2f}'),
                'c': Column('c', 'number', '{:,.1%}'),
                'd': Column('d', 'number', '{:,}'),
            }
        )
        expected = pd.DataFrame({
            'b': [1.0, 1.1],
            'c': [2.0, 2.1],
            'd': [3.0, 3.1],
            'X': [9.1, 9.4],  # 6.0+3.1, 6.3+3.1
        })
        assert_frame_equal(result['dataframe'], expected)
        # multicolumn op: Use format from first column
        self.assertEqual(result['column_formats'], {'X': '{:,.2f}'})

    def test_add_cell_not_number(self):
        result = render(
            pd.DataFrame({
                'b': [1.0, 1.1],
                'c': [2.0, 2.1],
                's': ['a', 'b'],
            }),
            P(operation='add', colnames='b,c', outcolname='X',
              single_value_selector='cell', single_value_row=2,
              single_value_col='s'),
            {
                'b': Column('b', 'number', '{:,.2f}'),
                'c': Column('c', 'number', '{:,.1%}'),
                's': Column('s', 'text', ''),
            }
        )
        self.assertEqual(result, 'The chosen cell does not contain a number')

    def test_add_cell_nan(self):
        result = render(
            pd.DataFrame({
                'b': [1.0, 1.1],
                'c': [2.0, 2.1],
                'd': [3.0, np.nan],
            }),
            P(operation='add', colnames='b,c', outcolname='X',
              single_value_selector='cell', single_value_row=2,
              single_value_col='d'),
            {
                'b': Column('b', 'number', '{:,.2f}'),
                'c': Column('c', 'number', '{:,.1%}'),
                'd': Column('d', 'number', '{:,}'),
            }
        )
        self.assertEqual(result, 'The chosen cell does not contain a number')

    def test_multiply_constant(self):
        result = render(
            pd.DataFrame({'b': [1.0], 'c': [2.0]}),
            P(operation='multiply', colnames='b,c', outcolname='X',
              single_value_selector='constant', single_value_constant=100.0)
        )
        expected = pd.DataFrame({'b': [1.0], 'c': [2.0], 'X': [200.0]})
        assert_frame_equal(result['dataframe'], expected)

    def test_only_one_column_no_op(self):
        # if only one column supplied, does nothing
        result = render(
            pd.DataFrame({'b': [1.0], 'c': [2.0]}),
            P(operation='multiply', colnames='b', outcolname='X')
        )
        expected = pd.DataFrame({'b': [1.0], 'c': [2.0]})
        assert_frame_equal(result, expected)

    def test_mean(self):
        result = render(
            pd.DataFrame({'a': [1, np.nan], 'b': [2.0, 3.0], 'c': [2, 2]}),
            P(operation='mean', colnames='a,b,c', outcolname='X')
        )
        expected = pd.DataFrame({
            'a': [1, np.nan],
            'b': [2.0, 3.0],
            'c': [2, 2],
            'X': [5.0 / 3, 2.5],
        })
        assert_frame_equal(result['dataframe'], expected)

    def test_mean_of_zero_sum_is_zero(self):
        result = render(
            pd.DataFrame({'A': [1], 'B': [-1]}),
            P(operation='mean', colnames='A,B', outcolname='X')
        )
        expected = pd.DataFrame({
            'A': [1],
            'B': [-1],
            'X': [0.0],  # not divide-by-zero error or inf or -inf.
        })

    def test_subtract(self):
        result = render(
            pd.DataFrame({'a': [1, 2], 'b': [2, np.nan]}),
            P(operation='subtract', col1='a', col2='b', outcolname='X'),
            {
                'a': Column('a', 'number', '{:,.2f}'),
                'b': Column('b', 'number', '{:.1%}'),
            }
        )
        expected = pd.DataFrame({
            'a': [1, 2],
            'b': [2, np.nan],
            # testing operation and NaN behavior all in one
            'X': [-1, np.nan],
        })
        assert_frame_equal(result['dataframe'], expected)
        # binary op: use format from first column
        self.assertEqual(result['column_formats'], {'X': '{:,.2f}'})

    def test_div_by_zero_is_nan(self):
        result = render(
            pd.DataFrame({
                'A': [1, -2, 3, -4],
                'B': [0, 0, 1, np.nan],
            }),
            P(operation='divide', col1='A', col2='B', outcolname='X'),
            {
                'A': Column('A', 'number', '{:,.2f}'),
                'B': Column('B', 'number', '{:.1%}'),
            }
        )
        expected = pd.DataFrame({
            'A': [1, -2, 3, -4],
            'B': [0, 0, 1, np.nan],
            'X': [np.nan, np.nan, 3, np.nan],
        })
        assert_frame_equal(result['dataframe'], expected)

    def test_two_column_default_output_name(self):
        result = render(
            pd.DataFrame({'a': [1], 'b': [2]}),
            P(operation='subtract', col1='a', col2='b')
        )
        expected = pd.DataFrame({'a': [1], 'b': [2], 'a minus b': [-1]})
        assert_frame_equal(result['dataframe'], expected)

    def test_percent_change(self):
        result = render(
            pd.DataFrame({'a': [1, 2], 'b': [1.6, np.nan]}),
            P(operation='percent_change', col1='a', col2='b')
        )
        expected = pd.DataFrame({
            'a': [1, 2],
            'b': [1.6, np.nan],
            'Percent change a to b': [0.6, np.nan],
        })
        assert_frame_equal(result['dataframe'], expected)
        self.assertEqual(result['column_formats'],
                         {'Percent change a to b': '{:,.1%}'})

    def test_percent_change_from_zero_is_nan(self):
        # test auto-colname, NaN and normal behavior all in one
        result = render(
            pd.DataFrame({'a': [1, 0], 'b': [1.6, 2]}),
            P(operation='percent_change', col1='a', col2='b')
        )
        expected = pd.DataFrame({
            'a': [1, 0],
            'b': [1.6, 2],
            'Percent change a to b': [0.6, np.nan],
        })
        assert_frame_equal(result['dataframe'], expected)
        self.assertEqual(result['column_formats'],
                         {'Percent change a to b': '{:,.1%}'})

    def test_percent_multiply(self):
        # test auto-colname, NaN and normal behavior all in one
        result = render(
            pd.DataFrame({'a': [0.6, 2], 'b': [1.6, np.nan]}),
            P(operation='percent_multiply', col1='a', col2='b')
        )
        expected = pd.DataFrame({
            'a': [0.6, 2],
            'b': [1.6, np.nan],
            'a percent of b': [0.6 * 1.6, np.nan],
        })
        assert_frame_equal(result['dataframe'], expected)
        self.assertEqual(result['column_formats'],
                         {'a percent of b': '{:,.1%}'})

    def test_percent_divide_over_zero_is_null(self):
        result = render(
            pd.DataFrame({'a': [1, 0.5], 'b': [1.6, 0]}),
            P(operation='percent_divide', col1='a', col2='b')
        )
        expected = pd.DataFrame({
            'a': [1, 0.5],
            'b': [1.6, 0],
            'a is this percent of b': [0.625, np.nan],
        })
        assert_frame_equal(result['dataframe'], expected)


if __name__ == '__main__':
    unittest.main()
