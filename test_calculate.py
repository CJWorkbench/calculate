import unittest
from typing import NamedTuple, Optional

import calculate
import numpy as np
import pandas as pd
from cjwmodule.testing.i18n import cjwmodule_i18n_message, i18n_message
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal

DefaultParams = {
    "operation": "add",
    "colnames": [],
    "col1": "",
    "col2": "",
    "single_value_selector": "none",
    "single_value_col": "",
    "single_value_row": 1,
    "single_value_constant": 1.0,
    "outcolname": "",
}


# Quick param (dict) factory
# don't allow keys that are not in DefaultParams, fill any non-specified keys with defaults
def P(**kwargs):
    assert not (set(kwargs.keys()) - set(DefaultParams.keys()))
    return {**DefaultParams, **kwargs}


class Column(NamedTuple):
    name: str
    type: str
    format: Optional[str]


class Settings(NamedTuple):
    MAX_BYTES_PER_COLUMN_NAME: int = 120


def render(table, params, *, settings=Settings(), input_columns=None):
    """
    calculate.render() helper that automatically adds input_columns to
    arguments if they aren't specified..

    TODO make a test framework so we don't have to implement this here.
    """
    if input_columns is None:

        def _infer_input_column(series: pd.Series) -> Column:
            if is_numeric_dtype(series):
                return Column(series.name, "number", "{:,}")
            else:
                return Column(series.name, "text", None)

        input_columns = {c: _infer_input_column(table[c]) for c in table.columns}

    return calculate.render(
        table, params, settings=settings, input_columns=input_columns
    )


class MigrateParamsTest(unittest.TestCase):
    def test_v0(self):
        self.assertEqual(
            calculate.migrate_params(
                {
                    "xtext": "",
                    "ytext": "",
                    "dividetext": "",
                    "newvaluetext": "",
                    "oldvaluetext": "",
                    "subtracttext": "",
                    "add_additional": "",
                    "multiply_additional": "",
                    "operation": 5,
                    "colnames": "",
                    "col1": "",
                    "col2": "",
                    "single_value_selector": 2,
                    "single_value_col": "",
                    "single_value_row": 1,
                    "single_value_constant": 1.0,
                    # 'outcolname': '', -- it can be missing!
                }
            ),
            {
                "operation": "mean",
                "colnames": [],
                "col1": "",
                "col2": "",
                "single_value_selector": "constant",
                "single_value_col": "",
                "single_value_row": 1,
                "single_value_constant": 1.0,
                "outcolname": "",
            },
        )

    def test_v0_alt(self):
        self.assertEqual(
            calculate.migrate_params(
                {
                    "operation": 5,
                    "colnames": "",
                    "col1": "",
                    "col2": "",
                    "single_value_selector": 2,
                    "single_value_col": "",
                    "single_value_row": 1,
                    "single_value_constant": 1.0,
                    # 'outcolname': '', -- it can be missing!
                }
            ),
            {
                "operation": "mean",
                "colnames": [],
                "col1": "",
                "col2": "",
                "single_value_selector": "constant",
                "single_value_col": "",
                "single_value_row": 1,
                "single_value_constant": 1.0,
                "outcolname": "",
            },
        )

    def test_v1(self):
        self.assertEqual(
            calculate.migrate_params(
                {
                    "operation": 5,
                    "colnames": "",
                    "col1": "",
                    "col2": "",
                    "single_value_selector": 2,
                    "single_value_col": "",
                    "single_value_row": 1,
                    "single_value_constant": 1.0,
                    "outcolname": "",
                }
            ),
            {
                "operation": "mean",
                "colnames": [],
                "col1": "",
                "col2": "",
                "single_value_selector": "constant",
                "single_value_col": "",
                "single_value_row": 1,
                "single_value_constant": 1.0,
                "outcolname": "",
            },
        )

    def test_v2_no_colnames(self):
        self.assertEqual(
            calculate.migrate_params(
                {
                    "operation": "add",
                    "colnames": "",
                    "col1": "",
                    "col2": "",
                    "single_value_selector": "none",
                    "single_value_col": "",
                    "single_value_row": 1,
                    "single_value_constant": 1.0,
                    "outcolname": "",
                }
            ),
            {
                "operation": "add",
                "colnames": [],
                "col1": "",
                "col2": "",
                "single_value_selector": "none",
                "single_value_col": "",
                "single_value_row": 1,
                "single_value_constant": 1.0,
                "outcolname": "",
            },
        )

    def test_v2(self):
        self.assertEqual(
            calculate.migrate_params(
                {
                    "operation": "add",
                    "colnames": "A,B",
                    "col1": "",
                    "col2": "",
                    "single_value_selector": "none",
                    "single_value_col": "",
                    "single_value_row": 1,
                    "single_value_constant": 1.0,
                    "outcolname": "",
                }
            ),
            {
                "operation": "add",
                "colnames": ["A", "B"],
                "col1": "",
                "col2": "",
                "single_value_selector": "none",
                "single_value_col": "",
                "single_value_row": 1,
                "single_value_constant": 1.0,
                "outcolname": "",
            },
        )

    def test_v3(self):
        self.assertEqual(
            calculate.migrate_params(
                {
                    "operation": "add",
                    "colnames": ["A", "B"],
                    "col1": "",
                    "col2": "",
                    "single_value_selector": "none",
                    "single_value_col": "",
                    "single_value_row": 1,
                    "single_value_constant": 1.0,
                    "outcolname": "",
                }
            ),
            {
                "operation": "add",
                "colnames": ["A", "B"],
                "col1": "",
                "col2": "",
                "single_value_selector": "none",
                "single_value_col": "",
                "single_value_row": 1,
                "single_value_constant": 1.0,
                "outcolname": "",
            },
        )


class RenderTest(unittest.TestCase):
    def setUp(self):
        # Test data includes some non number columns and some nulls and values
        # to check type handling
        self.table = pd.DataFrame(
            [
                ["fred", 2, 3, 4.5, 1],
                ["frederson", 5, 6, 7.5, 10],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                ["maggie", 8, 9, 10.5, 1],
                ["Fredrick", 11, 12, 13, 1],
            ],
            columns=["a", "b", "c", "d", "e"],
        )

    def test_no_multicolumn(self):
        # Missing columns on a multi-column operation
        result = render(self.table, P(operation="add", colnames=[]))
        # should NOP when first applied
        assert_frame_equal(result, self.table)

    def test_no_col1(self):
        # Missing first column on a two-column operation
        result = render(self.table, P(operation="subtract", col1="", col2="a"))
        # should NOP when first applied
        assert_frame_equal(result, self.table)

    def test_no_col2(self):
        # Missing second column on a two-column operation
        result = render(self.table, P(operation="subtract", col1="a", col2=""))
        # should NOP when first applied
        assert_frame_equal(result, self.table)

    def test_add(self):
        result = render(
            pd.DataFrame({"b": [1, 2, 3], "c": [1.2, 2.3, 3.4], "d": [np.nan, 2, 2]}),
            P(operation="add", colnames=["b", "c", "d"]),
        )
        expected = pd.DataFrame(
            {
                "b": [1, 2, 3],
                "c": [1.2, 2.3, 3.4],
                "d": [np.nan, 2, 2],
                "Sum of b, c, d": [2.2, 6.3, 8.4],
            }
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_output_name_multicolumn(self):
        result = render(
            pd.DataFrame({"b": [1.0], "c": [2.0], "d": [3.0]}),
            P(operation="add", colnames=["b", "c", "d"], outcolname="X"),
        )
        expected = pd.DataFrame({"b": [1.0], "c": [2.0], "d": [3.0], "X": [6.0]})
        assert_frame_equal(result["dataframe"], expected)

    def test_output_name_multicolumn_provided(self):
        result = render(
            pd.DataFrame({"b": [1.0], "c": [2.0], "d": [3.0]}),
            P(operation="add", colnames=["b", "c", "d"], outcolname="X"),
        )
        expected = pd.DataFrame({"b": [1.0], "c": [2.0], "d": [3.0], "X": [6.0]})
        assert_frame_equal(result["dataframe"], expected)

    def test_output_name_multicolumn_default(self):
        result = render(
            pd.DataFrame({"b": [1.0], "c": [2.0], "d": [3.0]}),
            P(operation="add", colnames=["b", "c", "d"], outcolname=""),
        )
        expected = pd.DataFrame(
            {"b": [1.0], "c": [2.0], "d": [3.0], "Sum of b, c, d": [6.0]}
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_output_name_multicolumn_default_with_many_cols(self):
        result = render(
            pd.DataFrame({"b": [1.0], "c": [2.0], "d": [3.0], "e": [4.0]}),
            P(operation="add", colnames=["b", "c", "d", "e"], outcolname=""),
        )
        expected = pd.DataFrame(
            {"b": [1.0], "c": [2.0], "d": [3.0], "e": [4.0], "Sum of 4 columns": [10.0]}
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_add_constant(self):
        result = render(
            pd.DataFrame({"b": [1.0], "c": [2.0]}),
            P(
                operation="add",
                colnames=["b", "c"],
                outcolname="X",
                single_value_selector="constant",
                single_value_constant=100.0,
            ),
        )
        expected = pd.DataFrame({"b": [1.0], "c": [2.0], "X": [103.0]})
        assert_frame_equal(result["dataframe"], expected)

    def test_add_cell(self):
        result = render(
            pd.DataFrame({"b": [1.0, 1.1], "c": [2.0, 2.1], "d": [3.0, 3.1]}),
            P(
                operation="add",
                colnames=["b", "c", "d"],
                outcolname="X",
                single_value_selector="cell",
                single_value_row=2,
                single_value_col="d",
            ),
            input_columns={
                "b": Column("b", "number", "{:,.2f}"),
                "c": Column("c", "number", "{:,.1%}"),
                "d": Column("d", "number", "{:,}"),
            },
        )
        expected = pd.DataFrame(
            {
                "b": [1.0, 1.1],
                "c": [2.0, 2.1],
                "d": [3.0, 3.1],
                "X": [9.1, 9.4],  # 6.0+3.1, 6.3+3.1
            }
        )
        assert_frame_equal(result["dataframe"], expected)
        # multicolumn op: Use format from first column
        self.assertEqual(result["column_formats"], {"X": "{:,.2f}"})

    def test_add_cell_not_number(self):
        result = render(
            pd.DataFrame({"b": [1.0, 1.1], "c": [2.0, 2.1], "s": ["a", "b"]}),
            P(
                operation="add",
                colnames=["b", "c"],
                outcolname="X",
                single_value_selector="cell",
                single_value_row=2,
                single_value_col="s",
            ),
            input_columns={
                "b": Column("b", "number", "{:,.2f}"),
                "c": Column("c", "number", "{:,.1%}"),
                "s": Column("s", "text", ""),
            },
        )
        self.assertEqual(
            result, i18n_message("badParam.single_value_col.notANumber"),
        )

    def test_add_cell_no_column_selected(self):
        result = render(
            pd.DataFrame({"b": [1.0, 1.1], "c": [2.0, 2.1], "d": [3.0, 3.1]}),
            P(
                operation="add",
                colnames=["b", "c", "d"],
                outcolname="X",
                single_value_selector="cell",
                single_value_row=2,
                single_value_col="",
            ),
            input_columns={
                "b": Column("b", "number", "{:,.2f}"),
                "c": Column("c", "number", "{:,.1%}"),
                "d": Column("d", "number", "{:,}"),
            },
        )
        self.assertEqual(
            result, i18n_message("badParam.single_value_col.missing"),
        )

    def test_add_cell_nan(self):
        result = render(
            pd.DataFrame({"b": [1.0, 1.1], "c": [2.0, 2.1], "d": [3.0, np.nan]}),
            P(
                operation="add",
                colnames=["b", "c"],
                outcolname="X",
                single_value_selector="cell",
                single_value_row=2,
                single_value_col="d",
            ),
            input_columns={
                "b": Column("b", "number", "{:,.2f}"),
                "c": Column("c", "number", "{:,.1%}"),
                "d": Column("d", "number", "{:,}"),
            },
        )
        self.assertEqual(
            result, i18n_message("badParam.single_value_col.notANumber"),
        )

    def test_multiply_constant(self):
        result = render(
            pd.DataFrame({"b": [1.0], "c": [2.0]}),
            P(
                operation="multiply",
                colnames=["b", "c"],
                outcolname="X",
                single_value_selector="constant",
                single_value_constant=100.0,
            ),
        )
        expected = pd.DataFrame({"b": [1.0], "c": [2.0], "X": [200.0]})
        assert_frame_equal(result["dataframe"], expected)

    def test_only_one_column_no_op(self):
        # if only one column supplied, does nothing
        result = render(
            pd.DataFrame({"b": [1.0], "c": [2.0]}),
            P(operation="multiply", colnames=["b"], outcolname="X"),
        )
        expected = pd.DataFrame({"b": [1.0], "c": [2.0]})
        assert_frame_equal(result, expected)

    def test_mean(self):
        result = render(
            pd.DataFrame({"a": [1, np.nan], "b": [2.0, 3.0], "c": [2, 2]}),
            P(operation="mean", colnames=["a", "b", "c"], outcolname="X"),
        )
        expected = pd.DataFrame(
            {"a": [1, np.nan], "b": [2.0, 3.0], "c": [2, 2], "X": [5.0 / 3, 2.5]}
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_mean_of_zero_sum_is_zero(self):
        result = render(
            pd.DataFrame({"A": [1], "B": [-1]}),
            P(operation="mean", colnames=["A", "B"], outcolname="X"),
        )
        expected = pd.DataFrame(
            {
                "A": [1],
                "B": [-1],
                "X": [0.0],  # not divide-by-zero error or inf or -inf.
            }
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_subtract(self):
        result = render(
            pd.DataFrame({"a": [1, 2], "b": [2, np.nan]}),
            P(operation="subtract", col1="a", col2="b", outcolname="X"),
            input_columns={
                "a": Column("a", "number", "{:,.2f}"),
                "b": Column("b", "number", "{:.1%}"),
            },
        )
        expected = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [2, np.nan],
                # testing operation and NaN behavior all in one
                "X": [-1, np.nan],
            }
        )
        assert_frame_equal(result["dataframe"], expected)
        # binary op: use format from first column
        self.assertEqual(result["column_formats"], {"X": "{:,.2f}"})

    def test_div_by_zero_is_nan(self):
        result = render(
            pd.DataFrame({"A": [1, -2, 3, -4], "B": [0, 0, 1, np.nan]}),
            P(operation="divide", col1="A", col2="B", outcolname="X"),
            input_columns={
                "A": Column("A", "number", "{:,.2f}"),
                "B": Column("B", "number", "{:.1%}"),
            },
        )
        expected = pd.DataFrame(
            {
                "A": [1, -2, 3, -4],
                "B": [0, 0, 1, np.nan],
                "X": [np.nan, np.nan, 3, np.nan],
            }
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_two_column_default_output_name(self):
        result = render(
            pd.DataFrame({"a": [1], "b": [2]}),
            P(operation="subtract", col1="a", col2="b"),
        )
        expected = pd.DataFrame({"a": [1], "b": [2], "a minus b": [-1]})
        assert_frame_equal(result["dataframe"], expected)

    def test_percent_change(self):
        result = render(
            pd.DataFrame({"a": [1, 2], "b": [1.6, np.nan]}),
            P(operation="percent_change", col1="a", col2="b"),
        )
        expected = pd.DataFrame(
            {"a": [1, 2], "b": [1.6, np.nan], "Percent change a to b": [0.6, np.nan]}
        )
        assert_frame_equal(result["dataframe"], expected)
        self.assertEqual(result["column_formats"], {"Percent change a to b": "{:,.1%}"})

    def test_percent_change_from_zero_is_nan(self):
        # test auto-colname, NaN and normal behavior all in one
        result = render(
            pd.DataFrame({"a": [1, 0], "b": [1.6, 2]}),
            P(operation="percent_change", col1="a", col2="b"),
        )
        expected = pd.DataFrame(
            {"a": [1, 0], "b": [1.6, 2], "Percent change a to b": [0.6, np.nan]}
        )
        assert_frame_equal(result["dataframe"], expected)
        self.assertEqual(result["column_formats"], {"Percent change a to b": "{:,.1%}"})

    def test_percent_multiply(self):
        # test auto-colname, NaN, and percentage/not percentage input behavior
        table = pd.DataFrame(
            {
                "a": [6, 6],  # 6% not percentage formatted
                "b": [0.06, 0.06],  # 6% as percentage formatted
                "c": [1.6, np.nan],
            }
        )

        # non-percentage formatted input, col A
        result = render(
            table,
            P(operation="percent_multiply", col1="a", col2="c"),
            input_columns={
                "a": Column("a", "number", "{:,}"),
                "c": Column("c", "number", "{:,}"),
            },
        )
        expected = table.copy()
        expected["a percent of c"] = [0.06 * 1.6, np.nan]

        assert_frame_equal(result["dataframe"], expected)
        self.assertEqual(result["column_formats"], {"a percent of c": "{:,}"})

        # percentage formatted input, col B
        result = render(
            table,
            P(operation="percent_multiply", col1="b", col2="c"),
            input_columns={
                "b": Column("b", "number", "{:,.1%}"),
                "c": Column("c", "number", "{:,}"),
            },
        )
        expected = table.copy()
        expected["b percent of c"] = [0.06 * 1.6, np.nan]

        assert_frame_equal(result["dataframe"], expected)
        self.assertEqual(result["column_formats"], {"b percent of c": "{:,}"})

    def test_percent_divide_over_zero_is_null(self):
        result = render(
            pd.DataFrame({"a": [1, 0.5], "b": [1.6, 0]}),
            P(operation="percent_divide", col1="a", col2="b"),
        )
        expected = pd.DataFrame(
            {"a": [1, 0.5], "b": [1.6, 0], "a is this percent of b": [0.625, np.nan]}
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_percent_of_column_sum(self):
        result = render(
            pd.DataFrame({"a": [1, 3]}), P(operation="percent_of_column_sum", col1="a")
        )
        expected = pd.DataFrame({"a": [1, 3], "Percent of a": [0.25, 0.75]})
        assert_frame_equal(result["dataframe"], expected)

    def test_percent_of_column_sum_null(self):
        result = render(
            pd.DataFrame({"a": [np.nan, 1, 3]}),
            P(operation="percent_of_column_sum", col1="a"),
        )
        expected = pd.DataFrame(
            {"a": [np.nan, 1, 3], "Percent of a": [np.nan, 0.25, 0.75]}
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_percent_of_column_sum_all_null(self):
        result = render(
            pd.DataFrame({"a": [np.nan, np.nan]}),
            P(operation="percent_of_column_sum", col1="a"),
        )
        expected = pd.DataFrame(
            {"a": [np.nan, np.nan], "Percent of a": [np.nan, np.nan]}
        )
        assert_frame_equal(result["dataframe"], expected)

    def test_percent_of_column_sum_inf(self):
        result = render(
            pd.DataFrame({"a": [-1, 1]}), P(operation="percent_of_column_sum", col1="a")
        )
        self.assertEqual(
            result, i18n_message("badData.percent_of_column_sum.sumIsZero"),
        )

    def test_truncate_result_column_name(self):
        result = render(
            pd.DataFrame({"A A A": [1], "A A B": [2]}),
            P(operation="divide", col1="A A A", col2="A A B"),
            settings=Settings(MAX_BYTES_PER_COLUMN_NAME=5),
        )
        expected = pd.DataFrame({"A A A": [1], "A A B": [2], "A A 2": [0.5]})
        assert_frame_equal(result["dataframe"], expected)
        self.assertEqual(
            result["errors"],
            [
                cjwmodule_i18n_message(
                    "util.colnames.warnings.truncated",
                    {"n_columns": 1, "first_colname": "A A 2", "n_bytes": 5},
                ),
                cjwmodule_i18n_message(
                    "util.colnames.warnings.numbered",
                    {"n_columns": 1, "first_colname": "A A 2"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
