import math
import warnings
import pandas as pd
from spanda.core.typing import *
from spanda.core.utils import wrap_col_args

# helper functions
sum_ = sum
abs_ = abs


def _elementwise_apply(f: Callable):
    def f_elementwise(x: pd.Series) -> pd.Series:
        return x.apply(f)
    return f_elementwise


# classes
class Column:
    """
    A column in Spanda dataframe
    """

    def __init__(self, name: Optional[str]):
        self.name = name
        self.op = lambda df: df[name]

    def setOp(self, op: Callable):
        self.op = op

    @staticmethod
    def getName(col: Union['Column', str]) -> str:
        if isinstance(col, Column):
            return col.name
        else:
            return str(col)

    @staticmethod
    def _apply(col: 'Column', df: pd.DataFrame) -> Union[pd.Series, Any]:
        if isinstance(col, Column):
            return col.op(df)
        else:
            return col

    @staticmethod
    def _transformColumn(name: str, operation: Callable) -> 'Column':
        col = Column(None)
        col.name = name
        col.op = operation
        return col

    def _simpleBinaryTransformColumn(self, opname: str, opfunc: Callable, other: Union['Column', Any],
                                     is_other_col: bool = True) -> 'Column':
        return Column._transformColumn(f"({Column.getName(self)} {opname} {Column.getName(other)})",
                                       lambda df: opfunc(Column._apply(self, df),
                                                         Column._apply(other, df) if is_other_col else other))

    def _simpleUnaryTransformColumn(self, opname: str, opfunc: Callable) -> 'Column':
        return Column._transformColumn(f"({opname}{Column.getName(self)})", lambda df: opfunc(Column._apply(self, df)))

    def __repr__(self):
        return f"<Column {self.name}>"

    def alias(self, name):
        """
        Return column with new name
        """

        return Column._transformColumn(name, self.op)

    def between(self, start, end):
        """
         A boolean expression that is evaluated to true if the value of this
         expression is between the given columns.

         >>> df.select('name', F.col('age').between(2, 4))
         +-----+---------------------------+
         | name|((age >= 2) AND (age <= 4))|
         +-----+---------------------------+
         |Alice|                       true|
         |  Bob|                      false|
         +-----+---------------------------+
         """

        return (self >= start) & (self <= end)

    def apply(self, func: Callable) -> 'Column':
        return udf(func)(self)

    def isin(self, values):
        """
        A boolean expression that is evaluated to true if the value of this
        expression is contained by the evaluated values of the arguments.
        """
        return self._simpleBinaryTransformColumn('IN', lambda x, y: x.isin(y), values, is_other_col=False)

    # operators
    def __eq__(self, other):
        return self._simpleBinaryTransformColumn('==', lambda x, y: x == y, other)

    def __ne__(self, other):
        return self._simpleBinaryTransformColumn('!=', lambda x, y: x != y, other)

    def __gt__(self, other):
        return self._simpleBinaryTransformColumn('>', lambda x, y: x > y, other)

    def __lt__(self, other):
        return self._simpleBinaryTransformColumn('<', lambda x, y: x < y, other)

    def __ge__(self, other):
        return self._simpleBinaryTransformColumn('>=', lambda x, y: x >= y, other)

    def __le__(self, other):
        return self._simpleBinaryTransformColumn('<=', lambda x, y: x <= y, other)

    def __add__(self, other):
        return self._simpleBinaryTransformColumn('+', lambda x, y: x + y, other)

    def __mul__(self, other):
        return self._simpleBinaryTransformColumn('*', lambda x, y: x * y, other)

    def __div__(self, other):
        return self._simpleBinaryTransformColumn('/', lambda x, y: x / y, other)

    def __sub__(self, other):
        return self._simpleBinaryTransformColumn('-', lambda x, y: x - y, other)

    def __and__(self, other):
        return self._simpleBinaryTransformColumn(' AND ', lambda x, y: x & y, other)

    def __or__(self, other):
        return self._simpleBinaryTransformColumn(' OR ', lambda x, y: x | y, other)

    def __neg__(self, other):
        return self._simpleUnaryTransformColumn('-', lambda x: -x)

    def __invert__(self):
        return self._simpleUnaryTransformColumn('NOT ', lambda x: ~x)


class AggColumn:
    def __init__(self, name, orig_col, op):
        self._name = name
        self._orig_col = orig_col
        self._op = op

    def alias(self, name):
        # TODO: might want to clone orig_col and op (maybe also in Column)
        return AggColumn(name=name, orig_col=self._orig_col, op=self._op)

    @staticmethod
    def getName(agg_col):
        return f"{agg_col._name} ({Column.getName(agg_col._orig_col)})"

    @staticmethod
    def _apply(agg_col: 'AggColumn', df: pd.DataFrame) -> pd.Series:
        return agg_col._op(Column._apply(agg_col._orig_col, df))

    def over(self, window_spec) -> Column:
        def f(df):
            inputs = Column._apply(self._orig_col, df)
            # in the following, row2grp represents the representative group of each row, while grp2rows is a dictionary
            # of all rows that the group aggregates over. possibly not intuitive but group may include rows that are
            # not represented by this group (for example if we apply lead(...) we aggregate over the next row which is
            # not represented by the current group)
            row2grp, grp2rows = window_spec._get_group_data(df)
            grp_agg_val = {grp: self._op(inputs.loc[vals]) for grp, vals in grp2rows.items()}
            data = {row_idx: grp_agg_val[grp_name] for row_idx, grp_name in row2grp.items()}
            return pd.Series(data=data, index=inputs.index)

        col = Column(name=f"{AggColumn.getName(self)} OVER ({window_spec._name})")
        col.setOp(f)
        return col


class WindowTransformationColumn(AggColumn):
    def __init(self, name: str, orig_col: Column, op: Callable):
        self._name = name
        self._orig_col = orig_col
        self._op = op

    @staticmethod
    def _apply(agg_col, df: pd.DataFrame):
        assert False, f"cannot aggregate grouped data using `{agg_col._name}`. can be used only over windows."

    def over(self, window_spec) -> Column:
        def f(df):
            orig_col = self._orig_col
            if orig_col is None:
                orig_col = window_spec._get_orderby_col()
            inputs = Column._apply(orig_col, df)
            row2grp, grp2rows = window_spec._get_group_data(df)
            data = {row_idx: self._op(inputs, grp2rows[grp_name], grp2rows[grp_name].index(row_idx))
                    for row_idx, grp_name in row2grp.items()}
            return pd.Series(data=data, index=inputs.index)

        col = Column(name=f"{AggColumn.getName(self)} OVER ({window_spec._name})")
        col.setOp(f)
        return col


# functions
def col(name: str) -> Column:
    """
    Creates a column object with this name
    """
    if name == "*":
        return Column._transformColumn("*", lambda df: df.apply(lambda x: tuple(x), axis='columns'))
    return Column(name)


def lit(value: Any):
    """
    Returns a column representing the literal value it received as value
    """
    return Column._transformColumn(f"LIT( {str(value)} )", lambda df: value)


# column functions
@wrap_col_args
def sqrt(col: Column) -> Column:
    """
    Computes square root
    """
    return col._simpleUnaryTransformColumn("SQRT ", _elementwise_apply(math.sqrt))


@wrap_col_args
def exp(col: Column) -> Column:
    """
    Computes exp function
    """
    return col._simpleUnaryTransformColumn("EXP ", _elementwise_apply(math.exp))


@wrap_col_args
def log(col: Column) -> Column:
    """
    Computes logarithm function
    """
    return col._simpleUnaryTransformColumn("LOG ", _elementwise_apply(math.log))


@wrap_col_args
def _abs(col: Column) -> Column:
    """
    Computes absolute value
    """
    return col._simpleUnaryTransformColumn("ABS ", _elementwise_apply(abs_))


@wrap_col_args
def array(*cols: Column) -> Column:
    """
    Return column of arrays
    """
    return (struct(*cols).apply(list)).alias(f"[{', '.join([Column.getName(c) for c in cols])}]")


def array_contains(col: Column, value: Any) -> Column:
    """
    Return whether value is contained in the array
    """
    return col._simpleUnaryTransformColumn(f"{Column.getName(lit(value))} CONTAINED IN ",
                                           _elementwise_apply(lambda x: value in x))


@wrap_col_args
def array_distinct(col: Column) -> Column:
    """
    Return for every entry in the column (of arrays), remove duplicates
    """
    return col._simpleUnaryTransformColumn(f"ARRAY_DISTINCT ", _elementwise_apply(lambda x: list(set(x))))


@wrap_col_args
def array_min(col: Column) -> Column:
    """
    Return for every entry in the column (of arrays), its minimum
    """
    return col._simpleUnaryTransformColumn(f"ARRAY_MIN ", _elementwise_apply(min_))


@wrap_col_args
def array_max(col: Column) -> Column:
    """
    Return for every entry in the column (of arrays), its maximum
    """
    return col._simpleUnaryTransformColumn(f"ARRAY_MAX ", _elementwise_apply(max_))


@wrap_col_args
def array_sort(col: Column) -> Column:
    """
    Return for every entry in the column (of arrays), the sorted array
    """
    return col._simpleUnaryTransformColumn(f"ARRAY_SORT ", _elementwise_apply(sorted))


def _concat_pd_cols(*pd_cols: pd.Series):
    return pd.concat(pd_cols, axis='column')


@wrap_col_args
def array_union(col1: Column, col2: Column) -> Column:
    """
    Return for corresponding entries in the columns (of arrays), the union of the arrays without duplicates
    """
    def _union_cols(pd_df: pd.DataFrame):
        # TODO: assert lists
        return pd_df.apply(lambda row: set(row[0]).union(row[1]), axis='columns')
    return col1._simpleBinaryTransformColumn(f"ARRAY_UNION ", lambda x, y: _union_cols(_concat_pd_cols(x, y)), col2)


@wrap_col_args
def array_intersect(col1: Column, col2: Column) -> Column:
    """
    Return for corresponding entries in the columns (of arrays), the intersection of the arrays without duplicates
    """
    def _intersect_cols(pd_df: pd.DataFrame):
        # TODO: assert lists
        return pd_df.apply(lambda row: set(row[0]).intersection(row[1]), axis='columns')
    return col1._simpleBinaryTransformColumn(f"ARRAY_INTERSECT ", lambda x, y: _intersect_cols(_concat_pd_cols(x, y)), col2)


@wrap_col_args
def array_except(col1: Column, col2: Column) -> Column:
    """
    Return for corresponding entries in the columns (of arrays), an array of elements contained in the first
     array (col1) and not in the second array (col2). The output is without duplicates.
    """
    def _except_cols(pd_df: pd.DataFrame):
        # TODO: assert lists
        return pd_df.apply(lambda row: set(row[0]).difference(row[1]), axis='columns')
    return col1._simpleBinaryTransformColumn(f"ARRAY_EXCEPT ", lambda x, y: _except_cols(_concat_pd_cols(x, y)), col2)


def array_position(col: Column, value: Any) -> Column:
    """
    First argument is a column of arrays. Return 1-based position of value in the array.
    If not in the array return 0. If either of the arguments is None return None.
    """

    def _find_index(arr, value):
        if (arr is None) or (value is None):
            return None
        pos = arr.index(value) if value in arr else -1
        return pos + 1

    return col._simpleBinaryTransformColumn(f"ARRAY_POSITION ", _find_index,
                                            value, is_other_col=False)


def array_remove(col: Column, value: Any) -> Column:
    """
    First argument is a column of arrays. For every array in column, return array without value.
    """

    def _remove_value(arr, value):
        return list(filter(lambda x: x != value, arr))

    return col._simpleBinaryTransformColumn(f"ARRAY_REMOVE ", _remove_value,
                                            value, is_other_col=False)


def array_join(col: Column, delimiter: str, null_replacement=None) -> Column:
    """
    First argument is a column of arrays. Concatenates the elements of col using the delimiter.
    None values are replaced with null_replacement if set, otherwise they are ignored.
    """

    def _join_arr(arr, delimiter):
        if null_replacement is None:
            new_arr = filter(lambda x: x is not None, arr)
        else:
            new_arr = [null_replacement if x is None else x for x in arr]
        return delimiter.join(new_arr)

    return col._simpleBinaryTransformColumn(f"ARRAY_JOIN ", _join_arr, delimiter, is_other_col=False)


def array_repeat(col: Column, count: int) -> Column:
    """
    creates an array containing a column repeated count times
    """

    def _repeat_arr(val, cnt):
        return [val for _ in range(cnt)]

    return col._simpleBinaryTransformColumn(f"ARRAY_REPEAT ", _repeat_arr, count, is_other_col=False)


def element_at(col: Column, extraction: Hashable) -> Column:
    """
    col is an array column.
    Returns element of array at given index in extraction if col is array.
    Returns value for the given key in extraction if col is a dictionary.
    NOTE: Index in array in 1-based!!
    """

    def _element_at(arr, ext):
        if isinstance(arr, list):
            ext -= 1
            assert ext >= 0, "make sure you use 1-based indexing. only positive values are allowed."
        return arr[ext] if ext in arr else None

    return col._simpleBinaryTransformColumn(f"ELEMENT_AT ", _element_at, extraction, is_other_col=False)


@wrap_col_args
def arrays_overlap(col1: Column, col2: Column) -> Column:
    """
    returns true if the arrays contain any common non-null element;
    if not, returns null if both the arrays are non-empty and any of them contains a null element;
    returns false otherwise.
    """
    def _overlap_cols(pd_df: pd.DataFrame):
        def _overlap_cols_inner(row: pd.Series):
            x, y = row[0], row[1]
            res = set(x).intersection(y)
            if len(res) == 0:
                return False
            if len(res) >= 2:
                return True
            if None in res:
                return None
            return True

        # TODO: assert lists
        return pd_df.apply(_overlap_cols_inner, axis='columns')
    return col1._simpleBinaryTransformColumn(f"ARRAYS_OVERLAP ",
                                             lambda x, y: _overlap_cols(_concat_pd_cols(x, y)), col2)


@wrap_col_args
def arrays_zip(*cols: Column) -> Column:
    """
    Zip array columns
    """
    def _zip_cols(combined_col: tuple):
        return list(zip(*combined_col))
    return (struct(*cols).apply(_zip_cols)
            .alias(f'ARRAYS_ZIP({", ".join([Column.getName(c) for c in cols])})'))


@wrap_col_args
def concat(*cols: Column) -> Column:
    """
    Concatenate strings or arrays
    """
    def _concat_with_assert(xs):
        # TODO: actually need to assert they are the same type
        assert all([isinstance(x, (str, list)) for x in xs]), "concat() can take in either string or list."
        return sum(xs)

    return (struct(*cols).apply(_concat_with_assert)
            .alias(f'CONCAT("{str(sep)}", {", ".join([Column.getName(c) for c in cols])})'))


def concat_ws(sep: str, *cols: Column) -> Column:
    """
    Concatenate string cols with sep as the separator
    """
    return (struct(*cols).apply(sep.join)
            .alias(f'CONCAT_WS("{str(sep)}", {", ".join([Column.getName(c) for c in cols])})'))


@wrap_col_args
def size(col: Column) -> Column:
    """
    Computes length of array or dict
    """
    def _len_with_assert(x):
        assert isinstance(x, (list, tuple, dict)), "cannot apply size() to elements of type other than array, tuple"\
                                                   " or dictionary."
        return len(x)

    return col._simpleUnaryTransformColumn("SIZE ", _elementwise_apply(_len_with_assert))


@wrap_col_args
def signum(col: Column) -> Column:
    """
    Computes the signum of a value
    """
    def _signum(x):
        if x == 0:
            return 0
        return 1 if x > 0 else -1

    return col._simpleUnaryTransformColumn("SIGNUM ", _elementwise_apply(_signum))


@wrap_col_args
def acos(col: Column) -> Column:
    """
    Computes inverse cosine function
    """
    return col._simpleUnaryTransformColumn("ACOS ", _elementwise_apply(math.acos))


@wrap_col_args
def asin(col: Column) -> Column:
    """
    Computes inverse sine function
    """
    return col._simpleUnaryTransformColumn("ASIN ", _elementwise_apply(math.asin))


@wrap_col_args
def atan(col: Column) -> Column:
    """
    Computes inverse tangent function
    """
    return col._simpleUnaryTransformColumn("ATAN ", _elementwise_apply(math.atan))


@wrap_col_args
def atan2(col: Column) -> Column:
    """
    Computes atan2 function
    """
    return col._simpleUnaryTransformColumn("ATAN2 ", _elementwise_apply(math.atan2))


@wrap_col_args
def cosh(col: Column) -> Column:
    """
    Computes hyperbolic cosine function
    """
    return col._simpleUnaryTransformColumn("COSH ", _elementwise_apply(math.cosh))


@wrap_col_args
def acosh(col: Column) -> Column:
    """
    Computes inverse hyperbolic cosine function
    """
    return col._simpleUnaryTransformColumn("ACOSH ", _elementwise_apply(math.acosh))


@wrap_col_args
def sinh(col: Column) -> Column:
    """
    Computes hyperbolic sine function
    """
    return col._simpleUnaryTransformColumn("SINH ", _elementwise_apply(math.sinh))


@wrap_col_args
def asinh(col: Column) -> Column:
    """
    Computes inverse hyperbolic sine function
    """
    return col._simpleUnaryTransformColumn("ASINH ", _elementwise_apply(math.asinh))


@wrap_col_args
def tanh(col: Column) -> Column:
    """
    Computes hyperbolic tangent function
    """
    return col._simpleUnaryTransformColumn("TANH ", _elementwise_apply(math.tanh))


@wrap_col_args
def atanh(col: Column) -> Column:
    """
    Computes inverse hyperbolic tangent function
    """
    return col._simpleUnaryTransformColumn("ATANH ", _elementwise_apply(math.atanh))


@wrap_col_args
def cos(col: Column) -> Column:
    """
    Computes cosine function
    """
    return col._simpleUnaryTransformColumn("COS ", _elementwise_apply(math.cos))


@wrap_col_args
def sin(col: Column) -> Column:
    """
    Computes sine function
    """
    return col._simpleUnaryTransformColumn("SIN ", _elementwise_apply(math.sin))


@wrap_col_args
def tan(col: Column) -> Column:
    """
    Computes tangent function
    """
    return col._simpleUnaryTransformColumn("TAN ", _elementwise_apply(math.tan))


@wrap_col_args
def struct(*cols: Column) -> Column:
    """
    Takes in columns and returns a single struct column
    """
    return udf(lambda *x: tuple(x))(*cols).alias(f"({', '.join([Column.getName(col) for col in cols])})")


def udf(func: Callable) -> Callable:
    """
    Transforms function `func` into a function that applies `func` elementwise on a column
    """

    @wrap_col_args
    def f(*cols: Column) -> Column:
        return Column._transformColumn(f"UDF `{func.__name__}` ({', '.join([Column.getName(col) for col in cols])})",
                                       lambda df: pd.concat([Column._apply(col, df) for col in cols],
                                                            axis='columns').apply(lambda c: func(*c), axis='columns'))
    return f


# aggregate functions

def corr(col1: Union[str, Column], col2: Union[str, Column], method='pearson') -> AggColumn:
    """
    Compute the correlation between col1 and col2
    """

    def _corr_func(xy: pd.Series):
        x, y = xy.apply(lambda x: x[0]), xy.apply(lambda x: x[1])
        return x.corr(y)

    return AggColumn(name="CORR ", orig_col=struct(col1, col2), op=_corr_func)


@wrap_col_args
def _min(col: Column) -> AggColumn:
    """
    Aggregate function: compute minimum
    """
    return AggColumn(name="MIN", orig_col=col, op=lambda x: x.min())


@wrap_col_args
def _max(col: Column) -> AggColumn:
    """
    Aggregate function: compute maximum
    """
    return AggColumn(name="MAX", orig_col=col, op=lambda x: x.max())


@wrap_col_args
def mean(col: Column) -> AggColumn:
    """
    Aggregate function: compute mean
    """
    return AggColumn(name="MEAN", orig_col=col, op=lambda x: x.mean())


@wrap_col_args
def first(col: Column) -> AggColumn:
    """
    Aggregate function: take first entry in the group
    """
    return AggColumn(name="FIRST", orig_col=col, op=lambda x: x.iloc[0])


@wrap_col_args
def last(col: Column) -> AggColumn:
    """
    Aggregate function: take last entry in the group
    """
    return AggColumn(name="LAST", orig_col=col, op=lambda x: x.iloc[-1])


def count(col: Column) -> AggColumn:
    """
    Aggregate function: count the number of elements in the group.
    NOTE: Not affected by the value of `col`!!
    """
    if col != "*":
        warnings.warn("count(col) ignores the column it gets. Use count('*') instead to avoid this message")
    return AggColumn(name="COUNT", orig_col=col, op=len)


@wrap_col_args
def countDistinct(col: Column) -> AggColumn:
    """
    Aggregate function: count the number of distinct elements in `col` for each group
    """
    # TODO: this functions needs to be fixed
    return AggColumn(name="COUNT DISTINCT", orig_col=col, op=lambda x: len(set(x[Column.getName(col)])))


@wrap_col_args
def collect_list(col: Column) -> AggColumn:
    """
    Aggregate function: collect all elements in group into a list
    """
    return AggColumn(name="COLLECT_LIST", orig_col=col, op=list)


@wrap_col_args
def collect_set(col: Column) -> AggColumn:
    """
    Aggregate function: collect all elements in group into a set
    """
    return AggColumn(name="COLLECT_SET", orig_col=col, op=set)


@wrap_col_args
def _sum(col: Column) -> AggColumn:
    """
    Aggregate function: compute sum
    """
    return AggColumn(name="SUM", orig_col=col, op=lambda x: x.sum())


@wrap_col_args
def sumDistinct(col: Column) -> AggColumn:
    """
    Aggregate function: sum distinct values
    """
    return AggColumn(name="SUM DISTINCT", orig_col=col, op=lambda x: sum_(set(x)))


# window-only functions
@wrap_col_args
def lead(col: Column, count: int = 1) -> WindowTransformationColumn:
    """
    Window function: lead function
    """
    return WindowTransformationColumn(name=f"LEAD {count}", orig_col=col,
                                      op=lambda col, grp, pos: col.loc[grp[pos+count]] if pos+count < len(grp) else None)


@wrap_col_args
def lag(col: Column, count: int = 1) -> WindowTransformationColumn:
    """
    Window function: lag function
    """
    return WindowTransformationColumn(name=f"LAG {count}", orig_col=col,
                                      op=lambda col, grp, pos: col.loc[grp[pos-count]] if pos-count >= 0 else None)


def dense_rank() -> WindowTransformationColumn:
    """
    Window function: dense rank function
    """
    return WindowTransformationColumn(name=f"DENSE RANK", orig_col=None,
                                      op=lambda col, grp, pos: pos)


def rank() -> WindowTransformationColumn:
    """
    Window function: rank function
    """
    # TODO: check
    return WindowTransformationColumn(name=f"RANK", orig_col=None,
                                      op=lambda col, grp, pos: list(col.loc[grp]).index(col[grp[pos]]))


min = _min
max = _max
sum = _sum
abs = _abs
