import pandas as pd
from spanda.core.typing import *
from spanda.core.utils import wrap_col_args


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
