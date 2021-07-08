import math

# helper functions
import pandas as pd


def _elementwise_apply(f):
    def f_elementwise(x):
        return x.apply(f)
    return f_elementwise


# functions
def col(name):
    return Column(name)


def lit(value):
    return value


# column functions
def sqrt(col):
    return col._simpleUnaryTransformColumn("SQRT ", _elementwise_apply(math.sqrt))


def cos(col):
    return col._simpleUnaryTransformColumn("COS ", _elementwise_apply(math.cos))


def sin(col):
    return col._simpleUnaryTransformColumn("SIN ", _elementwise_apply(math.sin))


def tan(col):
    return col._simpleUnaryTransformColumn("TAN ", _elementwise_apply(math.tan))


# aggregate functions
def _min(col):
    return AggColumn(name="MIN", orig_col=col, op=lambda x: x.min())


def _max(col):
    return AggColumn(name="MAX", orig_col=col, op=lambda x: x.max())


def mean(col):
    return AggColumn(name="MEAN", orig_col=col, op=lambda x: x.mean())


def first(col):
    return AggColumn(name="FIRST", orig_col=col, op=lambda x: x.iloc[0])


def last(col):
    return AggColumn(name="LAST", orig_col=col, op=lambda x: x.iloc[-1])


# window-only functions
def lead(col, count=1):
    raise NotImplementedError


def lag(col, count=1):
    raise NotImplementedError


# classes
class Column:
    def __init__(self, name):
        self.name = name
        self.op = lambda df: df[name]

    def setOp(self, op):
        self.op = op

    @staticmethod
    def getName(col):
        if isinstance(col, Column):
            return col.name
        else:
            return str(col)
    
    @staticmethod
    def _apply(col, df):
        if isinstance(col, Column):
            return col.op(df)
        else:
            return col
    
    @staticmethod
    def _transformColumn(name, operation):
        col = Column(None)
        col.name = name
        col.op = operation
        return col

    def _simpleBinaryTransformColumn(self, opname, opfunc, other, is_other_col=True):
        return Column._transformColumn(f"({Column.getName(self)} {opname} {Column.getName(other)})",
                                       lambda df: opfunc(Column._apply(self, df),
                                                         Column._apply(other, df) if is_other_col else other))

    def _simpleUnaryTransformColumn(self, opname, opfunc):
        return Column._transformColumn(f"({opname}{Column.getName(self)})", lambda df: opfunc(Column._apply(self, df)))

    def __repr__(self):
        return f"<Column {self.name}>"

    def alias(self, name):
        return Column._transformColumn(name, self.op)
    
    def isin(self, values):
        return self._simpleBinaryTransformColumn('IN', lambda x, y: x.isin(y), values, is_other_col=False)

    # operators
    def __eq__(self, other):
        return self._simpleBinaryTransformColumn('==', lambda x, y: x == y, other)
    
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
        return self._simpleBinaryTransformColumn(' AND ', lambda x, y: x and y, other)

    def __or__(self, other):
        return self._simpleBinaryTransformColumn(' OR ', lambda x, y: x or y, other)

    def __neg__(self, other): 
        return self._simpleUnaryTransformColumn('-', lambda x: -x)

    def __invert__(self, other): 
        return self._simpleUnaryTransformColumn('NOT ', lambda x: not x)


class AggColumn:
    def __init__(self, name, orig_col, op):
        self.name = name
        self.orig_col = orig_col
        self.op = op

    def alias(self, name):
        # TODO: might want to clone orig_col and op (maybe also in Column)
        return AggColumn(name=name, orig_col=self.orig_col, op=self.op)

    @staticmethod
    def getName(agg_col):
        return f"{agg_col.name} ({Column.getName(agg_col.orig_col)})"

    @staticmethod
    def _apply(agg_col, df):
        return agg_col.op(Column._apply(agg_col.orig_col, df))

    def over(self, window_spec):
        def f(df):
            inputs = Column._apply(self.orig_col, df)
            # in the following, row2grp represents the representative group of each row, while grp2rows is a dictionary
            # of all rows that the group aggregates over. possibly not intuitive but group may include rows that are
            # not represented by this group (for example if we apply lead(...) we aggregate over the next row which is
            # not represented by the current group)
            row2grp, grp2rows = window_spec._get_group_data(df)
            print((row2grp, grp2rows))
            grp_agg_val = {grp: self.op(inputs.loc[vals]) for grp, vals in grp2rows.items()}
            data = {row_idx: grp_agg_val[grp_name] for row_idx, grp_name in row2grp.items()}
            return pd.Series(data=data, index=inputs.index)
        col = Column(name=f"{AggColumn.getName(self)} OVER {window_spec._name}")
        col.setOp(f)
        return col


min = _min
max = _max
