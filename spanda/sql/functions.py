import math


# helper functions
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


# classes
class Column:
    def __init__(self, name):
        self.name = name
        self.op = lambda df: df[name]
    
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
    
    def __neg__(self, other): 
        return self._simpleUnaryTransformColumn('-', lambda x: -x)

    def __invert__(self, other): 
        return self._simpleUnaryTransformColumn('NOT ', lambda x: not x)
