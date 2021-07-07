## functions
def col(name):
    return Column(name)

def lit(value):
    return value

## classes
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

    def _simpleBinaryTransformColumn(self, opname, opfunc, other):
        return Column._transformColumn(
            f"({Column.getName(self)} {opname} {Column.getName(other)})",
            lambda df: opfunc(Column._apply(self, df), Column._apply(other, df))
            )

    def _simpleUnaryTransformColumn(self, opname, opfunc):
        return Column._transformColumn(
            f"({opname}{Column.getName(self)})",
            lambda df: opfunc(Column.getOp(self)(df))
            )

    def __repr__(self):
        return f"<Column {self.name}>"

    def alias(self, name):
        Column._transformColumn(name, self.op)
    
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
