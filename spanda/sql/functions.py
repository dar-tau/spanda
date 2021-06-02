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
            return str(name)
    
    @staticmethod
    def getOp(col):
        if isinstance(col, Column):
            return col.op
        else:
            return lambda df: col
    
    @staticmethod
    def _transformColumn(name, operation):
        col = Column(None)
        col.name = name
        col.op = operation
        return col

    def _simpleBinaryTransformColumn(self, opname, opfunc, other):
        return Column._transformColumn(
            f"({Column.getName(self)} {opname} {Column.getName(other)})",
            lambda df: opfunc(Column.getOp(other)(df), Column.getOp(other)(df))
            )

    def _simpleUnaryTransformColumn(self, opname, opfunc):
        return Column._transformColumn(
            f"({opname}{Column.getName(self)})",
            lambda df: opfunc(Column.getOp(self)(df))
            )

    def __repr__(self):
        return f"<Column {self.name}>"

    def __eq__(self, other):
        return _simpleBinaryTransformColumn('==', lambda x, y: x == y)
    
    def __gt__(self, other):
        return _simpleBinaryTransformColumn('>', lambda x, y: x > y)
    
    def __lt__(self, other):
        return _simpleBinaryTransformColumn('<', lambda x, y: x < y)
    
    def __ge__(self, other):
        return _simpleBinaryTransformColumn('>=', lambda x, y: x >= y)
    
    def __le__(self, other):
        return _simpleBinaryTransformColumn('<=', lambda x, y: x <= y)
    
    def __add__(self, other):
        return _simpleBinaryTransformColumn('+', lambda x, y: x + y)
    
    def __mul__(self, other): 
        return _simpleBinaryTransformColumn('*', lambda x, y: x * y)
    
    def __div__(self, other): 
        return _simpleBinaryTransformColumn('/', lambda x, y: x / y)

    def __sub__(self, other): 
        return _simpleBinaryTransformColumn('-', lambda x, y: x - y)
    
    def __neg__(self, other): 
        return _simpleUnaryTransformColumn('-', lambda x: -x)
