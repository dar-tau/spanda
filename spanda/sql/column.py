import re

from copy import deepcopy
import pandas as pd
from spanda.core.typing import *
from spanda.core.utils import wrap_col_args


class _SpecialSpandaColumn:
    """
    This kind of column is created for special column transformations that do not make sense as standard columns
    because the flexibility allowed by usual columns do not apply to them. Such columns include exploded rows which can
    only come inside a select operation (at least in Spanda). These columns do not allow any further transformation
    other than simple ones such as name aliases. They are "leaf-transformations" that can only appear in a select(...)
    clause or its derivatives (such as withColumn(...)).
    """

    EXPLODE_ROWS_TYPE = 1

    def __init__(self, transformation_type, transformed_col, name):
        self._name = name
        self._transformation_type = transformation_type
        self._transformed_col = transformed_col

    def alias(self, name):
        new_copy = deepcopy(self)
        new_copy._name = name
        return new_copy

    @staticmethod
    def _apply_special(col, df):
        if col._transformation_type == _SpecialSpandaColumn.EXPLODE_ROWS_TYPE:
            return df[col._transformed_col]
        else:
            raise NotImplementedError

    @staticmethod
    def _apply_special_postprocess(df, col_name, trans_type):
            if trans_type == _SpecialSpandaColumn.EXPLODE_ROWS_TYPE:
                return df.explode([col_name])
            else:
                raise NotImplementedError


class SpandaStruct:
    def __init__(self, values: Sequence, names=None):
        assert (names is None) or (len(names) == len(values)), "if field names are specified they should match field" \
                                                               " values in the number of parameters"
        if names is None:
            names = list(range(len(values)))
        self._struct_keys = tuple(names)
        self._struct_vals = tuple(values)

    def __iter__(self):
        def _my_iter():
            for i in range(len(self)):
                yield self[i]
        return _my_iter()

    def __len__(self):
        return len(self._struct_vals)

    def __repr__(self):
        return str(self._struct_vals)

    def __getitem__(self, item):
        assert isinstance(item, int), "SpandaStruct does not take in field names but only ints at the moments"
        return self._struct_vals[item]

    @staticmethod
    def _get_field(struct: 'SpandaStruct', field_name: str):
        assert field_name in struct._struct_keys, f'`{field_name}` is not in struct'
        idx = struct._struct_keys.index(field_name)
        # TODO: need to alert on duplicate keys but we can't disallow them because
        #       we must also support them (for e.g., F.array(..) internal implementation)
        return struct._struct_vals[idx]


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
        self._name = name
        self._op = lambda df: df[name]

    def _set_op(self, op: Callable):
        self._op = op

    @staticmethod
    def getName(col: Union['Column', str]) -> str:
        if isinstance(col, Column):
            return col._name
        else:
            return str(col)

    @staticmethod
    def _apply(col: 'Column', df: pd.DataFrame) -> Union[pd.Series, Any]:
        if isinstance(col, Column):
            return col._op(df)
        elif isinstance(col, _SpecialSpandaColumn):
            assert False, f"cannot apply anything special columns such as {Column.getName(col)}"
        else:
            return col

    @staticmethod
    def _transformColumn(name: str, operation: Callable) -> 'Column':
        col = Column(None)
        col._name = name
        col._op = operation
        return col

    def _simpleBinaryTransformColumn(self, opname: str, opfunc: Callable, other: Union['Column', Any],
                                     is_other_col: bool = True) -> 'Column':
        return Column._transformColumn(f"({Column.getName(self)} {opname} {Column.getName(other)})",
                                       lambda df: opfunc(Column._apply(self, df),
                                                         Column._apply(other, df) if is_other_col else other))

    def _simpleUnaryTransformColumn(self, opname: str, opfunc: Callable) -> 'Column':
        return Column._transformColumn(f"({opname}{Column.getName(self)})", lambda df: opfunc(Column._apply(self, df)))

    def __repr__(self):
        return f"<Column {self._name}>"

    def alias(self, name):
        """
        Return column with new name
        """

        return Column._transformColumn(name, self._op)

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

    def getItem(self, key: Union[int, Hashable]):
        """
        An expression that gets an item at position ordinal out of a list, or gets an item by key out of a dict
        """
        return self._simpleUnaryTransformColumn(f'GET_ITEM [{str(key)}]', lambda x: x[key])

    def isNotNull(self):
        """
        True if the current expression is NOT null.
        """
        return self._simpleUnaryTransformColumn('IS NOT NULL ', lambda x: x is not None)

    def isNull(self):
        """
        True if the current expression is null.
        """
        return self._simpleUnaryTransformColumn('IS NULL ', lambda x: x is None)

    def rlike(self, other: str):
        """
        SQL RLIKE expression (LIKE with Regex). Returns a boolean Column based on a regex match.
        """
        pattern = re.compile(other)
        return self._simpleUnaryTransformColumn(f'RLIKE ({str(pattern)}) ', lambda x: pattern.search(x) is not None)

    def like(self, other: str):
        """
        SQL like expression. Returns a boolean Column based on a SQL LIKE match.
        """
        _special_regex_chars = {
            ch: '\\' + ch
            for ch in '.^$*+?{}[]|()\\'
        }

        def _sql_like_fragment_to_regex_string(fragment):
            # https://codereview.stackexchange.com/a/36864/229677
            safe_fragment = ''.join([
                _special_regex_chars.get(ch, ch)
                for ch in fragment
            ])
            return '^' + safe_fragment.replace('%', '.*?').replace('_', '.') + '$'

        return self.rlike(_sql_like_fragment_to_regex_string(other))

    def apply(self, func: Callable) -> 'Column':
        return udf(func)(self)

    def isin(self, values):
        """
        A boolean expression that is evaluated to true if the value of this
        expression is contained by the evaluated values of the arguments.
        """
        return self._simpleBinaryTransformColumn('IN', lambda x, y: x.isin(y), values, is_other_col=False)

    def substr(self, startPos: int, length: int):
        """
        Return a Column which is a substring of the column
        """
        # TODO: assert is string
        return self._simpleUnaryTransformColumn(f'SUBSTR ({startPos}, {length}) ',
                                                lambda x: x[startPos: startPos + length])

    def startswith(self, other: str):
        """
        True if string begins with `other`
        """
        return self._simpleUnaryTransformColumn(f'STARTSWITH ("{other}") ',
                                                lambda x: x.startswith(other))

    def endswith(self, other: str):
        """
        True if string ends with `other`
        """
        return self._simpleUnaryTransformColumn(f'ENDSWITH ("{other}") ',
                                                lambda x: x.endswith(other))

    def contains(self, other: str):
        """
        True if string contains string `other`
        """
        # TODO: assert string
        return self._simpleUnaryTransformColumn(f'CONTAINS ("{other}") ',
                                                lambda x: other in x)

    def getField(self, name: str) -> 'Column':
        """
        Get the field of a SpandaStruct by the name `name`
        """
        full_name = Column.getName(self) + "." + name
        return self.apply(lambda x: SpandaStruct._get_field(x, name)).alias(full_name)

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
        col._set_op(f)
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
        col._set_op(f)
        return col
