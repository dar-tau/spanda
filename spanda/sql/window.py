import numpy as np
import pandas as pd

import spanda.sql.functions as F
from spanda.core.typing import *

class _SpandaWindowType:
    NONE            = 0
    PARTITION_BY    = 1
    ORDER_BY        = 2
    ROWS_BETWEEN    = 3
    RANGE_BETWEEN   = 4


class SpandaWindowSpec:
    """
    A window specification that defines the partitioning, ordering,
    and frame boundaries.

    Use the static methods in :class:`Window` to create a :class:`WindowSpec`.
    """

    def __init__(self, name: str, op: Callable, window_type: Union[Tuple[_SpandaWindowType], _SpandaWindowType],
                 _orderby_col: Optional[F.Column] = None):
        self._name = name
        self._op = op
        self._orderby_col = _orderby_col
        if isinstance(window_type, tuple):
            self._window_types = window_type
        else:
            self._window_types = (window_type,) if window_type != _SpandaWindowType.NONE else tuple()

    def _get_group_data(self, df: pd.DataFrame) -> Tuple[dict]:
        assert len(self._window_types) != 0, "cannot apply window function over empty Window"
        row2grp, grp2rows = self._op(df)
        return row2grp, grp2rows

    def _get_orderby_col(self):
        assert self._orderby_col is not None, "this window operation requires to use .orderBy()"
        return self._orderby_col

    def partitionBy(self, *cols: str) -> 'SpandaWindowSpec':
        """
         Defines the partitioning columns in a :class:`WindowSpec`.
         """

        assert len(self._window_types) == 0, ".partitionBy() must be used only after Window"

        def f(df):
            grp2rows = df.groupby(*cols).groups
            row2grp = {}
            for grp in grp2rows:
                for row in grp2rows[grp]:
                    row2grp[row] = grp
            return row2grp, grp2rows

        window_types = self._window_types + (_SpandaWindowType.PARTITION_BY,)
        return SpandaWindowSpec(self._name + f" PARTITION BY {', '.join([F.Column.getName(col) for col in cols])}",
                                lambda df: f(self._op(df)), window_types)

    def rowsBetween(self, start: int, end: int) -> 'SpandaWindowSpec':
        """
        Defines the frame boundaries, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative positions from the current row.
        For example, "0" means "current row", while "-1" means the row before
        the current row, and "5" means the fifth row after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        :param start: boundary start, inclusive.
                      The frame is unbounded if this is ``Window.unboundedPreceding``.
        :param end: boundary end, inclusive.
                    The frame is unbounded if this is ``Window.unboundedFollowing``.
        """
        assert len(self._window_types) == 0, ".rowsBetween() must be used only after Window"

        def f(df):
            row2grp = {idx: idx for idx in df.index}
            grp2rows = {}
            for i in range(len(df)):
                l = max(0, i+start)
                r = min(len(df), i+end+1)
                rows = df.iloc[l:r].index
                idx = df.iloc[i].name
                grp2rows[idx] = rows
            return row2grp, grp2rows

        window_types = self._window_types + (_SpandaWindowType.ROWS_BETWEEN,)
        return SpandaWindowSpec(self._name + f" ROWS BETWEEN ({start}, {end})", lambda df: f(self._op(df)),
                                window_types)

    def orderBy(self, *cols: str, ascending: bool = True) -> 'SpandaWindowSpec':
        """
        Defines the ordering columns in a :class:`WindowSpec`.
        """

        assert set(self._window_types).issubset({_SpandaWindowType.PARTITION_BY,
                                                 _SpandaWindowType.ROWS_BETWEEN}), \
            ".orderBy() can be used only after .partitionBy() or .rowsBetween()"
        if len(self._window_types) > 0:
            def f(group_data, df):
                row2grp, grp2rows = group_data

                def _idx_to_key(row_idx):
                    row = df.loc[row_idx]
                    return tuple([row[c] for c in cols])
                grp2rows = {grp_name: sorted(rows, key=_idx_to_key, reverse=not ascending)
                            for grp_name, rows in grp2rows.items()}
                return row2grp, grp2rows

            window_types = self._window_types + (_SpandaWindowType.ORDER_BY,)
            return SpandaWindowSpec(self._name + f" ORDER BY ({', '.join([F.Column.getName(col) for col in cols])})",
                                    lambda df: f(self._op(df), df),
                                    window_types, _orderby_col=F.struct(*map(F.col, cols)))
        else:
            raise NotImplementedError

    def rangeBetween(self, start: int, end: int) -> 'SpandaWindowSpec':
        """
        Defines the frame boundaries, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative from the current row. For example,
        "0" means "current row", while "-1" means one off before the current row,
        and "5" means the five off after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        :param start: boundary start, inclusive.
                      The frame is unbounded if this is ``Window.unboundedPreceding``.
        :param end: boundary end, inclusive.
                    The frame is unbounded if this is ``Window.unboundedFollowing``.
        """
        raise NotImplementedError


class Window:
    unboundedPreceding = -np.inf
    currentRow = 0
    unboundedFollowing = np.inf

    @staticmethod
    def _init():
        return SpandaWindowSpec(name="", op=lambda df: df, window_type=_SpandaWindowType.NONE)

    @staticmethod
    def partitionBy(*cols: str) -> SpandaWindowSpec:
        """
        Creates a `WindowSpec` with the partitioning defined.
        """
        return Window._init().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: str) -> SpandaWindowSpec:
        """
        Creates a `WindowSpec` with the ordering defined.
        """
        return Window._init().orderBy(*cols)

    @staticmethod
    def rangeBetween(start: int, end: int) -> SpandaWindowSpec:
        """
        Creates a `WindowSpec` with the frame boundaries defined,
        from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative from the current row. For example,
        "0" means "current row", while "-1" means one off before the current row,
        and "5" means the five off after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        :param start: boundary start, inclusive.
                      The frame is unbounded if this is ``Window.unboundedPreceding``.
        :param end: boundary end, inclusive.
                    The frame is unbounded if this is ``Window.unboundedFollowing``.
        """
        return Window._init().rangeBetween(start, end)

    @staticmethod
    def rowsBetween(start: int, end: int) -> SpandaWindowSpec:
        """
        Creates a `WindowSpec` with the frame boundaries defined,
        from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative positions from the current row.
        For example, "0" means "current row", while "-1" means the row before
        the current row, and "5" means the fifth row after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        :param start: boundary start, inclusive.
                      The frame is unbounded if this is ``Window.unboundedPreceding``
        :param end: boundary end, inclusive.
                    The frame is unbounded if this is ``Window.unboundedFollowing``
        """
        return Window._init().rowsBetween(start, end)
