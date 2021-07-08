import numpy as np


class _SpandaWindowType:
    NONE            = 0
    PARTITION_BY    = 1
    ORDER_BY        = 2
    ROWS_BETWEEN    = 3
    RANGE_BETWEEN   = 4


class SpandaWindowSpec:
    def __init__(self, name, op, window_type):
        self._name = name
        self._op = op
        if isinstance(window_type, tuple):
            self._window_types = window_type
        else:
            self._window_types = (window_type,) if window_type != _SpandaWindowType.NONE else tuple()

    def _get_group_data(self, df):
        assert len(self._window_types) != 0, "cannot apply window function over empty Window"
        row2grp, grp2rows = self._op(df)
        return row2grp, grp2rows

    def partitionBy(self, *cols):
        assert len(self._window_types) == 0, ".partitionBy() must be used only after Window"

        def f(df):
            grp2rows = df.groupby(*cols).groups
            row2grp = {}
            for grp in grp2rows:
                for row in grp2rows[grp]:
                    row2grp[row] = grp
            return row2grp, grp2rows

        window_types = self._window_types + (_SpandaWindowType.PARTITION_BY,)
        return SpandaWindowSpec(self._name + f" PARTITION BY {', '.join(cols)}", lambda df: f(self._op(df)),
                                window_types)

    def rowsBetween(self, start, end):
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

    def orderBy(self, *cols, ascending=True):
        assert set(self._window_types).issubset({_SpandaWindowType.PARTITION_BY,
                                                 _SpandaWindowType.ROWS_BETWEEN}), \
            ".orderBy() must be used only after .partitionBy() or .rowsBetween()"
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
            return SpandaWindowSpec(self._name + f" ORDER BY ({', '.join(cols)})", lambda df: f(self._op(df), df),
                                    window_types)
        else:
            raise NotImplementedError


class Window:
    unboundedPreceding = -np.inf
    currentRow = 0
    unboundedFollowing = np.inf

    @staticmethod
    def _init():
        return SpandaWindowSpec(name="", op=lambda df: df, window_type=_SpandaWindowType.NONE)

    @staticmethod
    def partitionBy(*cols):
        return Window._init().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols):
        return Window._init().orderBy(*cols)

    @staticmethod
    def rangeBetween(start, end):
        return Window._init().rangeBetween(start, end)

    @staticmethod
    def rowsBetween(start, end):
        return Window._init().rowsBetween(start, end)
