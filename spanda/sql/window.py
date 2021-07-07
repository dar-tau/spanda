from ..core import GroupedDataFrameWrapper


class SpandaWindowSpec:
    def __init__(self, name, op):
        self._name = name
        self._op = op

    def _get_group_data(self, df):
        row2grp, grp2rows = self._op(df)
        return row2grp, grp2rows


class Window:
    @staticmethod
    def partitionBy(*cols):
        def f(df):
            grp2rows = df.groupby(*cols).groups
            row2grp = {}
            for grp in grp2rows:
                for row in grp2rows[grp]:
                    row2grp[row] = grp
            return row2grp, grp2rows
        return SpandaWindowSpec(f"(PARTITION BY {', '.join(cols)})", f)

    @staticmethod
    def orderBy(*cols):
        raise NotImplementedError

    @staticmethod
    def rangeBetween(start, end):
        raise NotImplementedError

    @staticmethod
    def rowsBetween(start, end):
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
        return SpandaWindowSpec(f"(ROWS BETWEEN ({start}, {end}))", f)

