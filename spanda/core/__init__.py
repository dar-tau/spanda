import pandas as pd
from ..sql.functions import Column, AggColumn, min as F_min, max as F_max, col
from spanda.core.typing import *
from .utils import wrap_col_args, wrap_dataframe


class DataFrameWrapper:
    """
    DataFrameWrapper takes in a Pandas Dataframe and transforms it to a Spanda dataframe,
    which can be manipulated with (Spark inspired) Spanda functions.

    Example:
            sdf = DataFrameWrapper(df)
            employee_ids = sdf.filter("is_employee").select("id")
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def columns(self):
        return list(self._df.columns)

    @staticmethod
    def _tmp_col_name(cols):
        i = 0
        col_name = '__tmp'
        while col_name in cols:
            col_name = f'__tmp_{i}'
            i += 1
        return col_name

    @wrap_dataframe
    def intersect(self, other: 'DataFrameWrapper'):
        """
        Return intersection of dataframes
        """
        assert set(self.columns) == set(other.columns), "columns must be the same when intersecting dataframes"

        # TODO: CHECK
        return pd.merge(self._df, other._df, on=self.columns, how='inner').drop_duplicates()

    @wrap_dataframe
    def union(self, other: 'DataFrameWrapper'):
        """
        Returns union of dataframes
        """
        assert set(self.columns) == set(other.columns), "columns must be the same when unioning dataframes"
        return pd.concat([self._df, other._df], axis='index').drop_duplicates()

    @wrap_dataframe
    def subtract(self, other: 'DataFrameWrapper'):
        """
        Subtraction of dataframes (as sets of rows)
        """
        assert set(self.columns) == set(other.columns), "columns must be the same when subtracting dataframes"
        tmp_col = DataFrameWrapper._tmp_col_name(set(self.columns).union(other.columns))
        tmp_df = pd.concat([self.withColumn(tmp_col, 'A')._df.drop_duplicates(),
                   other.withColumn(tmp_col, 'B')._df.drop_duplicates()], axis='index')\

        tmp_df = tmp_df.groupby(self.columns).agg({tmp_col: tuple})
        tmp_df = tmp_df[tmp_df[tmp_col] == ('A',)]
        return pd.merge(self._df, tmp_df, on=self.columns, how='inner').drop(tmp_col, axis='columns')

    @wrap_dataframe
    def withColumn(self, name: str, col: Column):
        """
        Returns a new Spanda dataframe with a new column
        """
        return self._df.assign(**{name: Column._apply(col, self._df)})

    @wrap_dataframe
    def distinct(self):
        """
        Return new Spanda dataframe with no duplicate rows
        """
        return self._df.drop_duplicates()

    def drop(self, *cols: str):
        """
        Returns Spanda dataframe without the mentioned columns
        """
        return self.select(*filter(lambda c: c not in cols, self.columns))

    def count(self) -> int:
        """
        Returns the number of rows in dataframe
        """
        return len(self._df)

    @wrap_dataframe
    @wrap_col_args
    def filter(self, col: Column):
        """
        Returns a Spanda dataframe with only the records for which `col` equals True.
        """

        df = self._df
        if isinstance(col, Column):
            cond = Column._apply(col, df)
            return df[cond]
        elif isinstance(col, str):
            return df.query(col)
        else:
            raise NotImplementedError

    def agg(self, *agg_cols: AggColumn):
        """
        Aggregate entire dataframe as one group
        """
        grp_data = GroupedDataFrameWrapper(self._df, tuple(), {0: self._df.index})
        return grp_data.agg(*agg_cols)

    def min(self):
        """
        Compute minimum for each column in the dataframe
        """
        return self.agg(*[F_min(c) for c in self.columns])

    def max(self):
        """
        Compute maximum for each column in the dataframe
        """
        return self.agg(*[F_max(c) for c in self.columns])

    @wrap_dataframe
    def sort(self, *cols: str, ascending: bool = True):
        """
        order the rows by the columns named in cols.
        if ascending is True, it will be ordered in ascending order; otherwise - in descending order
        """
        return self._df.sort_values(list(cols), ascending=ascending)

    def where(self, col: Column):
        """
        Alias for `.filter()`
        """
        return self.filter(col)

    def withColumnRenamed(self, old_name: str, new_name: str):
        return self.withColumn(new_name, col(old_name)).drop(old_name)

    @wrap_dataframe
    def head(self, n :int = 5):
        """
        Return first n rows of dataframe
        """
        return self._df.head(n)

    @wrap_dataframe
    @wrap_col_args
    def select(self, *cols: Column):
        """
        Returns a Spanda dataframe with only the selected columns.
        """

        df = self._df
        col_names = []
        for col in cols:
            if isinstance(col, Column):
                df = df.assign(**{col.name: Column._apply(col, df)})
                col_names.append(col.name)
            elif isinstance(col, str):
                col_names.append(col)
            else:
                raise NotImplementedError

        return df[col_names]

    @wrap_dataframe
    def join(self, other: 'DataFrameWrapper', on: Union[str, List[str]], how: str = 'inner'):
        """
        Joins with another Spanda dataframe.
        `on` is a column name or a list of column names we join by.
        `how` decides which type of join will be used ('inner', 'outer', 'left', 'right', 'cross')
        """

        assert isinstance(other, DataFrameWrapper), "can join only with spanda dataframes"
        assert how in ['inner', 'outer', 'left', 'right', 'cross'], \
            "this join method ('how' parameter) is not supported"

        return pd.merge(self._df, other._df, on=on, how=how)

    def groupBy(self, *cols: str) -> 'GroupedDataFrameWrapper':
        """
        Groups by the column names `cols`
        """
        assert all(map(lambda x: isinstance(x, str), cols)), "only column names are allowed for now"
        group_by = self._df.groupby(list(cols))
        groups = group_by.groups
        return GroupedDataFrameWrapper(df=self._df, key=cols, groups=groups)

    def groupby(self, *cols: str) -> 'GroupedDataFrameWrapper':
        """
        Groups by the column names `cols`
        """
        return self.groupBy(*cols)

    def toPandas(self) -> pd.DataFrame:
        """
        Returns the Pandas dataframe corresponding to this Spanda dataframe
        """
        return self._df

    def __getitem__(self, name: str) -> Column:
        df = self._df
        return Column._transformColumn(name, lambda _: df[name])

    def __repr__(self):
        return self._df.__repr__()

    def _repr_html_(self):
        return self._df._repr_html_()

    def __getattr__(self, name: str):
        """
        After regular attribute access, try looking up the name
        This allows simpler access to columns for interactive use.
        """
        if not name.startswith('_'):
            return self[name]
        return object.__getattribute__(self, name)


class GroupedDataFrameWrapper:
    def __init__(self, df: pd.DataFrame, key: Tuple[str], groups: Dict[Hashable, pd.Index]):
        self._df = df
        self._keys = key
        self._groups = groups

    @wrap_dataframe
    def agg(self, *cols: AggColumn) -> pd.DataFrame:
        """
        Aggregate grouped data by aggregation column specified in `cols`
        """
        # TODO: check no duplicate names before
        # TODO CHECK: order is deterministic between keys() and values()
        df_dict = {}
        for i, key in enumerate(self._keys):
            assert key not in df_dict, "there are keys with the same name"
            df_dict[key] = list(map(lambda x: x[i], self._groups.keys()))

        for col in cols:
            col_name = AggColumn.getName(col)
            assert col_name not in df_dict, "cannot have duplicate names in aggregate dataframe"
            df_dict[col_name] = []
            for grp_idxs in self._groups.values():
                grp = self._df.loc[grp_idxs]
                df_dict[col_name].append(AggColumn._apply(col, grp))
        return pd.DataFrame(df_dict)
