import pandas as pd
from ..sql.functions import Column, AggColumn
from spanda.core.typing import *
from .utils import wrap_col_args
import functools


def wrap_dataframe(func: Callable) -> Callable:
    @functools.wraps(func)
    def f(*args, **kwargs):
        df = func(*args, **kwargs)
        return DataFrameWrapper(df)
    return f


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

    @wrap_dataframe
    @wrap_col_args
    def filter(self, col: Column) -> pd.DataFrame:
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

    @wrap_dataframe
    @wrap_col_args
    def select(self, *cols: Column) -> pd.DataFrame:
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
    def join(self, other: 'DataFrameWrapper', on: Union[str, List[str]], how: str = 'inner') -> pd.DataFrame:
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
        group_by = self._df.groupby(*cols)
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
        # TODO CHECK: order is deterministic between keys() and items()
        df_dict = {}
        for key in self._keys:
            assert key not in df_dict, "there are keys with the same name"
            df_dict[key] = list(self._groups.keys())

        for col in cols:
            col_name = AggColumn.getName(col)
            assert col_name not in df_dict, "cannot have duplicate names in aggregated dataframe"
            df_dict[col_name] = []
            for grp_key, grp_idxs in self._groups.items():
                grp = self._df.loc[grp_idxs]
                df_dict[col_name].append(AggColumn._apply(col, grp))
        return pd.DataFrame(df_dict)
