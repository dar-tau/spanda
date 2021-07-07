from ..sql.functions import Column
import functools

def wrap_dataframe(func):
    @functools.wraps(func)
    def f(*args, **kwargs):
        df = func(*args, **kwargs)
        return DataFrameWrapper(df)
    return f

class DataFrameWrapper:
    def __init__(self, df):
        self._df = df

    @wrap_dataframe
    def filter(self, col):
        df = self._df
        if isinstance(col, Column):
            cond = Column._apply(col, df)
            return df[cond]
        elif isinstance(col, str):
            return df.query(col)
        else:
            raise NotImplementedError

    @wrap_dataframe
    def select(self, *cols):
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

    def groupBy(self, *cols):
        raise NotImplementedError
    
    def groupby(self, *cols):
        return self.groupBy(*cols)

    def toPandas(self):
        return self._df

    def __getitem__(self, name):
        return self.select(name)

    def __repr__(self):
        return self._df.__repr__()

    def _repr_html_(self):
        return self._df._repr_html_()
