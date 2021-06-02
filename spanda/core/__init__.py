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
        self.df = df

    @wrap_dataframe
    def filter(self, col):
        df = self.df
        if isinstance(col, Column):
            cond = Column._apply(col, df)
            return df[cond]
        elif isinstance(col, str):
            return df.query(col)
        else:
            raise NotImplementedError

    @wrap_dataframe
    def select(self, *cols):
        df = self.df
        col_names = []
        for col in cols:
            if isinstance(col, Column):
                df = df.assign(**{col.name: Column._apply(col, df)})
            elif isinstance(col, str):
                pass
            else: 
                raise NotImplementedError
            col_names.append(col.name)
        return df[col_names]

    def groupBy(self, *cols):
        raise NotImplementedError
    
    def toPandas(self):
        return self.df