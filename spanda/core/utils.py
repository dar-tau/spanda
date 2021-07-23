import functools
import spanda.core as _core
from .typing import *
import spanda.sql.functions as F


def wrap_dataframe(func: Callable) -> Callable:
    @functools.wraps(func)
    def f(*args, **kwargs):
        df = func(*args, **kwargs)
        return _core.DataFrameWrapper(df)
    return f


def wrap_col_args(func: Callable) -> Callable:
    @functools.wraps(func)
    def f(*args, **kwargs):
        new_args = [F.col(a) if isinstance(a, str) else a for a in args]
        df = func(*new_args, **kwargs)
        return df
    return f
