import functools
from .typing import *
from ..sql.functions import col


def wrap_col_args(func: Callable) -> Callable:
    @functools.wraps(func)
    def f(*args, **kwargs):
        new_args = [col(a) if isinstance(a, str) else a for a in args]
        df = func(*new_args, **kwargs)
        return df
    return f
