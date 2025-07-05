import functools
import pandas as pd

def to_parquet(filename):
    def decorator_to_parquet(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            result.to_parquet(filename)

            return result

        return wrapper

    return decorator_to_parquet