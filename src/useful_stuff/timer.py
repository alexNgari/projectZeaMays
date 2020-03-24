"""
Decorator to time functions.
"""

from time import time
from functools import wraps

def time_this(function):
    """
    Wrap your function with @time_this to print the time it took for it to run.
    """
    @wraps(function)
    def wrap(*args, **kwargs):
        start_time = time()
        result = function(*args, **kwargs)
        end_time = time()
        print(f'{function.__name__}({args}) took {end_time-start_time:.2f} seconds')
        return result
    return wrap
