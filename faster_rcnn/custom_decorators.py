import timeit
from functools import wraps

_depth = 0
_stack = []


def profile(func):
    """
    Decorates a function to measure its the performance. Time measurements are printed in reverse order due to the
    stack, need a tree instead to fix this. Should only be used in single threaded contexts.
    :param func: function whose performance to measure.
    :return: wrapper around the same function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _depth

        _depth += 1
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        total_time = timeit.default_timer() - start_time
        _depth -= 1

        indent = '    ' * _depth
        _stack.append("{}Executed function {} in {} seconds".format(indent, func.__name__, total_time))

        if _depth == 0:
            while _stack:
                print(_stack.pop())
        return result

    return wrapper
