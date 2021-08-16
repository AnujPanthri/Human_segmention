from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool

def timeout(seconds):
    def decorator(function):
        def wrapper(*args, **kwargs):
            pool = ThreadPool(processes=1)
            result = pool.apply_async(function, args=args, kwds=kwargs)
            try:
                return result.get(timeout=seconds)
            except TimeoutError as e:
                return e
        return wrapper
    return decorator

@timeout(6)
def get_ch():
    return int(input("Enter a number:"))

ch = get_ch()
print(ch)
exit
if isinstance(ch, TimeoutError):
    print('\n\nYou took too long!')
else:
    print('you gave ',ch," number")