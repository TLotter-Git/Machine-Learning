import time
import numba

def fib_py(n):
    if n <= 1:
        return n
    return fib_py(n-1) + fib_py(n-2)

@numba.njit
def fib_numba(n):
    if n <= 1:
        return n
    return fib_numba(n-1) + fib_numba(n-2)

if __name__ == "__main__":
    n = 50

    '''# Pure Python timing
    start = time.time()
    result_py = fib_py(n)
    end = time.time()
    print(f"Python Fibonacci({n}) = {result_py}, Time: {end - start:.4f} seconds")'''

    # Numba timing (first call includes compilation)
    start = time.time()
    result_numba = fib_numba(n)
    end = time.time()
    print(f"Numba Fibonacci({n}) = {result_numba}, Time: {end - start:.4f} seconds")

    # Numba timing (second call, no compilation)
    start = time.time()
    result_numba = fib_numba(n)
    end = time.time()
    print(f"Numba Fibonacci({n}) [2nd call] = {result_numba}, Time: {end - start:.4f} seconds")