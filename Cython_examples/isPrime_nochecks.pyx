"""
Created on 10/11/2020

@author: Maurizio Ferrari Dacrema
"""

# These can be used to remove checks done at runtime (e.g. null pointers etc). Be careful as they can introduce errors
# For example cdivision performs the C division which can result in undesired integer divisions where
# floats are instead required
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def isPrime(long n):

    # Declare index of for loop
    cdef long i

    i = 2

    # Usually you loop up to sqrt(n)
    while i < n:
        if n % i == 0:
            return False

        i += 1

    return True