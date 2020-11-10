"""
Created on 10/11/2020

@author: Maurizio Ferrari Dacrema
"""

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