"""
Created on 10/11/2020

@author: Maurizio Ferrari Dacrema
"""

def isPrime(n):

    i = 2

    # Usually you loop up to sqrt(n)
    while i < n:
        if n % i == 0:
            return False

        i += 1

    return True