import math

a = 0.5
b = 1

def uniform(x):
    return x

def diagonal(x):
    return math.sqrt(x)

def diagonalab(x):
    return (-a + math.sqrt(a*a + x*b*b - x*a*a)) / (b - a)

def exp(x):
    return math.log(x * math.e - x + 1)
