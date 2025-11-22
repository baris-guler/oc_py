from random import randint

N = 100

def maybe_none(v):
    return v if randint(0, 1) else None