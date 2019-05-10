from low import *


def relu(x):
    zero = Constant(value=np.zeros((1,), dtype=np.float))
    take_max = Max(zero, x)
    return take_max
