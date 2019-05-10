from low import *


def neg(x):
    neg_one = Constant(value=np.asarray([-1.], dtype=np.float))
    return ScalarMul(neg_one, x)


def relu(x):
    zero = Constant(value=np.zeros((1,), dtype=np.float))
    take_max = Max(zero, x)
    return take_max


def softmax(x):
    one = Constant(value=np.asarray([1.]))
    neg_one = Constant(value=np.asarray([-1.]))
    a = add(1, exp(neg(x)))
    return power(one, neg_one)

