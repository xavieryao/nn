from .low import *


def neg(x: Node) -> Node:
    return ScalarMul(neg_one, x)


def relu(x: Node) -> Node:
    take_max = Max(zero, x)
    return take_max


def sigmoid(x: Node) -> Node:
    a = add(one, exp(neg(x)))
    return power(a, neg_one)


def binary_cross_entropy(p: Node, y: Node) -> Node:
    return -(y*log(p) + (one-y)*log(one-p))
