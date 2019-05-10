from __future__ import annotations
from abc import ABC
import numpy as np
from typing import *
import networkx
import matplotlib.pyplot as plt


class Node(ABC):
    op_name: str = None
    index: int = 0
    is_param: bool = False
    learnable: bool = False

    def __init__(self, name=None):
        if not name:
            name = f"{self.op_name}_{self.index}"
        self.__class__.index += 1
        self.name: str = name
        self.parents: List[Node] = []
        self.children: List[Node] = []
        self.value: Optional[np.ndarray] = None
        self.grad: Optional[Node] = None

    def op(self):
        pass

    def bp(self, wrt: Node, downstream_grad: Node) -> Node:
        pass

    def forward(self):
        if self.value is not None:
            return self.value
        for p in self.parents:
            p.forward()
        self.value = self.op()
        return self.value

    def backward(self):
        self.build_grad()
        return self.grad.forward()

    def build_grad(self) -> Node:
        if self.grad:
            return self.grad
        child_grads = []

        current_children = self.children.copy()
        for c in current_children:
            child_grad = c.build_grad()
            bp = c.bp(self, child_grad)
            child_grads.append(bp)
        if len(child_grads) > 1:
            self.grad = Add(*child_grads)
        elif len(child_grads) == 1:
            self.grad = child_grads[0]
        else:
            self.grad = Constant(np.ones(shape=self.value.shape))
        return self.grad

    def link_to(self, other):
        self.children.append(other)
        other.parents.append(self)

    def link_from(self, other):
        self.parents.append(other)
        other.children.append(self)

    def reset(self):
        if not self.is_param:
            self.value = None

    def simple_apply_grad(self, lr):
        self.value -= lr * self.grad.value

    def simple_apply_grad_upstream(self, lr):
        self.simple_apply_grad(lr)
        for p in self.parents:
            p.simple_apply_grad_upstream(lr)

    def reset_upstream(self):
        self.reset()
        for p in self.parents:
            p.reset_upstream()

    def __str__(self):
        return f"<{self.name}{' ' + str(self.value.shape) if self.value is not None else ''}>"

    def show(self, indent: int = 0):
        indentation = "                " * indent
        print(indentation + "<-- " + str(self))
        for c in self.parents:
            c.show(indent + 1)


class PlaceHolder(Node):
    op_name = "PlaceHolder"
    is_param = True

    def __init__(self, shape: Tuple, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def op(self):
        return self.value

    def fill_value(self, value: np.ndarray):
        assert value.shape == self.shape
        self.value = value

    def bp(self, wrt: Node, downstream_grad: Node):
        pass


class Constant(Node):
    op_name = "Constant"
    is_param = True

    def __init__(self, value: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def op(self):
        return self.value

    def bp(self, wrt: Node, downstream_grad: Node):
        pass


class Parameter(Node):
    op_name = "Parameter"
    is_param = True
    learnable = True

    def __init__(self, value: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def op(self):
        return self.value

    def bp(self, wrt: Node, downstream_grad: Node):
        pass


class Transpose(Node):
    op_name = "Transpose"

    def __init__(self, parent, **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.link_from(parent)

    def op(self):
        return np.transpose(self.parent.value)

    def bp(self, wrt: Node, downstream_grad: Node):
        return downstream_grad


class Add(Node):
    op_name = "Add"

    def __init__(self, *in_nodes, **kwargs):
        super().__init__(**kwargs)
        self.in_nodes = in_nodes
        for n in in_nodes:
            self.link_from(n)

    def op(self):
        return np.sum(np.stack([x.value for x in self.in_nodes]), 0)

    def bp(self, wrt: Node, downstream_grad: Node):
        return downstream_grad


class ScalarMul(Node):
    op_name = "ScalarMul"

    def __init__(self, k: Node, mat: Node, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.mat = mat
        k.link_to(self)
        mat.link_to(self)

    def op(self):
        return self.k.value.reshape(1,) * self.mat.value

    def bp(self, wrt: Node, downstream_grad: Node) -> Node:
        if wrt == self.mat:
            return ScalarMul(self.k, downstream_grad)
        elif wrt == self.k:
            raise NotImplemented
        else:
            raise ValueError


class MatMul(Node):
    op_name = "MatMul"

    def __init__(self, w, x, **kwargs):
        super().__init__(**kwargs)
        self.w = w
        self.x = x
        self.link_from(w)
        self.link_from(x)

    def op(self):
        return np.dot(self.w.value, self.x.value)

    def bp(self, wrt: Node, downstream_grad: Node):
        if wrt == self.w:
            return MatMul(downstream_grad, Transpose(self.x))
        elif wrt == self.x:
            return MatMul(Transpose(self.w), downstream_grad)
        else:
            raise ValueError


def to_networkx(g, node: Node):
    g.add_node(node)
    for n in node.children:
        if n in g:
            g.add_edge(node, n)
        else:
            g.add_node(n)
            g.add_edge(node, n)
            to_networkx(g, n)
    for n in node.parents:
        if n in g:
            g.add_edge(n, node)
        else:
            g.add_node(n)
            g.add_edge(n, node)
            to_networkx(g, n)


def draw_nx(node):
    g = networkx.DiGraph()
    to_networkx(g, node)
    networkx.draw(g, with_labels=True)
    plt.show()


if __name__ == '__main__':
    # Build graph
    x = PlaceHolder(shape=(10, 1), name='x')
    y_true = PlaceHolder(shape=(1, 1), name='y_true')
    weight = Parameter(np.random.rand(1, 10), name='weight')
    bias = Parameter(np.random.rand(1, 1), name='bias')

    v = MatMul(weight, x, name='wx')
    y_pred = Add(v, bias, name='y_pred')

    # Loss
    minus_one = Constant(np.array([-1]), name='-1')
    minus_y = ScalarMul(minus_one, y_true, name='-y')
    diff = Add(y_pred, minus_y, name='y-y_true')
    loss = MatMul(diff, diff, name='loss')

    loss.show()

    # Train
    real_w = np.random.rand(1, 10)
    real_b = np.random.rand(1, 1)
    for i in range(10000):
        # generate and fill data
        real_x = np.random.rand(10, 1)
        real_y = real_w.dot(real_x) + real_b
        x.fill_value(real_x)
        y_true.fill_value(real_y)

        loss_val = loss.forward()[0][0]
        w_grad = weight.backward()
        b_grad = bias.backward()

        weight.simple_apply_grad_upstream(0.01)
        bias.simple_apply_grad_upstream(0.01)

        loss.reset_upstream()
        weight.grad.reset_upstream()
        bias.grad.reset_upstream()

        print(f"{i}   {loss_val}")