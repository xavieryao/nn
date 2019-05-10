from __future__ import annotations
from abc import ABC
import numpy as np
from typing import *
import networkx
import matplotlib.pyplot as plt


class Node(ABC):
    op_name: str = None
    index: int = 0

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
        self.grad.show()
        return self.grad.forward()

    def build_grad(self) -> Node:
        if self.grad:
            return self.grad
        child_grads = []
        for c in self.children:
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

    def __str__(self):
        return f"<{self.name}>"

    def show(self, indent: int = 0):
        indentation = "                " * indent
        print(indentation + "<-- " + str(self))
        for c in self.parents:
            c.show(indent + 1)


class PlaceHolder(Node):
    op_name = "PlaceHolder"

    def __init__(self, shape: Tuple):
        self.shape = shape
        super().__init__()

    def op(self):
        return self.value

    def fill_value(self, value: np.ndarray):
        assert value.shape == self.shape
        self.value = value

    def bp(self, wrt: Node, downstream_grad: Node):
        pass


class Constant(Node):
    op_name = "Constant"

    def __init__(self, value: np.ndarray):
        super().__init__()
        self.value = value

    def op(self):
        return self.value

    def bp(self, wrt: Node, downstream_grad: Node):
        pass


class Transpose(Node):
    op_name = "Transpose"

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.link_from(parent)

    def op(self):
        return np.transpose(self.parent.value)

    def bp(self, wrt: Node, downstream_grad: Node):
        return downstream_grad


class Add(Node):
    op_name = "Add"

    def __init__(self, *in_nodes):
        super().__init__()
        self.in_nodes = in_nodes
        for n in in_nodes:
            self.link_from(n)

    def op(self):
        return np.sum(np.stack([x.value for x in self.in_nodes]), 0)

    def bp(self, wrt: Node, downstream_grad: Node):
        return downstream_grad


class MatMul(Node):
    op_name = "MatMul"

    def __init__(self, w, x):
        super().__init__()
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
            return MatMul(self.w, downstream_grad)
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
    input_1 = PlaceHolder((3, 1))
    input_2 = PlaceHolder((3, 1))
    weight_1 = Constant(np.asarray([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ]))
    weight_2 = Constant(np.asarray([
        [2, 1, 3],
        [3, 2, 1],
        [1, 3, 2]
    ]))

    v1 = MatMul(weight_1, input_1)
    v2 = MatMul(weight_2, input_2)
    r = Add(v1, v2)

    # Fill value
    input_1.fill_value(np.array([1, 2, 3]).reshape((3, 1)))
    input_2.fill_value(np.array([3, 2, 1]).reshape((3, 1)))

    # Compute
    result = r.forward()
    grad = weight_1.backward()
    r.show()
    print(result)
    print(grad)
    print('Hello World')
