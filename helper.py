from low import Node
import networkx
import matplotlib.pyplot as plt


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