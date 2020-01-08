import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_matrix():
    with open('data/input.txt', 'r') as f:
        matrix = [[int(num) for num in line.split(' ')] for line in f]
    return matrix


def get_color(graph):
    edges = graph.edges()
    return [graph[u][v]['color'] for u, v in edges]


def get_max_abs_root_of_the_characteristic_equation(matrix):
    roots = np.linalg.eigvals(matrix)
    return max(abs(roots))


def draw_iconic_digraph(matrix):
    G = nx.DiGraph(matrix, name="Iconic digraph")
    pos = nx.circular_layout(G)
    minusEdges = nx.DiGraph()
    plusEdges = nx.DiGraph()
    for edge in G.edges.data():
        weight = edge.__getitem__(2)['weight']
        from_node = edge.__getitem__(0)
        to_node = edge.__getitem__(1)
        color = np.random.rand(3, )
        if weight < 0:
            minusEdges.add_edge(from_node, to_node, color=color, style='dotted')
        else:
            plusEdges.add_edge(from_node, to_node, color=color, style='solid')
    colors = get_color(minusEdges)
    collection = nx.draw_networkx_edges(minusEdges, pos, edge_color=colors)
    for patch in collection:
        patch.set_linestyle('dashdot')
    colors = get_color(plusEdges)
    nx.draw_networkx_edges(plusEdges, pos, edge_color=colors)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)

    # plt.show()


def impulse_processes(matrix):
    cut = [1, 2, 4, 6, 11, 16]
    colors = ['red', 'green', 'blue', 'black', 'gold', 'violet']
    X0 = np.zeros(np.size(matrix, 1))
    X1 = X0.copy()
    X1[12] = 1
    facts = factors_table(matrix, X1, X0, 5)
    facts = facts[cut, :]
    # draw_plot(facts, colors)
    P0 = X1.copy()
    tends = tendentions_table(matrix, P0, 8)
    tends = tends[cut, :]
    draw_plot(tends, colors)
    # print(tends)


def draw_plot(facts, colors):
    for s in range(0, np.size(facts, 0)):
        for s1 in range(0, np.size(facts, 1) - 1):
            plt.plot([s1, s1 + 1], [facts[s, s1], facts[s, s1 + 1]], color=colors[s])


def mat_factors(matrix, X1, X0):
    return np.transpose((np.eye(np.size(matrix, 1)) + matrix) * X1 - matrix * X0)


def mat_tendetion(matrix, P1):
    return matrix * np.transpose(P1)


def factors_table(matrix, X1, X0, num):
    X = np.matrix([X0, X1])
    plt.xlim(0, 5), plt.ylim(-5, 5)
    for s in range(1, num):
        X = np.concatenate((X, mat_factors(matrix, np.transpose(X[s - 1]), np.transpose(X[s]))))
    X = np.transpose(X)
    return X


def tendentions_table(matrix, P0, num):
    X = np.matrix(P0)
    plt.xlim(0, 8), plt.ylim(-25, 25)
    for s in range(0, num):
        X = np.concatenate((X, np.transpose(mat_tendetion(matrix, X[s, :]))))
    X = np.transpose(X)
    return X


def main():
    # changed (16, 6)
    matrix = np.matrix(read_matrix())
    # draw_iconic_digraph(matrix)
    max_root = get_max_abs_root_of_the_characteristic_equation(matrix)
    print('max_root = ', max_root)
    if max_root >= 1:
        print('The model is not resistant to disturbances')
    else:
        print('The model is resistant to disturbances')

    impulse_processes(matrix)
    plt.show()


if __name__ == "__main__":
    main()
