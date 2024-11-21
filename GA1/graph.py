#graph.py
from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from consts import *

class Graph:
    def __init__(self):
        self.graph = _setup_graph()
        self.number_of_nodes = self.graph.number_of_nodes()
        self.adj_matrix = self.get_adj_matrix()
        # Создаем маппинг между текстовыми и числовыми значениями
        self.node_to_index = {node: idx for idx, node in enumerate(self.graph.nodes())}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}

    def show_graph(self, start: str, finish:str):
        pos = nx.spring_layout(self.graph)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')

        node_colors = ['lightblue' if node not in (start, finish) else
                       'green' if node == start else
                       'red' for node in self.graph.nodes()]

        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=16,
            font_weight='bold'
        )

        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels
        )

        plt.axis('off')
        plt.show()

    def show_adj_matrix(self):
        print(f"Your adjacency matrix: \n"
              f" {self.adj_matrix}")

    def get_nodes_list(self)->List[str]:
        return list(self.graph.nodes())

    def get_adj_matrix(self):
        nodes = list(self.graph.nodes())
        n = len(nodes)
        matrix = np.full((n, n), np.inf)  # Заполняем матрицу "бесконечностью"
        np.fill_diagonal(matrix, 0)  # Задаем ноль для расстояний до самого себя
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if self.graph.has_edge(node1, node2):
                    matrix[i][j] = self.graph[node1][node2]['weight']  # Устанавливаем веса для связанных узлов
        return matrix


def _read_input_set(g):
    vertices = input().split(",")
    vertices = [v.strip() for v in vertices if v.strip()]

    for vertex in vertices:
        neighbors_with_bandwidth = input().split(",")

        for neighbor_with_bandwidth in neighbors_with_bandwidth:
            parts = neighbor_with_bandwidth.strip().split(" ")
            if len(parts) != 2:
                print(f"Invalid format for '{neighbor_with_bandwidth}', expected 'neighbor bandwidth'")
                continue

            neighbor, bandwidth = parts[0].lower(), parts[1]
            try:
                bandwidth = int(bandwidth)
                g.add_edge(vertex, neighbor, weight=bandwidth)
            except ValueError:
                print(f"Invalid bandwidth '{bandwidth}' for edge '{vertex}-{neighbor}'")
    return g


def parse_vertices(input_str: str) -> List[str]:
    """Parse and clean vertex names from input string."""
    return [v.strip() for v in input_str.split(",") if v.strip()]


def parse_edge_data(edge_str: str) -> Tuple[str, int]:
    """Parse neighbor and bandwidth from edge string."""
    parts = edge_str.strip().split(" ")
    if len(parts) != 2:
        raise ValueError(f"Invalid format: '{edge_str}', expected 'neighbor bandwidth'")

    neighbor = parts[0].lower()
    try:
        bandwidth = int(parts[1])
        return neighbor, bandwidth
    except ValueError:
        raise ValueError(f"Invalid bandwidth: '{parts[1]}'")


def add_edges_to_graph(g: nx.Graph, vertex: str, edges_input: str) -> None:
    """Add edges with bandwidth to the graph for a given vertex."""
    if not edges_input.strip():
        return

    for edge_str in edges_input.split(","):
        try:
            neighbor, bandwidth = parse_edge_data(edge_str)
            g.add_edge(vertex, neighbor, weight=bandwidth)
        except ValueError as e:
            print(f"Error: {e}")


def read_input_set(g: nx.Graph) -> nx.Graph:
    """Read graph data from input without prompts."""
    vertices = parse_vertices(input())

    for vertex in vertices:
        g.add_edge(vertex, vertex, weight=0)
        edges_input = input()
        add_edges_to_graph(g, vertex, edges_input)

    return g


def run_dialog(g: nx.Graph) -> nx.Graph:
    """Read graph data from input with user prompts."""
    vertices = parse_vertices(
        input("Enter vertices names separated by commas: ")
    )
    for vertex in vertices:
        g.add_edge(vertex, vertex,weight=0)
        edges_input = input()
        add_edges_to_graph(g, vertex, edges_input)

    return g


def _setup_graph() -> nx.Graph:
    """Initialize and setup the graph based on mode."""
    g = nx.Graph()
    return read_input_set(g) if DEBUG else run_dialog(g)

