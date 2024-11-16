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

    def get_path_length(self, path):
        nodes_list = self.get_nodes_list()
        path_indices = [nodes_list.index(node) for node in path]

        total_length = 0
        for i in range(len(path_indices) - 1):
            start = path_indices[i]
            end = path_indices[i + 1]

            # Check if edge exists (not inf)
            if self.adj_matrix[start][end] == float('inf'):
                return float('inf')

            total_length += self.adj_matrix[start][end]

        return total_length

    def get_nodes_list(self)->List[str]:
        return list(self.graph.nodes())

    def get_adj_matrix(self):
        return nx.to_numpy_array(self.graph, weight="weight", nonedge=np.inf)


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
    print(f"Your vertices: {vertices}")

    for vertex in vertices:
        g.add_edge(vertex, vertex)
        edges_input = input(
            f"Enter neighbors for vertex '{vertex}' "
            "(format: neighbor bandwidth), separated by commas: "
        )
        add_edges_to_graph(g, vertex, edges_input)

    return g


def _setup_graph(debug: bool = False) -> nx.Graph:
    """Initialize and setup the graph based on mode."""
    g = nx.Graph()
    return read_input_set(g) if DEBUG else run_dialog(g)

