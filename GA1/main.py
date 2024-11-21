#main.py
from genetic_algoritm import *
from graph import *
from utils import *


def ask_for_start_and_finish_nodes(nodes: List[str]) -> Tuple[str, str]:
    def validate_node(node: str) -> bool:
        return node.strip().lower() in [n.lower() for n in nodes]

    input_config = {
        "start": {
            "prompt": "Enter start node: ",
            "validator": validate_node,
            "converter": str.strip,
            "error_message": f"Invalid node. Available nodes: {', '.join(nodes)}"
        },
        "finish": {
            "prompt": "Enter finish node: ",
            "validator": validate_node,
            "converter": str.strip,
            "error_message": f"Invalid node. Available nodes: {', '.join(nodes)}"
        }
    }

    while True:
        try:
            params = run_dialog({}, input_config) if not DEBUG else read_input_set({}, input_config)
            if params["start"] == params["finish"]:
                print("Start and finish nodes cannot be the same")
                continue
            return params["start"], params["finish"]
        except ValueError as e:
            print(f"Error: {e}")


def main():
    graph = Graph()
    start, finish = ask_for_start_and_finish_nodes(graph.get_nodes_list())
    graph.show_graph(start, finish)
    params = ask_for_evolution_machine_params()
    graph.show_adj_matrix()

    # Получаем список доступных узлов, исключая начальный и конечный
    available_genes = graph.get_nodes_list()
    available_genes.remove(start)
    available_genes.remove(finish)

    # Добавляем необходимые параметры
    params.update({
        "genes": available_genes,
        "chromosome_length": graph.number_of_nodes,
        "node_to_index": graph.node_to_index,
        "index_to_node": graph.index_to_node,
        "start": start,
        "finish": finish,
        "fitness_func": fitness_graph(graph.get_adj_matrix(), graph.node_to_index),
        "adj_matrix": graph.get_adj_matrix(),
    })

    evolution_machine = setup_evolution_machine(params)
    best_path = evolution_machine.evolve()
    print("Best Path:", best_path)

if __name__ == '__main__':
    main()

