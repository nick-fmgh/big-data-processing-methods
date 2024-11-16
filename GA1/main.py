from genetic_algoritm import *
from graph import *
from utils import *

def ask_for_start_and_finish_nodes(nodes: List[str])-> (int, int):
    input_config = {
        "start": {
            "prompt": "Enter start node: ",
            "validator": lambda node: node in nodes,
            "converter": str.lower,
            "error_message": "Invalid node",
        },
        "finish": {
            "prompt": "Enter finish node: ",
            "validator": lambda node: node in nodes,
            "converter": str.lower,
            "error_message": "Invalid node",
        }
    }
    params = {}
    if DEBUG:
        params = read_input_set(params, input_config)
    else:
        params = run_dialog(params, input_config)

    return params["start"], params["finish"]




def main():
    graph = Graph()
    start, finish = ask_for_start_and_finish_nodes(graph.get_nodes_list())
    graph.show_graph(start, finish)
    params = ask_for_evolution_machine_params()
    graph.show_adj_matrix()
    available_genes = graph.get_nodes_list()
    available_genes.remove(start)
    available_genes.remove(finish)
    if DEBUG: print(available_genes)
    params["genes"] = available_genes
    params["chromosome_length"] = graph.number_of_nodes - 2
    params["fitness_func"] = fitness_graph(graph, start, finish)

    evolution_machine = setup_evolution_machine(params)
    evolution_machine.evolve()

if __name__ == '__main__':
    main()

