#genetic_algorithm.py
import random

import numpy as np

from consts import *
from utils import *


class EvolutionMachine:
    def __init__(self, params: Dict):
        self.available_genes = params['genes']
        self.node_to_index = params['node_to_index']
        self.index_to_node = params['index_to_node']
        self.start_node = params['start']
        self.finish_node = params['finish']
        self.evaluate_fitness = params["fitness_func"]
        self.population_size = params["population_size"]
        self.chromosome_length = params["chromosome_length"]
        self.mutation_rate = params["mutation_probability"]
        self.number_of_generations = params["number_of_generations"]
        self.evolving_mode = params["evolving_mode"]
        self.adj_matrix = params["adj_matrix"]

        match params["crossover_variant"]:
            case "single": self.crossover = self.crossover_single

        self.population = self.initialize_population()

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            # Выбираем два случайных индекса для обмена, исключая начало и конец
            i = random.randint(1, len(chromosome) - 2)
            j = random.randint(1, len(chromosome) - 2)

            # Проверяем возможность обмена
            prev_i = chromosome[i - 1]
            next_i = chromosome[i + 1]
            prev_j = chromosome[j - 1]
            next_j = chromosome[j + 1]

            # Проверяем, что после обмена путь останется валидным
            if (not np.isinf(self.adj_matrix[self.node_to_index[prev_i]][self.node_to_index[chromosome[j]]]) and
                    not np.isinf(self.adj_matrix[self.node_to_index[chromosome[j]]][self.node_to_index[next_i]]) and
                    not np.isinf(self.adj_matrix[self.node_to_index[prev_j]][self.node_to_index[chromosome[i]]]) and
                    not np.isinf(self.adj_matrix[self.node_to_index[chromosome[i]]][self.node_to_index[next_j]])):
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    def crossover_single(self, parent1, parent2):
        # Выбираем точку разреза, исключая начальную и конечную точки
        point = random.randint(1, len(parent1) - 2)

        # Создаем начало пути из первого родителя
        child = parent1[:point]

        # Добавляем гены из второго родителя, проверяя возможность пути
        current = child[-1]
        remaining = [gene for gene in parent2 if gene not in child]

        for gene in remaining:
            # Проверяем существование пути между текущей и следующей вершиной
            if not np.isinf(self.adj_matrix[self.node_to_index[current]][self.node_to_index[gene]]):
                child.append(gene)
                current = gene

        # Добавляем конечную точку, если возможно
        if not np.isinf(self.adj_matrix[self.node_to_index[current]][self.node_to_index[self.finish_node]]):
            child.append(self.finish_node)

        # Если путь неполный или невозможен, генерируем новый
        if len(child) < 2 or child[-1] != self.finish_node:
            return self.initialize_valid_path()

        return child

    def initialize_valid_path(self):
        """Вспомогательный метод для генерации валидного пути"""
        path = [self.start_node]
        current = self.start_node

        while current != self.finish_node:
            # Получаем возможные следующие вершины
            next_nodes = [n for n in self.available_genes + [self.finish_node]
                          if not np.isinf(self.adj_matrix[self.node_to_index[current]]
                                          [self.node_to_index[n]])
                          and n not in path]

            if not next_nodes:
                if not np.isinf(self.adj_matrix[self.node_to_index[current]]
                                [self.node_to_index[self.finish_node]]):
                    path.append(self.finish_node)
                break

            current = random.choice(next_nodes)
            path.append(current)

        return path

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Начинаем с начальной и конечной точки
            path = [self.start_node]
            current = self.start_node

            # Добавляем промежуточные узлы
            while current != self.finish_node:
                # Получаем возможные следующие узлы
                next_nodes = [n for n in self.available_genes + [self.finish_node]
                              if not np.isinf(self.adj_matrix[self.node_to_index[current]]
                                              [self.node_to_index[n]])
                              and n not in path]  # Исключаем уже посещенные узлы

                if not next_nodes:
                    # Если нет доступных узлов, добавляем конечную точку если возможно
                    if not np.isinf(self.adj_matrix[self.node_to_index[current]]
                                    [self.node_to_index[self.finish_node]]):
                        path.append(self.finish_node)
                    break

                # Выбираем случайный следующий узел
                next_node = random.choice(next_nodes)
                path.append(next_node)
                current = next_node

            # Если путь не заканчивается конечной точкой, начинаем заново
            if path[-1] != self.finish_node:
                continue

            population.append(path)
            if len(population) == self.population_size:
                break

        return population

    def convert_path_to_indices(self, path):
        return [self.node_to_index[node] for node in path]

    def convert_indices_to_path(self, indices):
        return [self.index_to_node[idx] for idx in indices]

    def select_parents(self, fitnesses):
        # Преобразуем значения fitness для максимизации
        valid_fitnesses = [1 / (f + 1e-10) if not np.isinf(f) else 0 for f in fitnesses]
        total_fitness = sum(valid_fitnesses)

        if total_fitness == 0:
            # Если нет валидных путей, создаем новые
            return [self.initialize_population()[0] for _ in range(2)]

        probabilities = [f / total_fitness for f in valid_fitnesses]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def evolve(self):
        """Main evolution loop."""
        best_solution = None
        best_fitness = float('inf')

        for generation in range(self.number_of_generations):
            fitnesses = [self.evaluate_fitness(chromo) for chromo in self.population]
            next_generation = []

            min_fitness_idx = np.argmin(fitnesses)
            if fitnesses[min_fitness_idx] < best_fitness:
                best_fitness = fitnesses[min_fitness_idx]
                best_solution = self.population[min_fitness_idx]

            if self.evolving_mode == "step-by-step":
                print(f"Generation {generation}")
                print("Population and Fitnesses:", list(zip(self.population, fitnesses)))

            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(fitnesses)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.extend([child1, child2])

            population = sorted(next_generation, key=lambda chromo: self.evaluate_fitness(chromo))[:self.population_size]

            if self.evolving_mode == "cyclic":
                best_fitness = self.evaluate_fitness(population[0])
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

        return best_solution if best_solution else self.population[0]


def fitness_graph(graph, node_to_index) -> Callable:
    def fitness(chromosome):
        fitness_value = 0
        for i in range(len(chromosome) - 1):
            current = node_to_index[chromosome[i]]
            next_node = node_to_index[chromosome[i + 1]]
            cost = graph[current][next_node]
            if np.isinf(cost):
                return float('inf')
            fitness_value += cost
        return fitness_value if fitness_value > 0 else float('inf')
    return fitness

def setup_evolution_machine(params: Dict)-> EvolutionMachine:
    return EvolutionMachine(params)

def ask_for_evolution_machine_params():
    input_configs = {
        "population_size": {
            "prompt": "Enter population size: ",
            "validator": lambda x: x > 0,
            "converter": int,
            "error_message": "Population size must be greater than 0."
        },
        "number_of_generations": {
            "prompt": "Enter number of generations: ",
            "validator": lambda x: x > 0,
            "converter": int,
            "error_message": "Number of generations must be greater than 0."
        },
        "crossover_variant": {
            "prompt": "Enter crossover variant (e.g., 'single'): ",
            "validator": lambda x: x.lower() in ['single'],
            "converter": str.lower,
            "error_message": "Invalid crossover variant. Please choose either 'single'."
        },
        "mutation_probability": {
            "prompt": "Enter mutation probability: ",
            "validator": lambda x: 0 <= x <= 1,
            "converter": float,
            "error_message": "Mutation probability must be between 0 and 1."
        },
        "evolving_mode": {
            "prompt": "Enter evolving mode (e.g., 'step-by-step' or 'cyclic'): ",
            "validator": lambda x: x.lower() in ['step-by-step', 'cyclic'],
            "converter": str,
            "error_message": "Invalid evolving mode. Please choose either 'step-by-step' or 'cyclic'."
        }
    }
    params = {}
    if DEBUG:
        params = read_input_set(params, input_configs)
    else:
        params = run_dialog(params, input_configs)

    return params