import math
import random
from typing import List, Tuple

from consts import *
from utils import *


class EvolutionMachine:
    def __init__(self, params: Dict):
        self.available_genes = params['genes']
        self.fitness = params["fitness_func"]
        self.population_size = params["population_size"]
        self.chromosome_length = params["chromosome_length"]
        self.mutation_rate = params["mutation_probability"]
        self.number_of_generations = params["number_of_generations"]
        self.evolving_mode = params["evolving_mode"]

        match params["crossover_variant"]:
            case "single": self.crossover = self.crossover_single

        self.population = self.initialize_population()

    def generate_chromosome(self) -> List[str]:
        """
        Generate a chromosome using only available genes with equal slots
        """
        # Randomly choose how many different genes to use
        genes_count_in_chromosome = random.randint(1, self.chromosome_length)
        slots_count = genes_count_in_chromosome

        # Select random genes from available ones
        selected_genes = random.sample(self.available_genes, slots_count)

        chromosome = []
        for gene in selected_genes:
            slot = [gene for _ in range(math.floor(self.chromosome_length / slots_count))]
            chromosome.extend(slot)

        while len(chromosome) < self.chromosome_length:
            chromosome.append(chromosome[-1])
        return chromosome

    def crossover_single(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """Perform single-point crossover."""
        crossover_point = random.randint(1, self.chromosome_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, chromosome: List[str]) -> List[str]:
        """Mutate chromosome by randomly replacing genes with available ones."""
        return [random.choice(self.available_genes)
                if random.random() < self.mutation_rate
                else gene
                for gene in chromosome]

    def selection(self) -> Tuple[List[str], List[str]]:
        """
        Tournament selection implementation that returns two parents.
        Uses fitness scores to select the best individuals from random tournaments.
        """

        def tournament_select() -> List[str]:
            # Select random candidates for tournament
            tournament_size = 5  # Standard tournament size
            tournament_candidates = random.sample(self.population, tournament_size)

            # Find the best candidate based on fitness
            best_candidate = min(tournament_candidates,
                                 key=lambda chromosome: self.fitness(chromosome))
            return best_candidate

        # Select two parents using separate tournaments
        parent1 = tournament_select()
        parent2 = tournament_select()

        # Ensure parents are different
        while parent2 == parent1:
            parent2 = tournament_select()

        return parent1, parent2


    def initialize_population(self) -> List[List[str]]:
        """Initialize population with valid chromosomes."""
        return [self.generate_chromosome() for _ in range(self.population_size)]

    def evolve(self):
        """Main evolution loop."""
        for generation in range(self.number_of_generations):
            new_population = []

            if self.evolving_mode == "step-by-step":
                print(f"Generation {generation + 1}:")

            # Create new population
            for _ in range(self.population_size // 2):
                # Selection
                parent1, parent2 = self.selection()
                if self.evolving_mode == "step-by-step":
                    print("  Selection:")
                    print(f"    Parent 1: {parent1}")
                    print(f"    Parent 2: {parent2}")

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                if self.evolving_mode == "step-by-step":
                    print("  Crossover:")
                    print(f"    Child 1: {child1}")
                    print(f"    Child 2: {child2}")

                # Mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                if self.evolving_mode == "step-by-step":
                    print("  Mutation:")
                    print(f"    Child 1: {child1}")
                    print(f"    Child 2: {child2}")

                new_population.extend([child1, child2])

            self.population = new_population

            # Print statistics for cyclic mode
            if self.evolving_mode == "cyclic":
                fitness_values = [self.fitness(individual) for individual in self.population]
                avg_fitness = sum(fitness_values) / len(fitness_values)
                best_fitness = max(fitness_values)
                print(f"Generation {generation + 1} - Avg Fitness: {avg_fitness:.2f}, Best Fitness: {best_fitness}")

        print("\nFinal population:")
        for chromosome in self.population:
            print(chromosome)

def fitness_graph(graph, start, finish) -> Callable:
    """Create fitness function for path finding."""
    def fitness(chromosome):
        path_length = graph.get_path_length(list(start) + chromosome + list(finish))
        return path_length if path_length != float('inf') else float('inf')
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