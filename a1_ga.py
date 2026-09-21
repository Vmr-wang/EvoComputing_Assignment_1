import copy
import random
from pathlib import Path
from statistics import mean
import networkx as nx
from ariel.ec import EA, EAOperation, Individual, Population, config
from ariel.ec.genotypes.tree.operators import mutate_subtree_replacement
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from a1_crossover import CrossoverMode, crossover_trees
from a1_solution import MAX_MODULES, MIN_MODULES, fitness, make_random_genome

POPULATION_SIZE = 50
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2

def make_individual(genome: TreeGenome) -> Individual:
    individual = Individual()
    individual.genotype = genome.to_dict()
    return individual


def select_parent(population: Population) -> Individual:
    candidates = population.sample(TOURNAMENT_SIZE)
    return min(candidates, key=lambda individual: individual.fitness)


def make_children(parent_a: TreeGenome, parent_b: TreeGenome,
                  crossover_mode: CrossoverMode = "random") -> list[TreeGenome]:
    if random.random() < CROSSOVER_PROBABILITY:
        children = crossover_trees(parent_a, parent_b, crossover_mode)
    else:
        children = (copy.deepcopy(parent_a), copy.deepcopy(parent_b))
    offspring = []
    for child, parent in zip(children, (parent_a, parent_b), strict=True):
        if not MIN_MODULES <= len(child.nodes) <= MAX_MODULES:
            child = copy.deepcopy(parent)
        if random.random() < MUTATION_PROBABILITY:
            before_mutation = copy.deepcopy(child)
            mutate_subtree_replacement(child, max_modules=3)
            if not MIN_MODULES <= len(child.nodes) <= MAX_MODULES:
                child = before_mutation
        validate_genome_dict(child.to_dict())
        offspring.append(child)
    return offspring


def keep_best(population: Population, population_size: int = POPULATION_SIZE) -> Population:
    ranked = sorted(population.alive, key=lambda individual: individual.fitness)
    for individual in ranked[population_size:]:
        individual.alive = False
    return population


def run_ga(
    targets: list[nx.DiGraph],
    budget: int,
    seed: int,
    db_path: Path,
    crossover_mode: CrossoverMode = "random",
    *,
    population_size: int = POPULATION_SIZE,
) -> dict:
    if population_size < TOURNAMENT_SIZE or population_size % 2:
        raise ValueError("population size must be an even number and must not be less than the tournament size.")
    if budget < population_size or budget % population_size:
        raise ValueError(f"The budget must be at least {population_size} and must be an integer multiple of that value.")
    if crossover_mode not in ("random", "aligned"):
        raise ValueError(f"unknown crossover：{crossover_mode}")
    random.seed(seed)
    evaluations = []
    generations = []
    best_so_far = float("inf")

    def evaluate(population: Population) -> Population:
        nonlocal best_so_far
        for individual in population.unevaluated:
            genome = TreeGenome.from_dict(individual.genotype)
            individual.fitness = fitness(genome, targets)
            best_so_far = min(best_so_far, individual.fitness)
            evaluations.append({
                "evaluation": len(evaluations) + 1,
                "generation": len(generations),
                "modules": len(genome.nodes),
                "fitness": individual.fitness,
                "best_fitness": best_so_far,
            })
        return population

    def reproduce(population: Population) -> Population:
        parents = population.alive
        children = []
        for _ in range(population_size // 2):
            parent_a = select_parent(parents)
            parent_b = select_parent(parents)
            genomes = make_children(
                TreeGenome.from_dict(parent_a.genotype),
                TreeGenome.from_dict(parent_b.genotype),
                crossover_mode,
            )
            for genome in genomes:
                child = make_individual(genome)
                children.append(child)
        population.extend(children)
        return population

    def record_generation(population: Population) -> Population:
        scores = [individual.fitness for individual in population.alive]
        row = {
            "generation": len(generations),
            "evaluations": len(evaluations),
            "population_size": len(scores),
            "best_fitness": min(scores),
            "mean_fitness": mean(scores),
            "worst_fitness": max(scores),
        }
        generations.append(row)
        return population

    initial = Population([
        make_individual(make_random_genome()) for _ in range(population_size)
    ])
    evaluate(initial)
    record_generation(initial)

    operations = [
        EAOperation(reproduce),
        EAOperation(evaluate),
        EAOperation(keep_best, population_size=population_size),
        EAOperation(record_generation),
    ]
    config.target_population_size = population_size
    ea = EA(
        initial,
        operations,
        num_steps=budget // population_size - 1,
        first_generation_id=0,
        is_maximisation=False,
        quiet=True,
        db_file_path=db_path,
        db_handling="delete",
    )
    try:
        ea.run()
        best = ea.get_solution("best", only_alive=False)
        result = {
            "algorithm": f"ga_{crossover_mode}_subtree",
            "crossover_mode": crossover_mode,
            "seed": seed,
            "evaluation_budget": budget,
            "module_range": [MIN_MODULES, MAX_MODULES],
            "population_size": population_size,
            "offspring_per_generation": population_size,
            "tournament_size": TOURNAMENT_SIZE,
            "crossover_probability": CROSSOVER_PROBABILITY,
            "mutation_probability": MUTATION_PROBABILITY,
            "mutation_operator": "subtree_replacement",
            "mutation_max_modules": 3,
            "survivor_selection": "best_of_parents_and_offspring",
            "best_fitness": best.fitness,
            "best_genome": best.genotype,
            "history": evaluations,
            "generations": generations,
            "database": str(db_path),
        }
    finally:
        ea.engine.dispose()
    return result
