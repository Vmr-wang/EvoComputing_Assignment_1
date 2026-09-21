import random
from pathlib import Path

import networkx as nx

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from tree_edit_distance import mean_plus_std_tree_edit_distance

MIN_MODULES = 1
MAX_MODULES = 20
TARGET_DIR = Path(__file__).resolve().parent / "target_bodies"
RESULT_DIR = Path(__file__).resolve().parent / "results"


def load_targets() -> list[nx.DiGraph]:
    paths = sorted(TARGET_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no target body：{TARGET_DIR}")
    return [load_graph_from_json(path) for path in paths]


def fitness(genome: TreeGenome, targets: list[nx.DiGraph]) -> float:
    return mean_plus_std_tree_edit_distance(genome.to_networkx(), targets)


def make_random_genome() -> TreeGenome:
    num_modules = random.randint(MIN_MODULES, MAX_MODULES)
    genome = random_tree(max_modules=num_modules - 1)
    validate_genome_dict(genome.to_dict())
    return genome


def random_search(
    targets: list[nx.DiGraph], budget: int, seed: int
) -> tuple[TreeGenome, float, list[dict[str, int | float]]]:
    if budget < 1:
        raise ValueError("evaluation budget is 1 at least")
    random.seed(seed)
    best_genome = None
    best_score = float("inf")
    history = []

    for evaluation in range(1, budget + 1):
        genome = make_random_genome()
        score = fitness(genome, targets)

        if score < best_score:
            best_genome = genome
            best_score = score

        history.append({
            "evaluation": evaluation,
            "modules": len(genome.nodes),
            "fitness": score,
            "best_fitness": best_score,
        })

    assert best_genome is not None
    return best_genome, best_score, history
