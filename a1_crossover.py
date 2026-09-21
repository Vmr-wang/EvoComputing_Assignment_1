import copy
import random
from typing import Literal

from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE
from ariel.ec.genotypes.tree.operators import subtree_swap
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

type CrossoverMode = Literal["random", "aligned"]
type ConnectionPath = tuple[str, ...]


def nodes_by_path(genome: TreeGenome) -> dict[ConnectionPath, int]:
    children = {node: [] for node in genome.nodes}
    for edge in genome.edges:
        children[edge["parent"]].append((edge["face"], edge["child"]))

    paths = {}
    stack = [(IDX_OF_CORE, ())]
    while stack:
        node, path = stack.pop()
        paths[path] = node
        for face, child in children[node]:
            stack.append((child, path + (face,)))
    return paths


def choose_crossover_points(
    parent_a: TreeGenome, parent_b: TreeGenome, mode: CrossoverMode
) -> tuple[int, int] | None:
    if mode == "random":
        nodes_a = [node for node in parent_a.nodes if node != IDX_OF_CORE]
        nodes_b = [node for node in parent_b.nodes if node != IDX_OF_CORE]
        node_a = random.choice(nodes_a) if nodes_a else None
        node_b = random.choice(nodes_b) if nodes_b else None
        if node_a is None or node_b is None:
            return None
        return node_a, node_b

    if mode == "aligned":
        paths_a = nodes_by_path(parent_a)
        paths_b = nodes_by_path(parent_b)
        common_paths = sorted((paths_a.keys() & paths_b.keys()) - {()})
        if not common_paths:
            return None
        path = random.choice(common_paths)
        return paths_a[path], paths_b[path]

    raise ValueError(f"unknown crossover：{mode}")


def crossover_trees(
    parent_a: TreeGenome, parent_b: TreeGenome, mode: CrossoverMode
) -> tuple[TreeGenome, TreeGenome]:
    children = (copy.deepcopy(parent_a), copy.deepcopy(parent_b))
    points = choose_crossover_points(parent_a, parent_b, mode)
    if points is None:
        return children

    subtree_swap(children[0], children[1], *points)
    try:
        for child in children:
            validate_genome_dict(child.to_dict())
    except ValueError:
        return copy.deepcopy(parent_a), copy.deepcopy(parent_b)
    return children
