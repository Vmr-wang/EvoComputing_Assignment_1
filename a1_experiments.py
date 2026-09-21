import argparse
import json
from pathlib import Path
import a1_ga as ga
import a1_solution as baseline

METHODS = ("random_search", "random", "aligned")

def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_protocol() -> dict:
    return {
        'question': 'Does connection-path-aligned subtree crossover improve body matching over random subtree crossover at equal evaluation budget?',
        'methods': list(METHODS),
        'seeds': list(range(10)),
        'population_size': 50,
        'generations': 100,
        'evaluation_budget': 50 * (100 + 1),
        'initial_population_in_budget': True,
        'offspring_per_generation': 50,
        'module_range_including_core': [baseline.MIN_MODULES, baseline.MAX_MODULES],
        'initial_size_distribution': 'uniform integer, then official random_tree(size - 1)',
        'tournament_size': ga.TOURNAMENT_SIZE,
        'crossover_probability': ga.CROSSOVER_PROBABILITY,
        'mutation_probability': ga.MUTATION_PROBABILITY,
        'mutation_operator': 'subtree_replacement, max_modules=3',
        'survivor_selection': 'best 50 of parents and 50 offspring; stable sort breaks ties in favour of parents',
        'size_limit_policy': 'revert only the child operation that exceeds the module range',
        'unavailable_cut_policy': 'skip crossover, then continue mutation',
        'fitness': 'mean of five target distances plus their population standard deviation; minimize',
        'evaluation_policy': 'one official fitness call per initial individual or offspring; no score caching or duplicate filtering',
        'cross_run_standard_deviation': 'sample standard deviation (ddof=1)',
        'target_files': [path.name for path in sorted(baseline.TARGET_DIR.glob('*.json'))],
        'target_sizes': [len(target) for target in baseline.load_targets()],
    }


def run_one(method: str, seed: int, protocol: dict, output: Path) -> dict:
    run_dir = output / method / f"seed_{seed:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    targets = baseline.load_targets()
    budget = protocol["evaluation_budget"]
    if method == "random_search":
        best, score, history = baseline.random_search(targets, budget, seed)
        result = {"algorithm": "random_search", "seed": seed, "evaluation_budget": budget,
                  "module_range": [baseline.MIN_MODULES, baseline.MAX_MODULES],
                  "best_fitness": score, "best_genome": best.to_dict(), "history": history}
    else:
        result = ga.run_ga(targets, budget, seed, run_dir / "database.db", method,
                           population_size=protocol["population_size"])
        result["database"] = "database.db"
    save_json(run_dir / "result.json", result)
    return {"method": method, "seed": seed, "best_fitness": result["best_fitness"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=baseline.RESULT_DIR / "run_01")
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = build_protocol()
    save_json(output / "protocol.json", protocol)
    for seed in protocol["seeds"]:
        for method in METHODS:
            row = run_one(method, seed, protocol, output)
            print(f"{method}: seed={seed}, best={row['best_fitness']:.4f}", flush=True)
    print(f"results：{output}")


if __name__ == "__main__":
    main()
