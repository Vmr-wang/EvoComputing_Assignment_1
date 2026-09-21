import argparse
import json
import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("random_search", "random", "aligned")
LABELS = {"random_search": "Random search", "random": "Random crossover", "aligned": "Path-aligned crossover"}
COLORS = {"random_search": "#666666", "random": "#0072B2", "aligned": "#D55E00"}
STYLES = {"random_search": "--", "random": "-", "aligned": "-."}
MARKERS = {"random_search": "s", "random": "o", "aligned": "^"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_runs(root: Path, protocol: dict) -> dict:
    return {method: [read_json(root / method / f"seed_{seed:02d}" / "result.json")
                     for seed in protocol["seeds"]] for method in METHODS}


def summarize(runs: dict, protocol: dict) -> tuple[dict, dict]:
    population = protocol["population_size"]
    checkpoints = np.arange(population, protocol["evaluation_budget"] + 1, population)
    summary = {"n_seeds": len(protocol["seeds"]), "sd_definition": "sample standard deviation, ddof=1", "methods": {}}
    curves = {"generation": list(range(protocol["generations"] + 1)), "evaluations": checkpoints.tolist(), "methods": {}}
    for method in METHODS:
        values = np.array([run["best_fitness"] for run in runs[method]])
        summary["methods"][method] = {
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "per_seed": [{"seed": run["seed"], "fitness": run["best_fitness"], "modules": len(run["best_genome"]["nodes"])} for run in runs[method]],
        }
        best = np.array([[run["history"][i - 1]["best_fitness"] for i in checkpoints] for run in runs[method]])
        curves["methods"][method] = {"best_mean": best.mean(axis=0).tolist(), "best_sd": best.std(axis=0, ddof=1).tolist()}
        if method != "random_search":
            population_means = np.array([[row["mean_fitness"] for row in run["generations"]] for run in runs[method]])
            curves["methods"][method].update({"population_mean": population_means.mean(axis=0).tolist(), "population_mean_sd": population_means.std(axis=0, ddof=1).tolist()})
    return summary, curves


def plot_results(root: Path, protocol: dict, summary: dict, curves: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "svg.fonttype": "none", "pdf.fonttype": 42})

    def save_figure(fig, name):
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    def style_axis(ax):
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("Fitness (lower is better)")

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8), layout="constrained")
    x = np.array(curves["generation"])
    for panel, ax in enumerate(axes):
        style_axis(ax)
        mean_key, sd_key = ("best_mean", "best_sd") if panel == 0 else ("population_mean", "population_mean_sd")
        for method in METHODS if panel == 0 else ("random", "aligned"):
            data = curves["methods"][method]
            center, spread = np.array(data[mean_key]), np.array(data[sd_key])
            ax.fill_between(x, center - spread, center + spread, color=COLORS[method], alpha=0.13, linewidth=0)
            ax.plot(x, center, label=LABELS[method], color=COLORS[method], linestyle=STYLES[method], linewidth=1.5, marker=MARKERS[method], markersize=3, markevery=20)
        ax.set_xlim(0, protocol["generations"])
        ax.set_xlabel("Generation / matched random-search batch" if panel == 0 else "Generation")
        ax.set_title("(a) Best fitness found" if panel == 0 else "(b) Population mean fitness", loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"{summary['n_seeds']} seeds · {protocol['evaluation_budget']:,} evaluations per run · mean ± 1 SD", fontsize=9)
    save_figure(fig, "convergence")


def select_representative(runs: list[dict]) -> tuple[dict, float]:
    median = float(np.median([run["best_fitness"] for run in runs]))
    distances = [abs(run["best_fitness"] - median) for run in runs]
    nearest = min(distances)
    tied = [run for run, distance in zip(runs, distances, strict=True)
            if np.isclose(distance, nearest, rtol=0, atol=1e-12)]
    return min(tied, key=lambda run: run["seed"]), median


def render_bodies(bodies: list) -> tuple[list[np.ndarray], float]:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mj
    from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
    from ariel.simulation.environments import SimpleFlatWorld

    mj.set_mjcb_control(None)
    scenes = []
    field_of_view = 0.0
    for body in bodies:
        world = SimpleFlatWorld(load_precompiled=False)
        robot = construct_mjspec_from_graph(body)
        world.spawn(robot.spec, position=[0, 0, 0.1], correct_collision_with_floor=True)
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data) 
        camera = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "ortho-cam")
        if camera < 0:
            raise ValueError("needs ortho-cam")

        floor = model.geom_type == mj.mjtGeom.mjGEOM_PLANE
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = [0.97, 0.97, 0.97, 1]
        model.vis.quality.shadowsize = 1024

        rotation = data.cam_xmat[camera].reshape(3, 3)
        projected = (data.geom_xpos[~floor] - data.cam_xpos[camera]) @ rotation
        radii = model.geom_rbound[~floor, None]
        low = (projected[:, :2] - radii).min(axis=0)
        high = (projected[:, :2] + radii).max(axis=0)
        center = (low + high) / 2
        model.cam_pos[camera] += rotation[:, :2] @ center
        field_of_view = max(field_of_view, float((high - low).max()) * 1.15)
        scenes.append((model, data, camera))

    images = []
    for model, data, camera in scenes:
        model.cam_fovy[camera] = field_of_view
        mj.mj_forward(model, data)
        with mj.Renderer(model, width=512, height=512) as renderer:
            renderer.update_scene(data, camera=camera)
            images.append(renderer.render().copy())
    return images, field_of_view


def plot_body_comparison(root: Path, protocol: dict, runs: dict) -> None:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome
    from tree_edit_distance import distances_to_targets

    target_dir = Path(__file__).resolve().parent / "target_bodies"
    target_files = protocol["target_files"]
    targets = [load_graph_from_json(target_dir / name) for name in target_files]
    representatives = {}
    bodies = list(targets)
    for method in METHODS:
        run, median = select_representative(runs[method])
        body = TreeGenome.from_dict(run["best_genome"]).to_networkx()
        bodies.append(body)
        representatives[method] = {
            "seed": run["seed"],
            "fitness": run["best_fitness"],
            "median_final_fitness": median,
            "modules": len(body),
            "result_file": f"{method}/seed_{run['seed']:02d}/result.json",
            "distances_to_targets": list(distances_to_targets(body, targets)),
        }

    images, field_of_view = render_bodies(bodies)
    columns = max(len(targets), len(METHODS))
    fig, axes = plt.subplots(2, columns, figsize=(7.3, 4.3), squeeze=False)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.84, bottom=0.16,
                        wspace=0.10, hspace=0.65)
    for ax in axes.flat:
        ax.set_axis_off()
    for index, (name, body) in enumerate(zip(target_files, targets, strict=True)):
        ax = axes[0, index]
        ax.imshow(images[index])
        ax.set_title(f"{Path(name).stem}\n{len(body)} modules", fontsize=8)
    start = (columns - len(METHODS)) // 2
    for index, method in enumerate(METHODS):
        ax = axes[1, start + index]
        selected = representatives[method]
        ax.imshow(images[len(targets) + index])
        ax.set_title(
            f"{LABELS[method]}\nseed {selected['seed']} | {selected['modules']} modules\n"
            f"fitness = {selected['fitness']:.3f}",
            fontsize=7.5, color=COLORS[method],
        )
    fig.suptitle("Target bodies and representative solutions", fontsize=11, y=0.98)
    fig.text(0.5, 0.92, "Top: target set     Bottom: best body from each representative run",
             ha="center", fontsize=8)
    fig.text(0.5, 0.075,
             f"Representative run: closest to median final fitness across {len(protocol['seeds'])} seeds; ties: lowest seed.\n"
             "Identical view and scale; static pose. Fitness: lower is better.",
             ha="center", va="center", fontsize=7)
    try:
        fig.savefig(root / "body_comparison.pdf", bbox_inches="tight")
        fig.savefig(root / "body_comparison.png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    write_json(root / "body_comparison.json", {
        "selection_rule": "closest to median final best_fitness; ties within 1e-12: lowest seed",
        "target_files": target_files,
        "orthographic_view_height_m": field_of_view,
        "representatives": representatives,
    })



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--skip-bodies", action="store_true", help="仅生成统计与收敛图，跳过需要 OpenGL 的身体渲染")
    args = parser.parse_args()
    root = args.result_dir.resolve()
    protocol = read_json(root / "protocol.json")
    runs = load_runs(root, protocol)
    summary, curves = summarize(runs, protocol)
    write_json(root / "summary.json", summary)
    write_json(root / "curves.json", curves)
    plot_results(root, protocol, summary, curves)
    print(f"mean, std saved：{root}")
    if not args.skip_bodies:
        plot_body_comparison(root, protocol, runs)
        print(f"body comparison saved：{root / 'body_comparison.pdf'}")


if __name__ == "__main__":
    main()
