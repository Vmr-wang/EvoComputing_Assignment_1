# EvoComputing_Assignment_1

This repository is for the experiment code implemented by Standard Assignment Group 42 in Evolutionary Computing 2026 in VU Amsterdam.

Course repository: [EvolutionaryComputing2026](https://github.com/AndrzejSzczepura/EvolutionaryComputing2026).

Set up the environment by following the instructions in the course repository.

Then copy all five `a1_*.py` files from this repository into `assignments/assignment_1/` inside the course repository.

To run the experiments:

```bash
uv run python assignments/assignment_1/a1_experiments.py --output assignments/assignment_1/results/run_01
```

Then generate the statistics and figures:

```bash
uv run python assignments/assignment_1/a1_analyze.py assignments/assignment_1/results/run_01
```

**Or** 
You can activate the environment first:

```bash
source .venv/bin/activate
```

then run the experiments and generate the figures:

```bash
python assignments/assignment_1/a1_experiments.py --output assignments/assignment_1/results/run_01
python assignments/assignment_1/a1_analyze.py assignments/assignment_1/results/run_01
```

Always Remember to use a new output directory for each experiment batch, such as `run_02` when `run_01` already exists.
