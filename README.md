# Behavior Graphs

Code for **Behavior Graphs: Unsupervised Graph Structures for Interpreting
Robot Deployment Data**.

[Paper](https://arxiv.org/abs/2611.00405) · [Website](https://behavior-graphs.com)

A Behavior Graph compresses a policy's rollouts into one interpretable graph:
the policy's own embeddings are windowed, reduced and clustered into behaviors,
and the transitions between behaviors and outcomes form the graph.

## Setup

```bash
git clone --recurse-submodules https://github.com/erikbr01/behavior_graphs.git
cd behavior_graphs
uv sync --all-extras
pytest
```

The submodules in `third_party/` are only needed to train policies and collect
rollouts; each has its own environment, described in [docs/envs.md](docs/envs.md).

## Usage

```bash
# train a policy and collect rollouts with per-timestep policy embeddings
python scripts/train_policy.py env=robomimic_lift_lowdim env.dataset_path=/path/to/low_dim.hdf5
python scripts/collect_rollouts.py env=robomimic_lift_lowdim env.dataset_path=/path/to/low_dim.hdf5 \
    checkpoint=runs/train/.../checkpoints/latest.ckpt num_episodes=50

# build the Behavior Graph
python -m behavior_graphs.pipeline.run rollout_dir=runs/rollouts/<...>

# inspect it
streamlit run apps/demo/Home.py
```

See [docs/data.md](docs/data.md) for the data layout, [docs/experiments.md](docs/experiments.md)
for the experiments, and [REPRODUCE.md](REPRODUCE.md) to reproduce the paper's results.

## Citation

```bibtex
@article{bauer2026behaviorgraphs,
  title   = {Behavior Graphs: Unsupervised Graph Structures for Interpreting
             Robot Deployment Data},
  author  = {Bauer, Erik and Thumm, Jakob and Agia, Christopher and Sinha, Rohan
             and Xu, Chen and Nishimura, Haruki and Bohg, Jeannette
             and Pavone, Marco and Itkina, Masha},
  journal = {arXiv preprint arXiv:2611.00405},
  year    = {2026},
}
```

## License

MIT, see [LICENSE](LICENSE). The submodules in `third_party/` carry their own licenses.
