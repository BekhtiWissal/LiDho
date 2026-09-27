# LiDho — built on EPyMARL

This repository contains **LiDho**, our multi-agent reinforcement learning framework, implemented on top of
[EPyMARL](https://github.com/uoe-agents/epymarl) (Extended PyMARL). All EPyMARL algorithms (QMIX, VDN, IQL, COMA,
IA2C, IPPO, MAA2C, MAPPO, MADDPG, PAC, …) and environments remain available and are run exactly as in the original
EPyMARL codebase.

LiDho adds three components that can be configured from the command line:

- **Dynamic horizons** — separate episode horizons for the exploration and exploitation phases.
- **Lipschitz-constrained networks** — the spectral norm of every feedforward weight matrix is kept below a chosen Lipschitz constant.
- **Homeostatic plasticity** — optional activity-dependent gain scaling of the agents' recurrent hidden units.

---

## 1. Installation

### 1.1 Python dependencies

Clone this repository and install the core dependencies:

```sh
pip install -r requirements.txt
```

Then install the supported environments (SMAC, SMACv2, SMAClite, matrix games, LBF, RWARE, PettingZoo, VMAS):

```sh
pip install -r env_requirements.txt
```

### 1.2 StarCraft II and SMAC

SMAC needs a local StarCraft II installation and the SMAC map files. Follow the official
[SMAC installation instructions](https://github.com/oxwhirl/smac#installing-starcraft-ii). In short:

1. **Install SMAC** (already included in `env_requirements.txt`):
   ```sh
   pip install git+https://github.com/oxwhirl/smac.git
   ```
2. **Install StarCraft II (Linux).** Download the Linux build of StarCraft II from
   [Blizzard's s2client-proto repository](https://github.com/Blizzard/s2client-proto#downloads) (SMAC requires
   version >= 3.16.1; the SMAC results use version 4.10). By default SMAC expects the game in `~/StarCraftII/`.
   If you install it somewhere else, set the `SC2PATH` environment variable:
   ```sh
   export SC2PATH=/path/to/StarCraftII
   ```
   On macOS and Windows, install StarCraft II through the Battle.net client.
3. **Install the SMAC maps.** Download
   [`SMAC_Maps.zip`](https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip) and extract it into
   `$SC2PATH/Maps/` (create the `Maps` directory if it does not exist).

For SMACv2, also see the [SMACv2 repository](https://github.com/oxwhirl/smacv2). For more environment-specific
details, see the [EPyMARL README](https://github.com/uoe-agents/epymarl#installation--run-instructions).

---

## 2. Running experiments

All experiments are launched through `src/main.py`, using
[Sacred](https://github.com/IDSIA/sacred)'s `with` syntax to override config values:

```sh
python src/main.py --config=<algorithm> --env-config=<environment> with <key>=<value> ...
```

- `--config` selects an algorithm config from `src/config/algs/` (e.g. `qmix`, `vdn`, `mappo`, `lidho`).
- `--env-config` selects an environment config from `src/config/envs/` (e.g. `sc2`, `gymma`).
- Defaults are in `src/config/default.yaml`. Any value can be overridden after `with`.

### 2.1 Baseline algorithms (EPyMARL)

To run any algorithm on any SMAC map, e.g. QMIX on `3s5z`:

```sh
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name="3s5z"
```

To run any algorithm on matrix games (and on other Gymnasium-registered environments), use the `gymma` environment
config, e.g. QMIX on the Penalty game:

```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:penalty-100-nostate-v0"
```

Replace `qmix` with any other algorithm config and `3s5z` / `env_args.key` with the map or task of your choice. See the
[EPyMARL README](https://github.com/uoe-agents/epymarl) for the full list of environments, keys and options.

### 2.2 LiDho (our framework)

```sh
python src/main.py --config=lidho --env-config=sc2 with env_args.map_name="3s5z" episode_limit_explore=150 episode_limit_exploit=170 lipschitz_constant=2 homeostatic_plasticity=True
```

| Argument | Description | Default |
|---|---|---|
| `env_args.map_name` | SMAC map to train on (e.g. `3m`, `3s5z`, `MMM2`, `corridor`). | `3m` |
| `episode_limit_explore` | Episode horizon (max. steps) during the **exploration** phase. | `150` |
| `episode_limit_exploit` | Episode horizon (max. steps) during the **exploitation** phase. | `150` |
| `episode_limit_exploit_start_t` | Environment timestep at which the exploitation phase starts. | `50000` |
| `lipschitz_constant` | Upper bound on the spectral norm of each feedforward weight matrix. | `6` |
| `use_lipschitz` | Turn the Lipschitz constraint on or off. | `True` |
| `homeostatic_plasticity` | Turn homeostatic plasticity in the agents' RNN on (`True`) or off (`False`). | `False` |

When `homeostatic_plasticity=True`, the following hyperparameters can also be set:

| Argument | Description | Default |
|---|---|---|
| `homeo_target` | Target mean absolute activation of the hidden units. | `0.7` |
| `homeo_lr` | Learning rate of the gain update. | `1e-4` |
| `homeo_beta` | Smoothing factor of the running activation mean. | `1e-3` |
| `homeo_start_t` | Number of agent forward passes before homeostatic scaling starts. | `50000` |

The defaults are in `src/config/algs/lidho.yaml` and `src/config/default.yaml`.

---

## 3. Results

By default, results are logged with Sacred under `results/`. See the
[EPyMARL README](https://github.com/uoe-agents/epymarl#logging) for TensorBoard, Weights & Biases logging, saving and
loading models, and plotting.

---

## Acknowledgements

This code is built on [EPyMARL](https://github.com/uoe-agents/epymarl) and [PyMARL](https://github.com/oxwhirl/pymarl),
and uses [SMAC](https://github.com/oxwhirl/smac). If you use this code, please also cite:

```bibtex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
}

@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philip H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

## License

This project is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`), as is EPyMARL.
