# Experiments

This directory contains a tool for *nix platforms to perform imperceptible pertubation experiments.

Before running the tool, the dependencies should be installed by running:
```bash
./setup.sh
```

NVIDIA APEX can improve the runtime of these experiments. It can optionally be installed by running:
```bash
./install-apex.sh
```

The *experiment.py* tool can then be invoked. To see a summary of experiment options, run:
```bash
./experiment.py --help
```