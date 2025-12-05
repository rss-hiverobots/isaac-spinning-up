# Motion Tracking Example

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Keywords:** motion-tracking, deepmimic, humanoids

## Installation

- Install Isaac Lab v2.3.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:leggedrobotics/isaaclab-spinning-up.git

# Option 2: HTTPS
git clone https://github.com/leggedrobotics/isaaclab-spinning-up.git
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/motion_tracking
```

## Motion Tracking

### Motion Preprocessing & Registry Setup

In order to manage the large set of motions we used in this work, we leverage the WandB registry to store and load
reference motions automatically.

Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

  - Unitree-retargeted LAFAN1 Dataset is available
    on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
  - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
  - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
  - Balance motions are from [HuB](https://hub-robot.github.io/)

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash
python scripts/motions/pkl_to_csv.py \
    --pkl-file source/motion_tracking/data/motions/pkl/g1_spinkick.pkl \
    --csv-file source/motion_tracking/data/motions/csv/g1_spinkick.csv \
    --duration 2.65 \
    --add-start-transition \
    --add-end-transition \
    --transition-duration 0.5 \
    --pad-duration 1.0
```

```bash
python scripts/motions/csv_to_npz.py \
    --input_file source/motion_tracking/data/motions/csv/g1_spinkick.csv \
    --input_fps 30 \
    --output_file source/motion_tracking/data/motions/npz/g1_spinkick.npz \
    --output_fps 50 \
    --headless
```

### Policy Training

- Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Motion-Tracking-G1-v0 \
    --headless \
    --logger wandb \
    --log_project_name unitree_g1_motion \
    --run_name sidekick
```

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Motion-Tracking-G1-v0 --num_envs=2
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
