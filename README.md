# Chronos

Official implementation of **Chronos: A Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation**.

[arXiv](https://arxiv.org/abs/2606.30318) | [Code](https://github.com/yulinzhouZYL/Chronos)

Chronos is a physics-informed full-history imitation learning framework for memory-dependent long-horizon manipulation. It treats observation history as the latent state of the policy dynamics and refines multimodal action priors through a second-order Schrödinger-inspired acceleration bridge.
<img width="2408" height="837" alt="e39d25de-303e-4b0d-aa4f-c7e9f2f195bf" src="https://github.com/user-attachments/assets/be021f1c-61ca-4cb4-9568-540af040f397" />

This repository provides the RMBench-based Chronos implementation. The included `RMBench/` folder contains the benchmark environment, and the Chronos policy is located at `RMBench/policy/Chronos`. Other RMBench policy folders may also be included for convenience. For the environment setup and configuration of other RMBench policies, please refer to the official RMBench repository: https://github.com/robotwin-Platform/rmbench.

## News

- **2026-06**: Initial public release of Chronos on RMBench.
- **Coming soon**: ALOHA code, RoboTwin 2.0 experiment scripts, real-world dual-arm deployment code, pretrained checkpoints, and full experiment configs.

## Repository Structure

```text
Chronos/
├── README.md
├── LICENSE
└── RMBench/
    ├── assets/
    ├── data/
    ├── description/
    ├── envs/
    ├── policy/
    │   └── Chronos/
    │       ├── deploy_policy.py
    │       ├── deploy_policy.yml
    │       ├── eval.sh
    │       ├── mamba_policy_par_3D_IMLE.py
    │       ├── mamba_controller.py
    │       ├── M_dataset_robotwin3D_E.py
    │       ├── train_par_3D_IMLE_EE.py
    │       └── checkpoints/
    ├── script/
    ├── task_config/
    └── collect_data.sh
```

The current release focuses on the RMBench implementation of Chronos. ALOHA, RoboTwin 2.0, real-world deployment code, pretrained checkpoints, and additional experiment configurations will be released in later updates.

## Installation

Create and activate the Chronos environment:

```bash
conda create -n Chronos python=3.10 -y
conda activate Chronos
```

Enter the included RMBench directory:

```bash
cd RMBench
```

Install the RMBench simulation environment and dependencies:

```bash
bash script/_install.sh
```

Download RMBench assets and data:

```bash
bash script/_download_assets.sh
bash script/_download_data.sh
```

## Mamba Dependencies

Chronos uses a Mamba-based full-history state encoder. After installing the RMBench environment, install the Mamba-related dependencies:

```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
pip install einops torchvision huggingface_hub
```

If installation fails, please check that your PyTorch, CUDA, and compiler versions are compatible with `mamba-ssm` and `causal-conv1d`.

## Data Collection Example

The following example uses the RMBench `cover_blocks` task with the `demo_clean` configuration.

From the `RMBench` root directory, collect demonstrations:

```bash
bash collect_data.sh cover_blocks demo_clean 0
```

For the current release, we use 55 demonstrations. After data collection, create a dataset split with two folders:

```text
train/
test/
```

Place 50 demonstrations into `train/` and 5 demonstrations into `test/`. The trajectory file format should remain unchanged.

A typical structure is:

```text
RMBench/
└── data/
    └── cover_blocks/
        └── demo_clean/
            ├── train/
            └── test/
```

If your RMBench data path differs, please update the corresponding paths in the Chronos dataset script before training.

## Dataset Normalization

Before training, generate the normalization/statistics file:

```bash
python policy/Chronos/M_dataset_robotwin3D_E.py
```

Please check the task name, task configuration, and dataset path inside the script before running. The default example is intended for:

```text
task_name   = cover_blocks
task_config = demo_clean
```

## Training

Start Chronos training with:

```bash
python policy/Chronos/train_par_3D_IMLE_EE.py
```

The current RMBench release trains the 3D point-cloud Chronos policy with:

- full-history Mamba state encoding,
- IMLE coarse action prior generation,
- second-order acceleration-field refinement,
- end-effector action prediction.

Checkpoints are expected to be saved under:

```text
policy/Chronos/checkpoints/cover_blocks/EE_16/
```

## Evaluation

After training, run evaluation from the Chronos policy directory:

```bash
cd policy/Chronos
bash eval.sh cover_blocks demo_clean Chronos 42 0
```

The arguments are:

```text
bash eval.sh <task_name> <task_config> <ckpt_name> <seed> <gpu_id>
```

Example:

```text
task_name   = cover_blocks
task_config = demo_clean
ckpt_name   = Chronos
seed        = 42
gpu_id      = 0
```

The evaluation script calls the RMBench evaluation entry point and loads:

```text
policy/Chronos/deploy_policy.yml
```

with checkpoints from:

```text
policy/Chronos/checkpoints/<task_name>/EE_16/
```

## Current Release Scope

Available now:

- RMBench-compatible Chronos policy code
- 3D point-cloud Chronos implementation
- full-history Mamba state encoder
- IMLE coarse action prior generator
- second-order acceleration-field refinement module
- RMBench training and evaluation entry points

Coming soon:

- ALOHA benchmark code
- RoboTwin 2.0 experiment scripts
- real-world dual-arm deployment code
- pretrained checkpoints
- full configuration files
- additional documentation and troubleshooting guide

## Citation

If you find Chronos useful, please cite:

```bibtex
@article{zhou2026chronos,
  title={Chronos: A Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation},
  author={Zhou, Yulin and Wang, Yimeng and Wang, Nengyu and Xing, Shaojia and Tu, Shiyun and Li, Xiang and Zhang, Jingkai and Jiang, Ningbo and Lin, Yuankai and Yang, Hua and Zeng, Xiangrui and Yin, Zhouping},
  journal={arXiv preprint arXiv:2606.30318},
  year={2026}
}
```

## Acknowledgement

This repository builds on the RMBench and RoboTwin 2.0 simulation ecosystem. We thank the authors of RMBench and RoboTwin for providing open-source robotic manipulation environments.

For the setup and configuration of other RMBench policies, please refer to the official RMBench repository:

```text
https://github.com/robotwin-Platform/rmbench
```

## License

This repository includes code adapted for RMBench-based Chronos experiments. RMBench is released under the MIT License. Please follow the original RMBench license terms when using or redistributing benchmark components.

Chronos policy code is released for research use. A formal license statement will be updated in a later release.
