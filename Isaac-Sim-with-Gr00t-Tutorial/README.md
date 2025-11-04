# Isaac Sim with GR00T Tutorial

This project demonstrates how to fine-tune NVIDIA's GR00T-N1.5-3B model and run inference in Isaac Sim for robotic manipulation tasks.

## Prerequisites

- **Two conda environments:**
  - `gr00t`: Environment with GR00T installed (for training and inference server)
  - `isaacsim`: Environment with Isaac Sim installed (for simulation)

## Workflow

### 1. Download Dataset

First, download your training dataset and place it in the `datasets/` directory. The expected format is a LeRobot-compatible dataset.

For example, the default configuration expects:
```
datasets/gr1_arms_only.Nut_pouring_task/
```

You can modify the `DATASET_PATH` variable in `run_finetune.py` to point to your specific dataset.

### 2. Fine-tune the Model

Fine-tune the NVIDIA GR00T-N1.5-3B model using your dataset.

**Important:** This step requires the `gr00t` conda environment.

```bash
conda activate gr00t
python run_finetune.py
```

**Configuration options in `run_finetune.py`:**
- `DATASET_PATH`: Path to your training dataset
- `FINETUNED_OUTPUT_DIRECTORY`: Where to save the fine-tuned model
- `BATCH_SIZE`: Training batch size
- `MAX_STEPS`: Maximum training steps
- `SAVE_STEPS`: Checkpoint saving frequency
- `TUNE_PROJECTOR`: Whether to tune the projector model (default: True)
- `TUNE_DIFFUSION_MODEL`: Whether to tune the diffusion model (default: False)

The fine-tuned model will be saved to the `finetuned/` directory (default: `./finetuned/gr1_arms_only.Nut_pouring_batch32_nodiffusion/`).

### 3. Launch the Inference Server

Start the inference server that will provide predictions for the simulation.

**Important:** This step requires the `gr00t` conda environment.

```bash
conda activate gr00t
python run_inference_server.py
```

The server will start on `localhost:9876` and provide a `/inference` endpoint.

**Note:** Make sure the `MODEL_PATH` in `run_inference_server.py` matches your fine-tuned model directory.

### 4. Run the Simulation

In a **separate terminal**, launch the Isaac Sim simulation to observe the rollout.

**Important:** This step requires the `isaacsim` conda environment.

```bash
conda activate isaacsim
python run_simulation.py
```

The simulation will connect to the inference server and execute the robotic task based on the fine-tuned model's predictions.

## Directory Structure

```
.
├── datasets/              # Training datasets (download here)
├── finetuned/            # Fine-tuned model checkpoints
├── results/              # Experiment results
├── sim_environments/     # Isaac Sim environment files (.usd)
├── run_finetune.py       # Fine-tuning script
├── run_inference_server.py  # Inference server (FastAPI)
├── run_simulation.py     # Isaac Sim simulation runner
└── README.md            # This file
```

## Quick Start Summary

```bash
# Terminal 1: Fine-tune (one-time setup)
conda activate gr00t
python run_finetune.py

# Terminal 1: Start inference server (keep running)
conda activate gr00t
python run_inference_server.py

# Terminal 2: Run simulation
conda activate isaacsim
python run_simulation.py
```

## Troubleshooting

- **Model path errors:** Ensure `MODEL_PATH` in `run_inference_server.py` matches your fine-tuned model output directory
- **Dataset errors:** Verify your dataset is in LeRobot format and the path in `run_finetune.py` is correct
- **Connection errors:** Make sure the inference server is running before starting the simulation
- **Conda environment errors:** Double-check you're using the correct environment for each script

## Notes

- The inference server must remain running while the simulation is active
- Fine-tuning may take several hours depending on your dataset size and hardware
- Default configuration uses `bfloat16` precision for efficient training
- Training progress can be monitored via Weights & Biases (wandb) if configured

