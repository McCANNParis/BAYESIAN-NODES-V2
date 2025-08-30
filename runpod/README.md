# RunPod Deployment

This folder contains a **self-contained** optimizer for ComfyUI on RunPod cloud GPUs with CUDA acceleration. All necessary files are included - no dependencies on parent directories.

## Included Files

- `optimizer.py` - Main optimization engine
- `utils.py` - Image comparison and workflow utilities
- `config.yaml` - Configuration file
- `setup_runpod.py` - Intelligent package installer
- `deploy.sh` - Quick deployment script
- `requirements_runpod.txt` - Minimal dependencies
- `example_workflow.json` - Sample ComfyUI workflow

## Container Compatibility

Optimized for: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

This PyTorch container includes:
- PyTorch 2.2.0 with CUDA 12.1
- Python 3.10
- JupyterLab with extensions
- Common ML libraries pre-installed

## Prerequisites

- Active RunPod pod with ComfyUI installed
- RunPod PyTorch 2.2.0 container (recommended)
- CUDA-compatible GPU
- Target image for optimization

## Usage

### Basic Optimization
```bash
python optimizer.py --workflow example_workflow.json --target target_image.png
```

### Parallel Optimization (Multi-GPU)
```bash
python optimizer.py --workflow your_workflow.json --target target_image.png --jobs 4
```

### With Custom Configuration
```bash
python optimizer.py \
  --workflow your_workflow.json \
  --target target_image.png \
  --config config.yaml \
  --trials 100
```

## Environment Variables

```bash
export COMFYUI_PORT=8188  # If using non-default port
export RUNPOD_POD_ID=your_pod_id  # Automatically set by RunPod
```

## Features

- GPU acceleration with CUDA
- Parallel trials support for multi-GPU setups
- Memory management and GPU memory fraction control
- Automatic RunPod environment detection
- Advanced perceptual loss metrics with deep learning models

## Configuration

Edit `config.yaml` to customize:
- Parameter ranges and types
- GPU memory settings (`runpod.gpu_memory_fraction`)
- API URL and timeout settings
- Parallel job configuration

## Quick Setup on RunPod

### Automated Setup (Recommended)

1. **Create a RunPod Pod** with PyTorch 2.2.0 template
2. **SSH into the pod**:
   ```bash
   ssh root@[pod-ip] -p [pod-port]
   ```
3. **Clone this repository**:
   ```bash
   git clone [repo-url]
   cd BAYESIAN-NODES-V2/runpod
   ```
4. **Run the setup script** (checks and installs only missing packages):
   ```bash
   python setup_runpod.py
   ```
   This script will:
   - Check your RunPod environment
   - Detect pre-installed packages
   - Install only missing required packages
   - Optionally install recommended packages
   - Create a minimal requirements file

5. **Run optimization**:
   ```bash
   python optimizer.py --workflow your_workflow.json --target your_target.png
   ```

### Manual Setup

If you prefer manual installation:
```bash
# Install minimal requirements (recommended)
pip install -r requirements_runpod.txt
```

## Performance Tips

- Use multiple GPUs with `--jobs` parameter for faster optimization
- Reduce image resolution during optimization phase
- Enable pruning in config to stop bad trials early
- Mount persistent storage at `/workspace` to save results

## Troubleshooting

### CUDA Out of Memory
- Reduce `gpu_memory_fraction` in config
- Lower batch size in workflow
- Reduce image resolution

### Connection Timeout
- Increase timeout in config
- Check if ComfyUI is running: `curl http://localhost:8188/`
- Verify COMFYUI_PORT environment variable