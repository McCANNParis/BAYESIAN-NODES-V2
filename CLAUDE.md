# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bayesian optimization system for ComfyUI workflows that automatically finds optimal parameter settings by comparing generated images against target images. It uses Optuna for efficient parameter space exploration and supports both local macOS (ComfyUI Desktop) and RunPod cloud GPU environments.

## Project Structure

```
BAYESIAN-NODES-V2/
├── local/                     # macOS/ComfyUI Desktop deployment
│   ├── optimizer.py          # Main optimizer for local execution
│   ├── run.py               # Wrapper script with proper paths
│   └── README.md            # Local deployment documentation
├── runpod/                   # RunPod cloud deployment (self-contained)
│   ├── optimizer.py         # Main optimizer for RunPod
│   ├── utils.py            # Copy of shared utilities
│   ├── config.yaml         # Copy of configuration
│   ├── setup_runpod.py     # Environment checker and package installer
│   ├── deploy.sh            # Quick deployment script
│   ├── requirements_runpod.txt # Minimal requirements for RunPod
│   ├── example_workflow.json # Copy of example workflow
│   ├── run.py              # Wrapper script
│   ├── .gitignore          # Ignore generated files
│   └── README.md           # RunPod deployment documentation
├── utils.py                 # Shared utilities (accessible by both)
├── config.yaml             # Configuration file (shared)
├── requirements.txt        # Python dependencies (full list)
├── example_workflow.json   # Example ComfyUI workflow
└── outputs/                # Generated outputs directory
```

## Architecture

### Core Components

1. **Platform-Specific Optimizers**:
   - `local/optimizer.py`: macOS/ComfyUI Desktop implementation with Apple Silicon optimizations
   - `runpod/optimizer.py`: RunPod cloud GPU implementation with CUDA acceleration

2. **Shared Utilities** (`utils.py`):
   - `ImageComparator`: Advanced image similarity metrics (SSIM, LPIPS, perceptual loss)
   - `ConfigGenerator`: Auto-generates parameter configurations from workflows
   - `ResultsExporter`: Exports optimization results to various formats
   - `WorkflowManager`: Handles ComfyUI workflow manipulation

3. **Configuration** (`config.yaml`):
   - Defines parameters to optimize with ranges and types
   - Platform-specific settings (RunPod vs local)
   - Similarity metric weights and optimization settings

### Key Design Patterns

- **Bayesian Optimization**: Uses Optuna's TPE sampler for efficient parameter exploration
- **Platform Abstraction**: Separate optimizer classes for different deployment targets
- **Session Management**: Persistent HTTP sessions for ComfyUI API communication
- **Memory Management**: Platform-specific memory limits and GPU memory fraction controls

## Development Commands

### Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running Optimization

#### Local macOS (ComfyUI Desktop)
```bash
# Navigate to local directory
cd local/

# Basic optimization
python optimizer.py --workflow ../example_workflow.json --target ../target_image.png

# With custom trials
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png --trials 100

# Resume interrupted optimization
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png --resume
```

#### RunPod
```bash
# Navigate to runpod directory
cd runpod/

# Run setup script to check/install dependencies
python setup_runpod.py  # Only installs missing packages

# Basic optimization
python optimizer.py --workflow ../example_workflow.json --target ../target_image.png

# Parallel optimization with multiple GPUs
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png --jobs 4
```

**RunPod Specific Features:**
- **Self-contained folder**: All necessary files included, no parent directory dependencies
- `setup_runpod.py`: Intelligently checks pre-installed packages in PyTorch container
- `requirements_runpod.txt`: Minimal requirements excluding pre-installed packages
- `deploy.sh`: Automated deployment script for RunPod environments
- Optimized for `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` container
- Can be deployed independently by copying just the `runpod/` folder

### Configuration Management
```bash
# Generate config from workflow
python -c "from utils import ConfigGenerator; ConfigGenerator.generate_from_workflow('workflow.json', 'config.yaml')"

# Validate configuration
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

## Testing & Validation

Currently no formal test suite exists. To validate changes:

1. Test workflow loading and parsing
2. Verify API connectivity to ComfyUI
3. Validate image comparison metrics
4. Test optimization with small trial count (5-10 trials)

## Code Standards

- Use type hints for function signatures
- Handle API timeouts and connection errors gracefully
- Log important operations at INFO level
- Preserve workflow structure when updating parameters
- Validate parameter ranges before optimization

## Important Implementation Details

### ComfyUI API Integration
- API endpoint: `http://localhost:8188` (configurable)
- Queue prompt via POST to `/prompt`
- Poll `/history/{prompt_id}` for completion
- Retrieve images from `/view` endpoint

### Parameter Format
- Parameters identified as `"node_id.parameter_name"`
- Node IDs from exported workflow JSON
- Common parameters: cfg, steps, denoise, strength

### Image Comparison Pipeline
1. Resize images to same dimensions
2. Calculate multiple similarity metrics
3. Weighted combination for final score
4. Higher score = better match (0-1 range)

### Memory Management
- macOS: Monitor system memory, default 8GB limit
- RunPod: GPU memory fraction control (0.95 default)
- Automatic garbage collection between trials

## Common Issues & Solutions

- **ComfyUI not found**: Check installation path in `/Applications/`
- **API connection failed**: Ensure ComfyUI is running on correct port
- **Memory errors**: Reduce `max_memory_mb` in config.yaml
- **Slow optimization**: Reduce image resolution or sampling steps