# ComfyUI Bayesian Optimization Scripts

Automated parameter optimization for ComfyUI workflows using Bayesian optimization (Optuna). Find the perfect settings for your workflows by comparing generated images against a target image.

## Features

- **🎯 Bayesian Optimization**: Efficiently explores parameter space using Tree-structured Parzen Estimators
- **🖼️ Multiple Similarity Metrics**: SSIM, histogram comparison, edge detection, and perceptual loss
- **☁️ RunPod Support**: Optimized for cloud GPU environments with memory management
- **🍎 macOS Native**: Full support for ComfyUI Desktop on Apple Silicon and Intel Macs
- **📊 Real-time Visualization**: Track optimization progress with interactive plots
- **💾 Resume Capability**: Continue optimization from where you left off
- **🔄 Auto-detection**: Automatically finds ComfyUI installation and parameters

## Project Structure

The codebase is organized into two deployment-specific folders:

```
BAYESIAN-NODES-V2/
├── local/          # macOS/ComfyUI Desktop deployment
├── runpod/         # RunPod cloud GPU deployment (self-contained)
├── utils.py        # Shared utilities
├── config.yaml     # Configuration file
└── outputs/        # Generated results
```

**Note:** The `runpod/` folder is completely self-contained with all necessary files copied inside. You can deploy it independently by copying just that folder to your RunPod instance.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/comfyui-bayesian-optimizer.git
cd comfyui-bayesian-optimizer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### For macOS (ComfyUI Desktop)

```bash
# Navigate to local deployment folder
cd local/

# Simple optimization with default settings
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png

# With more trials for better results
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png --trials 100

# Using custom configuration
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png --config ../custom_config.yaml
```

#### For RunPod

```bash
# Navigate to RunPod deployment folder
cd runpod/

# Quick setup (checks existing packages, installs only what's needed)
python setup_runpod.py
# Or use the deployment script
./deploy.sh

# Basic optimization
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png

# With parallel trials (faster on multi-GPU)
python optimizer.py --workflow ../your_workflow.json --target ../target_image.png --trials 100 --jobs 4
```

**Note:** RunPod deployment is optimized for `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` container.

## Setup Guide

### 1. Prepare Your Workflow

1. Create your workflow in ComfyUI
2. Export it in API format:
   - Click the gear icon (⚙️) in ComfyUI
   - Select "Save (API Format)"
   - Save as `your_workflow.json`

### 2. Configure Parameters

Edit `config.yaml` to specify which parameters to optimize:

```yaml
parameters:
  "3.cfg":  # Node ID 3, parameter "cfg"
    type: float
    min: 1.0
    max: 20.0
    step: 0.5
    
  "3.steps":  # Node ID 3, parameter "steps"
    type: int
    min: 20
    max: 100
```

To find node IDs, check your exported workflow JSON file.

### 3. Prepare Target Image

Place your target image (the image you want to match) in the project directory:
- Supported formats: PNG, JPG, JPEG
- Any resolution (will be automatically resized for comparison)

## Advanced Configuration

### Auto-generate Configuration from Workflow

```python
from utils import ConfigGenerator

# Analyze workflow and create config
ConfigGenerator.generate_from_workflow("your_workflow.json", "config.yaml")
```

This will automatically detect all optimizable parameters in your workflow.

### Custom Similarity Metrics

Modify the similarity weights in `config.yaml`:

```yaml
optimization:
  similarity_weights:
    ssim: 0.5          # Structural similarity
    histogram: 0.3     # Color distribution  
    edge: 0.2          # Edge/structure similarity
```

### Memory Management

For RunPod:
```yaml
runpod:
  max_memory_gb: 24
  gpu_memory_fraction: 0.95
```

For macOS:
```yaml
local:
  max_memory_mb: 8192  # Adjust based on your Mac's RAM
```

## Workflow Examples

### Example 1: Optimize Sampling Parameters

```bash
# For local macOS
cd local/
python optimizer.py \
  --workflow ../sampling_workflow.json \
  --target ../reference.png \
  --trials 50
```

### Example 2: ControlNet Strength Tuning

```bash
# For local macOS
cd local/
python optimizer.py \
  --workflow ../controlnet_workflow.json \
  --target ../target.png \
  --config ../controlnet_config.yaml
```

### Example 3: LoRA Weight Optimization

```bash
# For RunPod
cd runpod/
python optimizer.py \
  --workflow ../multi_lora_workflow.json \
  --target ../style_reference.png \
  --trials 100
```

## Output Files

After optimization, you'll find:

```
outputs/
├── best_*.png                    # Best generated images
├── best_parameters.json          # Optimal parameter values
├── optimized_workflow.json       # Workflow with best parameters
├── optimization_history.html     # Interactive optimization plot
└── param_importances.html        # Parameter importance analysis
```

## Tips for Best Results

### 1. Start Small
Begin with 20-30 trials to test your setup, then increase for final optimization.

### 2. Choose the Right Parameters
Focus on the most impactful parameters:
- `cfg_scale`: Controls adherence to prompt (usually 1-20)
- `steps`: Number of denoising steps (20-150)
- `denoise`: Denoising strength (0-1)
- ControlNet/LoRA strengths if using them

### 3. Use Appropriate Metrics
- **SSIM**: Good for overall structure
- **Histogram**: Better for color matching
- **Edge**: Important for line art or precise shapes

### 4. Monitor Progress
The scripts show real-time progress and scores. If scores plateau early, you might need to:
- Adjust parameter ranges
- Add more parameters to optimize
- Use a different target image

### 5. Resume Interrupted Runs
Optimization automatically saves progress. To resume:
```bash
cd local/  # or cd runpod/
python optimizer.py --workflow ../same_workflow.json --target ../same_target.png
```

## Troubleshooting

### ComfyUI Not Found
- **macOS**: Ensure ComfyUI Desktop is installed in `/Applications/`
- **RunPod**: Check that ComfyUI is running on the expected port

### Slow Generation
- Reduce image resolution in your workflow
- Use fewer sampling steps during optimization
- Enable GPU acceleration (automatic on RunPod)

### Memory Issues
- Reduce `max_memory_mb` in config
- Lower batch size in workflow
- Close other applications

### API Connection Failed
1. Ensure ComfyUI is running
2. Check the API port (default: 8188)
3. For ComfyUI Desktop, it may need to be started manually first

## RunPod Specific Setup

### Environment Variables
```bash
export COMFYUI_PORT=8188  # If using non-default port
export RUNPOD_POD_ID=your_pod_id  # Automatically set by RunPod
```

### Persistent Storage
Mount volume at `/workspace` to persist optimization results across sessions.

### Multi-GPU Setup
```bash
# Use multiple GPUs for parallel trials
python runpod_optimizer.py --workflow workflow.json --target target.png --jobs 4
```

## macOS Specific Features

### Apple Silicon Optimization
The script automatically detects M1/M2/M3 chips and optimizes memory usage accordingly.

### ComfyUI Desktop Integration
- Auto-starts ComfyUI Desktop if not running
- Finds installation automatically
- Optimized for Metal Performance Shaders

## Export Results

### Generate Report
```python
from utils import ResultsExporter
import optuna

# Load study
study = optuna.load_study(
    study_name="comfyui_optimization",
    storage="sqlite:///optimization.db"
)

# Export results
ResultsExporter.export_to_csv(study, "results.csv")
ResultsExporter.export_to_json(study, "results.json")
ResultsExporter.create_report(study, "report.md")
```

## Performance Benchmarks

Typical optimization times (50 trials):
- **RunPod (A100)**: 15-30 minutes
- **RunPod (RTX 4090)**: 20-40 minutes  
- **Mac M1 Pro**: 45-90 minutes
- **Mac M2 Max**: 30-60 minutes

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Built with [Optuna](https://optuna.org/) for Bayesian optimization
- Designed for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Optimized for [RunPod](https://runpod.io/) cloud GPUs