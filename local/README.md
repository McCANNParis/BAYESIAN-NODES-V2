# Local macOS Deployment

This folder contains the optimizer for ComfyUI Desktop on macOS, optimized for both Apple Silicon (M1/M2/M3) and Intel Macs.

## Prerequisites

- ComfyUI Desktop installed in `/Applications/`
- Python 3.8+ with dependencies installed (see `../requirements.txt`)
- Target image for optimization

## Usage

### Basic Optimization
```bash
python optimizer.py --workflow ../example_workflow.json --target ../target_image.png
```

### With Custom Configuration
```bash
python optimizer.py --workflow ../example_workflow.json --target ../target_image.png --config ../config.yaml
```

### Advanced Options
```bash
python optimizer.py \
  --workflow ../example_workflow.json \
  --target ../target_image.png \
  --trials 100 \
  --resume \
  --output-dir ../outputs
```

## Features

- Auto-detects ComfyUI Desktop installation
- Optimized for Apple Silicon with Metal Performance Shaders
- Memory management for macOS
- Automatic ComfyUI Desktop startup if not running
- Real-time progress visualization

## Configuration

Edit `../config.yaml` to customize:
- Parameter ranges and types
- Memory limits (`local.max_memory_mb`)
- API settings
- Similarity metrics weights

## Troubleshooting

### ComfyUI Desktop Not Found
Ensure ComfyUI Desktop is installed in `/Applications/`. If installed elsewhere, you may need to start it manually.

### Memory Issues
Reduce `max_memory_mb` in the config file or close other applications.

### Connection Failed
1. Check if ComfyUI Desktop is running
2. Verify port 8188 is accessible
3. Try starting ComfyUI Desktop manually first