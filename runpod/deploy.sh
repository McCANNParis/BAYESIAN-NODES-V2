#!/bin/bash
# RunPod Deployment Script
# Quick setup for RunPod PyTorch 2.2.0 container

set -e  # Exit on error

echo "================================================"
echo "   RunPod Bayesian Optimizer Deployment"
echo "   Container: PyTorch 2.2.0 + CUDA 12.1"
echo "================================================"
echo ""

# Check if we're in RunPod environment
if [ -z "$RUNPOD_POD_ID" ]; then
    echo "⚠️  Warning: RUNPOD_POD_ID not set. This might not be a RunPod environment."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ RunPod Pod ID: $RUNPOD_POD_ID"
fi

# Check Python version
echo ""
echo "Checking Python environment..."
python_version=$(python3 --version 2>&1)
echo "  Python: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "  CUDA: Available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo "  CUDA: Not detected"
fi

# Run setup script
echo ""
echo "Running environment setup..."
python3 setup_runpod.py

# Check if ComfyUI is accessible
echo ""
echo "Checking ComfyUI connection..."
comfyui_port=${COMFYUI_PORT:-3000}
if curl -s -o /dev/null -w "%{http_code}" http://localhost:$comfyui_port/ | grep -q "200\|404"; then
    echo "✓ ComfyUI is accessible on port $comfyui_port"
else
    echo "⚠️  ComfyUI not detected on port $comfyui_port"
    echo "   RunPod ComfyUI typically runs on port 3000"
    echo "   Check the web UI at: http://0.0.0.0:3000"
fi

echo ""
echo "================================================"
echo "   Setup Complete!"
echo "================================================"
echo ""
echo "To run optimization:"
echo "  python optimizer.py --workflow ../example_workflow.json --target ../target.png"
echo ""
echo "For parallel optimization (multi-GPU):"
echo "  python optimizer.py --workflow ../example_workflow.json --target ../target.png --jobs 4"
echo ""