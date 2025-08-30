#!/usr/bin/env python3
"""
RunPod Environment Setup Script
Checks existing libraries and installs only what's needed for the optimizer
Compatible with: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
"""

import subprocess
import sys
import importlib.util
from typing import Dict, List, Tuple
import pkg_resources

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and get its version"""
    if import_name is None:
        import_name = package_name.split('[')[0].split('>=')[0].split('==')[0]
    
    try:
        # Try to get version from pkg_resources
        try:
            version = pkg_resources.get_distribution(import_name).version
            return True, version
        except:
            # Fallback to import check
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                # Try to get version from module
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                return True, version
            return False, None
    except:
        return False, None

def get_required_packages() -> Dict[str, Dict[str, str]]:
    """Define required packages for the optimizer"""
    
    # Packages that are typically pre-installed in PyTorch containers
    preinstalled = {
        'torch': {'min_version': '2.0.0', 'import_name': 'torch'},
        'torchvision': {'min_version': '0.15.0', 'import_name': 'torchvision'},
        'numpy': {'min_version': '1.24.0', 'import_name': 'numpy'},
        'pillow': {'min_version': '10.0.0', 'import_name': 'PIL'},
        'jupyter': {'min_version': None, 'import_name': 'jupyter'},
        'jupyterlab': {'min_version': None, 'import_name': 'jupyterlab'},
    }
    
    # Essential packages for the optimizer
    essential = {
        'optuna': {'min_version': '3.3.0', 'import_name': 'optuna'},
        'requests': {'min_version': '2.31.0', 'import_name': 'requests'},
        'opencv-python': {'min_version': '4.8.0', 'import_name': 'cv2'},
        'scikit-image': {'min_version': '0.21.0', 'import_name': 'skimage'},
        'PyYAML': {'min_version': '6.0', 'import_name': 'yaml'},
        'tqdm': {'min_version': '4.65.0', 'import_name': 'tqdm'},
        'joblib': {'min_version': '1.3.0', 'import_name': 'joblib'},
        'plotly': {'min_version': '5.15.0', 'import_name': 'plotly'},
        'sqlalchemy': {'min_version': '2.0.0', 'import_name': 'sqlalchemy'},
        'GitPython': {'min_version': '3.1.0', 'import_name': 'git'},  # Required for ComfyUI-Manager
    }
    
    # Optional but recommended packages
    optional = {
        'lpips': {'min_version': '0.1.4', 'import_name': 'lpips'},
        'kaleido': {'min_version': '0.2.1', 'import_name': 'kaleido'},
        'pandas': {'min_version': '2.0.0', 'import_name': 'pandas'},
        'psutil': {'min_version': '5.9.0', 'import_name': 'psutil'},
    }
    
    return {
        'preinstalled': preinstalled,
        'essential': essential,
        'optional': optional
    }

def check_environment():
    """Check the RunPod environment and installed packages"""
    print(f"\n{Colors.BOLD}=== RunPod Environment Check ==={Colors.ENDC}\n")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"{Colors.BLUE}Python Version:{Colors.ENDC} {python_version}")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"{Colors.BLUE}CUDA:{Colors.ENDC} {Colors.GREEN}✓{Colors.ENDC} Version {cuda_version}")
            print(f"{Colors.BLUE}GPU(s):{Colors.ENDC} {gpu_count} x {gpu_name}")
        else:
            print(f"{Colors.BLUE}CUDA:{Colors.ENDC} {Colors.RED}✗{Colors.ENDC} Not available")
    except ImportError:
        print(f"{Colors.BLUE}PyTorch:{Colors.ENDC} {Colors.RED}✗{Colors.ENDC} Not installed")
    
    print(f"\n{Colors.BOLD}=== Package Status ==={Colors.ENDC}\n")
    
    packages = get_required_packages()
    to_install = []
    
    # Check preinstalled packages
    print(f"{Colors.BOLD}Pre-installed packages (PyTorch container):{Colors.ENDC}")
    for pkg, info in packages['preinstalled'].items():
        installed, version = check_package(pkg, info['import_name'])
        if installed:
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {pkg:20} {version}")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.ENDC} {pkg:20} Not found (usually pre-installed)")
    
    # Check essential packages
    print(f"\n{Colors.BOLD}Essential packages for optimizer:{Colors.ENDC}")
    for pkg, info in packages['essential'].items():
        installed, version = check_package(pkg, info['import_name'])
        if installed:
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {pkg:20} {version}")
        else:
            print(f"  {Colors.RED}✗{Colors.ENDC} {pkg:20} Not installed")
            min_ver = info['min_version']
            if min_ver:
                to_install.append(f"{pkg}>={min_ver}")
            else:
                to_install.append(pkg)
    
    # Check optional packages
    print(f"\n{Colors.BOLD}Optional packages (recommended):{Colors.ENDC}")
    optional_missing = []
    for pkg, info in packages['optional'].items():
        installed, version = check_package(pkg, info['import_name'])
        if installed:
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {pkg:20} {version}")
        else:
            print(f"  {Colors.YELLOW}○{Colors.ENDC} {pkg:20} Not installed (optional)")
            min_ver = info['min_version']
            if min_ver:
                optional_missing.append(f"{pkg}>={min_ver}")
            else:
                optional_missing.append(pkg)
    
    return to_install, optional_missing

def install_packages(packages: List[str], optional: bool = False):
    """Install missing packages"""
    if not packages:
        return True
    
    pkg_type = "optional" if optional else "required"
    print(f"\n{Colors.BOLD}Installing {pkg_type} packages:{Colors.ENDC}")
    for pkg in packages:
        print(f"  - {pkg}")
    
    if optional:
        response = input(f"\nInstall optional packages? (y/n): ").lower()
        if response != 'y':
            print("Skipping optional packages.")
            return True
    
    print(f"\n{Colors.YELLOW}Installing packages...{Colors.ENDC}")
    
    try:
        # Use pip to install packages
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{Colors.GREEN}✓ Successfully installed {pkg_type} packages{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.RED}✗ Error installing packages:{Colors.ENDC}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"{Colors.RED}✗ Installation failed: {e}{Colors.ENDC}")
        return False

def create_minimal_requirements():
    """Create a minimal requirements file for RunPod"""
    minimal_reqs = """# Minimal requirements for RunPod optimizer
# PyTorch and torchvision are pre-installed in RunPod container

# Core optimization
optuna>=3.3.0
requests>=2.31.0

# Image processing
opencv-python>=4.8.0
scikit-image>=0.21.0
Pillow>=10.0.0

# Configuration and utilities
PyYAML>=6.0
tqdm>=4.65.0
joblib>=1.3.0

# Visualization
plotly>=5.15.0

# Database for study persistence
sqlalchemy>=2.0.0

# ComfyUI dependencies
GitPython>=3.1.0  # Required for ComfyUI-Manager

# Optional but recommended
# lpips>=0.1.4  # For advanced perceptual metrics
# pandas>=2.0.0  # For data export
# psutil>=5.9.0  # For system monitoring
"""
    
    with open('requirements_runpod.txt', 'w') as f:
        f.write(minimal_reqs)
    
    print(f"\n{Colors.GREEN}✓ Created requirements_runpod.txt{Colors.ENDC}")

def main():
    """Main setup function"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════╗")
    print("║   RunPod Optimizer Environment Setup     ║")
    print("╚══════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    # Check environment and packages
    required_missing, optional_missing = check_environment()
    
    # Install missing packages
    if required_missing:
        print(f"\n{Colors.YELLOW}Missing {len(required_missing)} required package(s){Colors.ENDC}")
        success = install_packages(required_missing, optional=False)
        if not success:
            print(f"\n{Colors.RED}✗ Failed to install required packages{Colors.ENDC}")
            print("Please install manually or check error messages above.")
            sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}✓ All required packages are installed{Colors.ENDC}")
    
    # Offer to install optional packages
    if optional_missing:
        print(f"\n{Colors.YELLOW}{len(optional_missing)} optional package(s) not installed{Colors.ENDC}")
        install_packages(optional_missing, optional=True)
    
    # Create minimal requirements file
    create_minimal_requirements()
    
    # Final status
    print(f"\n{Colors.BOLD}=== Setup Complete ==={Colors.ENDC}")
    print(f"\n{Colors.GREEN}✓ RunPod environment is ready for optimization!{Colors.ENDC}")
    print(f"\nTo run the optimizer:")
    print(f"  {Colors.BLUE}python optimizer.py --workflow ../example_workflow.json --target ../target.png{Colors.ENDC}")
    
    # Check for ComfyUI
    print(f"\n{Colors.BOLD}Checking ComfyUI connection...{Colors.ENDC}")
    try:
        import requests
        response = requests.get("http://localhost:8188/", timeout=2)
        if response.status_code == 200:
            print(f"{Colors.GREEN}✓ ComfyUI is running on port 8188{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}⚠ ComfyUI responded but with status code {response.status_code}{Colors.ENDC}")
    except requests.exceptions.ConnectionError:
        print(f"{Colors.YELLOW}⚠ ComfyUI not accessible on port 8188{Colors.ENDC}")
        print(f"  Make sure ComfyUI is running with: {Colors.BLUE}python main.py --listen 0.0.0.0{Colors.ENDC}")
    except requests.exceptions.Timeout:
        print(f"{Colors.YELLOW}⚠ ComfyUI connection timed out{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Could not check ComfyUI: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()