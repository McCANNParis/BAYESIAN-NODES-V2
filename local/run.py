#!/usr/bin/env python3
"""
Wrapper script to run the local macOS optimizer
Usage: python run.py --workflow ../your_workflow.json --target ../target_image.png
"""

import os
import sys

# Add parent directory to path for accessing shared files
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import and run the optimizer
from optimizer import main

if __name__ == "__main__":
    # Update default config path if not specified
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', os.path.join(parent_dir, 'config.yaml')])
    
    main()