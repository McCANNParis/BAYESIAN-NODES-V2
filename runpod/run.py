#!/usr/bin/env python3
"""
Wrapper script to run the RunPod optimizer
Usage: python run.py --workflow your_workflow.json --target target_image.png
"""

import os
import sys

# Import and run the optimizer
from optimizer import main

if __name__ == "__main__":
    # Update default config path if not specified
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', 'config.yaml'])
    
    main()