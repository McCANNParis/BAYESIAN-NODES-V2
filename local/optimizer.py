#!/usr/bin/env python3
"""
Bayesian Optimization for ComfyUI Desktop on macOS
Optimized for Apple Silicon (M1/M2/M3) and Intel Macs
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import yaml
from tqdm import tqdm
from datetime import datetime

# Setup logging with better formatting for terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class ComfyUIDesktopOptimizer:
    """Bayesian optimizer for ComfyUI Desktop on macOS"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize optimizer with auto-detection for ComfyUI Desktop"""
        self.config = self._load_config(config_path)
        self.workflow = None
        self.target_image = None
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Detect ComfyUI Desktop installation
        self.comfyui_path = self._find_comfyui_desktop()
        self.api_url = self._get_api_url()
        
        # Check if running on Apple Silicon
        self.is_apple_silicon = platform.processor() == 'arm' or 'Apple' in platform.processor()
        if self.is_apple_silicon:
            logger.info(f"{Colors.OKCYAN}Running on Apple Silicon - optimizations enabled{Colors.ENDC}")
        
        # Memory management for Mac
        self.max_memory_mb = self.config.get('local', {}).get('max_memory_mb', 8192)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _find_comfyui_desktop(self) -> Optional[Path]:
        """Auto-detect ComfyUI Desktop installation on macOS"""
        possible_paths = [
            Path("/Applications/ComfyUI.app"),
            Path("~/Applications/ComfyUI.app").expanduser(),
            Path("/Applications/ComfyUI Desktop.app"),
            Path("~/Applications/ComfyUI Desktop.app").expanduser(),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"{Colors.OKGREEN}Found ComfyUI Desktop at: {path}{Colors.ENDC}")
                return path
        
        # Check if ComfyUI is running via process
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'ComfyUI'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"{Colors.OKGREEN}ComfyUI Desktop is already running{Colors.ENDC}")
                return Path("running")
        except:
            pass
        
        logger.warning(f"{Colors.WARNING}ComfyUI Desktop not found. Please ensure it's installed.{Colors.ENDC}")
        return None
    
    def _get_api_url(self) -> str:
        """Get ComfyUI API URL, checking if it's running"""
        default_ports = [8000, 8188, 8189, 3000]  # Added port 8000 which ComfyUI Desktop uses
        
        for port in default_ports:
            url = f"http://127.0.0.1:{port}"
            try:
                response = requests.get(f"{url}/system_stats", timeout=1)
                if response.status_code == 200:
                    logger.info(f"{Colors.OKGREEN}ComfyUI API found at {url}{Colors.ENDC}")
                    return url
            except:
                continue
        
        # Default to ComfyUI Desktop's port
        url = "http://127.0.0.1:8000"
        logger.info(f"{Colors.WARNING}Using default API URL: {url}{Colors.ENDC}")
        return url
    
    def start_comfyui(self) -> bool:
        """Start ComfyUI Desktop if not running"""
        try:
            # Check if already running
            response = requests.get(f"{self.api_url}/system_stats", timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        
        if self.comfyui_path and self.comfyui_path != Path("running"):
            logger.info(f"{Colors.OKCYAN}Starting ComfyUI Desktop...{Colors.ENDC}")
            try:
                subprocess.Popen(['open', str(self.comfyui_path)])
                
                # Wait for API to be available
                for i in range(30):
                    time.sleep(1)
                    try:
                        response = requests.get(f"{self.api_url}/system_stats", timeout=1)
                        if response.status_code == 200:
                            logger.info(f"{Colors.OKGREEN}ComfyUI Desktop started successfully{Colors.ENDC}")
                            return True
                    except:
                        continue
            except Exception as e:
                logger.error(f"{Colors.FAIL}Failed to start ComfyUI: {e}{Colors.ENDC}")
        
        return False
    
    def load_workflow(self, workflow_path: str) -> None:
        """Load ComfyUI workflow from JSON file"""
        with open(workflow_path, 'r') as f:
            self.workflow = json.load(f)
        logger.info(f"Loaded workflow: {workflow_path}")
        
        # Analyze workflow for optimizable parameters
        self._analyze_workflow()
    
    def _analyze_workflow(self) -> None:
        """Analyze workflow to identify optimizable parameters"""
        param_count = 0
        logger.info(f"\n{Colors.HEADER}=== Workflow Analysis ==={Colors.ENDC}")
        
        for node_id, node in self.workflow.items():
            if 'inputs' in node:
                node_class = node.get('class_type', 'Unknown')
                logger.info(f"Node {node_id} ({node_class}):")
                
                for param, value in node['inputs'].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        logger.info(f"  - {param}: {value} (optimizable)")
                        param_count += 1
        
        logger.info(f"\n{Colors.OKGREEN}Found {param_count} optimizable parameters{Colors.ENDC}\n")
    
    def load_target_image(self, image_path: str) -> None:
        """Load target image for comparison"""
        self.target_image = cv2.imread(image_path)
        if self.target_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        
        # Precompute features for faster comparison
        self.target_hist = cv2.calcHist([self.target_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        self.target_hist = cv2.normalize(self.target_hist, self.target_hist).flatten()
        
        logger.info(f"Loaded target image: {self.target_image.shape}")
    
    def run_workflow(self, parameters: Dict[str, Any]) -> Optional[np.ndarray]:
        """Execute workflow with given parameters"""
        workflow_copy = json.loads(json.dumps(self.workflow))
        
        # Update workflow with trial parameters
        for param_path, value in parameters.items():
            if '.' in param_path:
                node_id, param_name = param_path.split('.')
                if node_id in workflow_copy and 'inputs' in workflow_copy[node_id]:
                    workflow_copy[node_id]['inputs'][param_name] = value
        
        # Send to ComfyUI
        try:
            response = self.session.post(
                f"{self.api_url}/prompt",
                json={"prompt": workflow_copy},
                timeout=10
            )
            response.raise_for_status()
            prompt_id = response.json().get('prompt_id')
            
            if not prompt_id:
                logger.error("No prompt_id received")
                return None
            
            # Wait for completion with progress
            result = self._wait_for_completion(prompt_id)
            
            # Get output image
            if result and 'outputs' in result:
                return self._get_output_image(result['outputs'])
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            return None
    
    def _wait_for_completion(self, prompt_id: str, timeout: int = 120) -> Optional[Dict]:
        """Wait for ComfyUI to complete processing with progress indicator"""
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < timeout:
            try:
                # Check progress
                progress_response = self.session.get(f"{self.api_url}/prompt/{prompt_id}")
                if progress_response.status_code == 200:
                    progress_data = progress_response.json()
                    if progress_data and 'execution_progress' in progress_data:
                        current = progress_data['execution_progress'].get('current', 0)
                        total = progress_data['execution_progress'].get('total', 1)
                        if current != last_progress:
                            progress_pct = (current / total) * 100
                            print(f"\rGenerating: {progress_pct:.0f}%", end='', flush=True)
                            last_progress = current
                
                # Check completion
                history_response = self.session.get(f"{self.api_url}/history/{prompt_id}")
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        status = history[prompt_id].get('status', {})
                        if status.get('completed'):
                            print("\rGenerating: 100%", flush=True)
                            return history[prompt_id]
                        elif status.get('error'):
                            print(f"\r{Colors.FAIL}Error: {status.get('error')}{Colors.ENDC}", flush=True)
                            return None
            except:
                pass
            
            time.sleep(0.5)
        
        print(f"\r{Colors.FAIL}Timeout{Colors.ENDC}", flush=True)
        return None
    
    def _get_output_image(self, outputs: Dict) -> Optional[np.ndarray]:
        """Extract output image from ComfyUI response"""
        for node_id, node_output in outputs.items():
            if 'images' in node_output and node_output['images']:
                image_data = node_output['images'][0]
                filename = image_data.get('filename')
                subfolder = image_data.get('subfolder', '')
                folder_type = image_data.get('type', 'output')
                
                params = {'filename': filename, 'type': folder_type}
                if subfolder:
                    params['subfolder'] = subfolder
                
                try:
                    response = self.session.get(f"{self.api_url}/view", params=params, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                except:
                    pass
        return None
    
    def calculate_similarity(self, generated_image: np.ndarray) -> float:
        """Calculate multi-metric similarity score"""
        if generated_image is None:
            return 0.0
        
        # Resize if needed
        if generated_image.shape[:2] != self.target_image.shape[:2]:
            generated_image = cv2.resize(
                generated_image,
                (self.target_image.shape[1], self.target_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
        
        # 1. SSIM (structural similarity)
        generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        ssim_score = ssim(self.target_gray, generated_gray)
        
        # 2. Histogram comparison
        gen_hist = cv2.calcHist([generated_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        gen_hist = cv2.normalize(gen_hist, gen_hist).flatten()
        hist_score = 1 - cv2.compareHist(self.target_hist, gen_hist, cv2.HISTCMP_CHISQR_ALT)
        hist_score = max(0, min(1, hist_score))
        
        # 3. Edge similarity (for structure)
        target_edges = cv2.Canny(self.target_gray, 50, 150)
        gen_edges = cv2.Canny(generated_gray, 50, 150)
        edge_similarity = np.sum(target_edges == gen_edges) / target_edges.size
        
        # Weighted combination
        score = 0.5 * ssim_score + 0.3 * hist_score + 0.2 * edge_similarity
        
        return score
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function with memory management"""
        # Memory check for Mac
        if self.is_apple_silicon:
            self._check_memory()
        
        # Get parameter suggestions
        parameters = {}
        param_config = self.config.get('parameters', {})
        
        for param_path, settings in param_config.items():
            param_type = settings.get('type', 'float')
            if param_type == 'float':
                value = trial.suggest_float(
                    param_path,
                    settings.get('min', 0.0),
                    settings.get('max', 1.0),
                    step=settings.get('step')
                )
            elif param_type == 'int':
                value = trial.suggest_int(
                    param_path,
                    settings.get('min', 1),
                    settings.get('max', 100),
                    step=settings.get('step', 1)
                )
            elif param_type == 'categorical':
                value = trial.suggest_categorical(
                    param_path,
                    settings.get('choices', [])
                )
            parameters[param_path] = value
        
        # Display current trial
        print(f"\n{Colors.OKCYAN}Trial {trial.number}{Colors.ENDC}")
        for key, value in parameters.items():
            print(f"  {key}: {value}")
        
        # Run workflow
        generated_image = self.run_workflow(parameters)
        
        if generated_image is None:
            return 0.0
        
        # Calculate similarity
        score = self.calculate_similarity(generated_image)
        print(f"  {Colors.OKGREEN}Score: {score:.4f}{Colors.ENDC}")
        
        # Save if best
        if trial.number == 0 or score > trial.study.best_value:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/best_{timestamp}_score_{score:.4f}.png"
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(output_path, generated_image)
            print(f"  {Colors.OKGREEN}New best! Saved to {output_path}{Colors.ENDC}")
        
        return score
    
    def _check_memory(self) -> None:
        """Check and manage memory usage on Mac"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_mb = (memory.total - memory.available) / 1024 / 1024
            
            if used_mb > self.max_memory_mb:
                logger.warning(f"{Colors.WARNING}High memory usage: {used_mb:.0f}MB{Colors.ENDC}")
                # Force garbage collection
                import gc
                gc.collect()
        except:
            pass
    
    def optimize(self, n_trials: int = 50) -> optuna.Study:
        """Run Bayesian optimization with visual feedback"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}Starting Bayesian Optimization{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        # Ensure ComfyUI is running
        if not self.start_comfyui():
            logger.error(f"{Colors.FAIL}Could not start ComfyUI Desktop{Colors.ENDC}")
            logger.info(f"{Colors.WARNING}Please start ComfyUI Desktop manually and try again{Colors.ENDC}")
            return None
        
        # Create or load study
        study_name = self.config.get('study_name', 'comfyui_mac_optimization')
        storage = self.config.get('storage', f'sqlite:///{study_name}.db')
        
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                direction='maximize'
            )
            print(f"{Colors.OKCYAN}Resuming study with {len(study.trials)} existing trials{Colors.ENDC}\n")
        except:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            print(f"{Colors.OKCYAN}Created new study{Colors.ENDC}\n")
        
        # Progress tracking
        remaining_trials = n_trials
        start_time = time.time()
        
        def show_progress(study, trial):
            nonlocal remaining_trials
            remaining_trials -= 1
            elapsed = time.time() - start_time
            avg_time = elapsed / (n_trials - remaining_trials)
            eta = avg_time * remaining_trials
            
            print(f"\n{Colors.HEADER}Progress: {n_trials - remaining_trials}/{n_trials} trials{Colors.ENDC}")
            print(f"Best score so far: {Colors.OKGREEN}{study.best_value:.4f}{Colors.ENDC}")
            print(f"ETA: {eta/60:.1f} minutes")
            print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=[show_progress],
            show_progress_bar=False
        )
        
        # Save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study: optuna.Study) -> None:
        """Save optimization results with visualization"""
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        with open(output_dir / "best_parameters.json", 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save optimized workflow
        optimized_workflow = json.loads(json.dumps(self.workflow))
        for param_path, value in study.best_params.items():
            if '.' in param_path:
                node_id, param_name = param_path.split('.')
                if node_id in optimized_workflow and 'inputs' in optimized_workflow[node_id]:
                    optimized_workflow[node_id]['inputs'][param_name] = value
        
        with open(output_dir / "optimized_workflow.json", 'w') as f:
            json.dump(optimized_workflow, f, indent=2)
        
        # Generate visualization
        try:
            fig = plot_optimization_history(study)
            fig.write_html(str(output_dir / "optimization_history.html"))
            
            if len(study.trials) > 1:
                fig = plot_param_importances(study)
                fig.write_html(str(output_dir / "param_importances.html"))
            
            # Open visualization in browser
            subprocess.run(['open', str(output_dir / "optimization_history.html")])
        except:
            pass
        
        # Print summary
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}OPTIMIZATION COMPLETE!{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"\n{Colors.OKGREEN}Best Score:{Colors.ENDC} {study.best_value:.4f}")
        print(f"\n{Colors.OKGREEN}Best Parameters:{Colors.ENDC}")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"\n{Colors.OKCYAN}Results saved to:{Colors.ENDC} {output_dir}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(
        description='🎨 Optimize ComfyUI Desktop workflows on macOS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --workflow my_workflow.json --target target.png
  %(prog)s --workflow my_workflow.json --target target.png --trials 100
  %(prog)s --config custom_config.yaml --workflow workflow.json --target image.png
        """
    )
    
    parser.add_argument('--workflow', type=str, required=True,
                        help='Path to ComfyUI workflow (JSON format)')
    parser.add_argument('--target', type=str, required=True,
                        help='Path to target image for optimization')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of optimization trials (default: 50)')
    
    args = parser.parse_args()
    
    # Print header
    print(f"{Colors.HEADER}╔════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.HEADER}║     ComfyUI Desktop Bayesian Optimizer for macOS      ║{Colors.ENDC}")
    print(f"{Colors.HEADER}╚════════════════════════════════════════════════════════╝{Colors.ENDC}\n")
    
    # Initialize optimizer
    optimizer = ComfyUIDesktopOptimizer(args.config)
    
    # Load workflow and target
    optimizer.load_workflow(args.workflow)
    optimizer.load_target_image(args.target)
    
    # Run optimization
    study = optimizer.optimize(n_trials=args.trials)
    
    if study:
        print(f"{Colors.OKGREEN}✨ Optimization complete! Check the outputs folder for results.{Colors.ENDC}")


if __name__ == "__main__":
    main()