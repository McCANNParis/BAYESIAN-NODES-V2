#!/usr/bin/env python3
"""
Bayesian Optimization for ComfyUI on RunPod
Optimizes workflow parameters using Optuna with GPU acceleration
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

import requests
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import joblib

# Import utils from the same directory
from utils import ImageComparator, WorkflowAnalyzer, ConfigGenerator, ResultsExporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComfyUIOptimizer:
    """Bayesian optimizer for ComfyUI workflows on RunPod"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize optimizer with configuration"""
        self.config = self._load_config(config_path)
        self.api_url = self.config.get('runpod', {}).get('api_url', 'http://localhost:3000')
        self.workflow = None
        self.target_image = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # RunPod specific settings
        self.runpod_mode = os.environ.get('RUNPOD_POD_ID') is not None
        if self.runpod_mode:
            logger.info(f"Running on RunPod Pod: {os.environ.get('RUNPOD_POD_ID')}")
            self.api_url = f"http://localhost:{os.environ.get('COMFYUI_PORT', '3000')}"
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_workflow(self, workflow_path: str) -> None:
        """Load ComfyUI workflow from JSON file"""
        with open(workflow_path, 'r') as f:
            self.workflow = json.load(f)
        logger.info(f"Loaded workflow from {workflow_path}")
        
    def load_target_image(self, image_path: str) -> None:
        """Load target image for comparison"""
        self.target_image = cv2.imread(image_path)
        if self.target_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        self.target_tensor = self._image_to_tensor(self.target_image)
        logger.info(f"Loaded target image: {self.target_image.shape}")
        
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to torch tensor"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict:
        """Wait for ComfyUI to complete processing"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.api_url}/history/{prompt_id}")
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        status = history[prompt_id].get('status', {})
                        if status.get('completed'):
                            return history[prompt_id]
                        elif status.get('error'):
                            raise RuntimeError(f"ComfyUI error: {status.get('error')}")
            except requests.RequestException as e:
                logger.warning(f"Request error: {e}")
            time.sleep(1)
        raise TimeoutError(f"Timeout waiting for prompt {prompt_id}")
    
    def _get_output_image(self, outputs: Dict) -> Optional[np.ndarray]:
        """Extract output image from ComfyUI response"""
        for node_id, node_output in outputs.items():
            if 'images' in node_output and node_output['images']:
                image_data = node_output['images'][0]
                filename = image_data.get('filename')
                subfolder = image_data.get('subfolder', '')
                folder_type = image_data.get('type', 'output')
                
                # Construct the URL for fetching the image
                params = {'filename': filename, 'type': folder_type}
                if subfolder:
                    params['subfolder'] = subfolder
                
                response = self.session.get(f"{self.api_url}/view", params=params)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return None
    
    def run_workflow(self, parameters: Dict[str, Any]) -> Optional[np.ndarray]:
        """Execute workflow with given parameters"""
        workflow_copy = json.loads(json.dumps(self.workflow))
        
        # Update workflow with trial parameters
        for param_path, value in parameters.items():
            node_id, param_name = param_path.split('.')
            if node_id in workflow_copy and 'inputs' in workflow_copy[node_id]:
                workflow_copy[node_id]['inputs'][param_name] = value
        
        # Send to ComfyUI
        try:
            response = self.session.post(
                f"{self.api_url}/prompt",
                json={"prompt": workflow_copy}
            )
            response.raise_for_status()
            prompt_id = response.json().get('prompt_id')
            
            if not prompt_id:
                logger.error("No prompt_id received from ComfyUI")
                return None
            
            # Wait for completion
            result = self._wait_for_completion(prompt_id)
            
            # Get output image
            if 'outputs' in result:
                return self._get_output_image(result['outputs'])
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            return None
    
    def calculate_similarity(self, generated_image: np.ndarray) -> float:
        """Calculate similarity between generated and target images"""
        if generated_image is None:
            return 0.0
        
        # Resize if needed
        if generated_image.shape[:2] != self.target_image.shape[:2]:
            generated_image = cv2.resize(
                generated_image, 
                (self.target_image.shape[1], self.target_image.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        # Convert to grayscale for SSIM
        generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_score = ssim(self.target_gray, generated_gray)
        
        # Calculate perceptual loss if GPU available
        if self.device.type == 'cuda':
            generated_tensor = self._image_to_tensor(generated_image)
            with torch.no_grad():
                mse_loss = F.mse_loss(generated_tensor, self.target_tensor).item()
                perceptual_score = 1.0 / (1.0 + mse_loss)
        else:
            perceptual_score = ssim_score
        
        # Weighted combination
        return 0.7 * ssim_score + 0.3 * perceptual_score
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        # Get parameter suggestions based on config
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
        
        # Run workflow
        logger.info(f"Trial {trial.number}: {parameters}")
        generated_image = self.run_workflow(parameters)
        
        if generated_image is None:
            return 0.0
        
        # Calculate similarity
        score = self.calculate_similarity(generated_image)
        
        # Save best image
        if trial.number == 0 or score > trial.study.best_value:
            output_path = f"outputs/best_trial_{trial.number}_score_{score:.4f}.png"
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(output_path, generated_image)
            logger.info(f"Saved best image to {output_path}")
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return score
    
    def optimize(self, n_trials: int = 50, n_jobs: int = 1) -> optuna.Study:
        """Run Bayesian optimization"""
        # Create or load study
        study_name = self.config.get('study_name', 'comfyui_optimization')
        storage = self.config.get('storage', f'sqlite:///{study_name}.db')
        
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                direction='maximize'
            )
            logger.info(f"Loaded existing study with {len(study.trials)} trials")
        except:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='maximize'
            )
            logger.info("Created new study")
        
        # Add progress bar callback
        pbar = tqdm(total=n_trials, initial=len(study.trials))
        
        def progress_callback(study, trial):
            pbar.update(1)
            pbar.set_description(f"Best: {study.best_value:.4f}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[progress_callback],
            show_progress_bar=False
        )
        
        pbar.close()
        
        # Save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study: optuna.Study) -> None:
        """Save optimization results"""
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        best_params_path = output_dir / "best_parameters.json"
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save optimized workflow
        optimized_workflow = json.loads(json.dumps(self.workflow))
        for param_path, value in study.best_params.items():
            node_id, param_name = param_path.split('.')
            if node_id in optimized_workflow and 'inputs' in optimized_workflow[node_id]:
                optimized_workflow[node_id]['inputs'][param_name] = value
        
        optimized_workflow_path = output_dir / "optimized_workflow.json"
        with open(optimized_workflow_path, 'w') as f:
            json.dump(optimized_workflow, f, indent=2)
        
        # Generate and save plots
        try:
            fig = plot_optimization_history(study)
            fig.write_html(str(output_dir / "optimization_history.html"))
            
            if len(study.trials) > 1:
                fig = plot_param_importances(study)
                fig.write_html(str(output_dir / "param_importances.html"))
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Optimize ComfyUI workflow parameters')
    parser.add_argument('--workflow', type=str, required=True, help='Path to workflow JSON')
    parser.add_argument('--target', type=str, required=True, help='Path to target image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ComfyUIOptimizer(args.config)
    
    # Load workflow and target
    optimizer.load_workflow(args.workflow)
    optimizer.load_target_image(args.target)
    
    # Run optimization
    study = optimizer.optimize(n_trials=args.trials, n_jobs=args.jobs)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best Score: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()