#!/usr/bin/env python3
"""
Shared utility functions for ComfyUI Bayesian Optimization
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ImageComparator:
    """Advanced image comparison metrics"""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize image comparator with optional device specification"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.perceptual_loss = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_perceptual_model(self) -> None:
        """Load VGG16 for perceptual loss calculation"""
        if self.perceptual_loss is None:
            vgg = models.vgg16(pretrained=True).features.to(self.device).eval()
            self.perceptual_loss = vgg
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity)"""
        try:
            import lpips
            loss_fn = lpips.LPIPS(net='alex').to(self.device)
            
            # Convert images to tensors
            img1_tensor = self._prepare_image_tensor(img1)
            img2_tensor = self._prepare_image_tensor(img2)
            
            with torch.no_grad():
                distance = loss_fn(img1_tensor, img2_tensor)
            
            return 1.0 - distance.item()  # Convert distance to similarity
        except ImportError:
            # Fallback to MSE if LPIPS not available
            return self.calculate_mse_similarity(img1, img2)
    
    def calculate_mse_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate MSE-based similarity (inverse of MSE)"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return 1.0 / (1.0 + mse / 1000.0)  # Normalize to 0-1 range
    
    def calculate_feature_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity using deep features"""
        self.load_perceptual_model()
        
        img1_tensor = self._prepare_image_tensor(img1, normalize=True)
        img2_tensor = self._prepare_image_tensor(img2, normalize=True)
        
        with torch.no_grad():
            features1 = self._extract_features(img1_tensor)
            features2 = self._extract_features(img2_tensor)
            
            similarity = 0.0
            for f1, f2 in zip(features1, features2):
                similarity += torch.nn.functional.cosine_similarity(
                    f1.flatten().unsqueeze(0),
                    f2.flatten().unsqueeze(0)
                ).item()
            
            return similarity / len(features1)
    
    def _prepare_image_tensor(self, img: np.ndarray, normalize: bool = False) -> torch.Tensor:
        """Convert numpy image to torch tensor"""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        if img.shape[2] == 3 and img.dtype == np.uint8:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(img)
        
        if normalize:
            tensor = self.transform(pil_img).unsqueeze(0)
        else:
            tensor = transforms.ToTensor()(pil_img).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _extract_features(self, img_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from VGG"""
        features = []
        x = img_tensor
        
        for i, layer in enumerate(self.perceptual_loss):
            x = layer(x)
            if i in [3, 8, 15, 22, 29]:  # Conv layers at different scales
                features.append(x)
        
        return features


class WorkflowAnalyzer:
    """Analyze ComfyUI workflows to identify optimizable parameters"""
    
    # Node types that commonly have optimizable parameters
    OPTIMIZABLE_NODES = {
        'KSampler': ['cfg', 'steps', 'denoise'],
        'KSamplerAdvanced': ['cfg', 'steps', 'noise_seed', 'start_at_step', 'end_at_step'],
        'ControlNetApply': ['strength'],
        'ControlNetApplyAdvanced': ['strength', 'start_percent', 'end_percent'],
        'IPAdapterApply': ['weight', 'weight_type'],
        'LoraLoader': ['strength_model', 'strength_clip'],
        'CLIPTextEncode': [],  # Text prompts are not numerically optimizable
        'VAEDecode': [],
        'VAEEncode': [],
        'ImageScale': ['width', 'height'],
        'ImageScaleBy': ['scale_by'],
        'LatentUpscale': ['width', 'height'],
        'LatentUpscaleBy': ['scale_by'],
    }
    
    @classmethod
    def analyze(cls, workflow: Dict) -> Dict[str, Dict[str, Any]]:
        """Analyze workflow and return optimizable parameters"""
        parameters = {}
        
        for node_id, node in workflow.items():
            if not isinstance(node, dict) or 'inputs' not in node:
                continue
            
            node_type = node.get('class_type', '')
            
            # Check if this is an optimizable node type
            if node_type in cls.OPTIMIZABLE_NODES:
                optimizable_params = cls.OPTIMIZABLE_NODES[node_type]
                
                for param in optimizable_params:
                    if param in node['inputs']:
                        current_value = node['inputs'][param]
                        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
                            param_info = cls._infer_param_range(param, current_value, node_type)
                            parameters[f"{node_id}.{param}"] = param_info
            
            # Also check for any numeric parameters not in our predefined list
            elif 'inputs' in node:
                for param, value in node['inputs'].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        # Skip node connections (arrays with node ID)
                        if not isinstance(value, list):
                            param_key = f"{node_id}.{param}"
                            if param_key not in parameters:
                                param_info = cls._infer_param_range(param, value, node_type)
                                parameters[param_key] = param_info
        
        return parameters
    
    @classmethod
    def _infer_param_range(cls, param_name: str, current_value: Any, node_type: str) -> Dict[str, Any]:
        """Infer reasonable parameter ranges based on name and current value"""
        param_name_lower = param_name.lower()
        
        # Common parameter patterns
        if 'cfg' in param_name_lower or 'scale' in param_name_lower and 'cfg' in param_name_lower:
            return {
                'type': 'float',
                'min': 1.0,
                'max': 20.0,
                'current': current_value
            }
        elif 'steps' in param_name_lower:
            return {
                'type': 'int',
                'min': 10,
                'max': 150,
                'current': current_value
            }
        elif 'denoise' in param_name_lower or 'strength' in param_name_lower:
            return {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'current': current_value
            }
        elif 'seed' in param_name_lower:
            return {
                'type': 'int',
                'min': 0,
                'max': 2**31 - 1,
                'current': current_value
            }
        elif 'width' in param_name_lower or 'height' in param_name_lower:
            return {
                'type': 'int',
                'min': 64,
                'max': 2048,
                'step': 64,
                'current': current_value
            }
        elif 'weight' in param_name_lower:
            return {
                'type': 'float',
                'min': 0.0,
                'max': 2.0,
                'current': current_value
            }
        elif 'percent' in param_name_lower:
            return {
                'type': 'float',
                'min': 0.0,
                'max': 100.0,
                'current': current_value
            }
        else:
            # Default ranges based on current value
            if isinstance(current_value, int):
                return {
                    'type': 'int',
                    'min': max(0, current_value // 2),
                    'max': current_value * 2,
                    'current': current_value
                }
            else:
                return {
                    'type': 'float',
                    'min': max(0.0, current_value * 0.5),
                    'max': current_value * 2.0,
                    'current': current_value
                }


class ConfigGenerator:
    """Generate configuration files from workflow analysis"""
    
    @staticmethod
    def generate_from_workflow(workflow_path: str, output_path: str = "config.yaml") -> Dict:
        """Generate a config file from workflow analysis"""
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        analyzer = WorkflowAnalyzer()
        parameters = analyzer.analyze(workflow)
        
        config = {
            'study_name': Path(workflow_path).stem + '_optimization',
            'storage': f'sqlite:///{Path(workflow_path).stem}_optimization.db',
            'parameters': {},
            'runpod': {
                'api_url': 'http://localhost:8188',
                'max_memory_gb': 24
            },
            'local': {
                'api_url': 'http://127.0.0.1:8188',
                'max_memory_mb': 8192
            }
        }
        
        # Add discovered parameters to config
        for param_path, param_info in parameters.items():
            config['parameters'][param_path] = {
                'type': param_info['type'],
                'min': param_info['min'],
                'max': param_info['max'],
                'current': param_info['current']
            }
            if 'step' in param_info:
                config['parameters'][param_path]['step'] = param_info['step']
        
        # Save config
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return config


class ResultsExporter:
    """Export optimization results in various formats"""
    
    @staticmethod
    def export_to_csv(study, output_path: str = "optimization_results.csv") -> None:
        """Export study results to CSV"""
        import pandas as pd
        
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'state': str(trial.state),
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
            }
            trial_data.update(trial.params)
            trials_data.append(trial_data)
        
        df = pd.DataFrame(trials_data)
        df.to_csv(output_path, index=False)
    
    @staticmethod
    def export_to_json(study, output_path: str = "optimization_results.json") -> None:
        """Export study results to JSON"""
        results = {
            'study_name': study.study_name,
            'direction': study.direction.name,
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params,
            },
            'trials': []
        }
        
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
            }
            results['trials'].append(trial_data)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    @staticmethod
    def create_report(study, output_path: str = "optimization_report.md") -> None:
        """Create a markdown report of the optimization results"""
        with open(output_path, 'w') as f:
            f.write("# ComfyUI Workflow Optimization Report\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Trials**: {len(study.trials)}\n")
            f.write(f"- **Best Score**: {study.best_value:.4f}\n")
            f.write(f"- **Best Trial**: #{study.best_trial.number}\n\n")
            
            f.write("## Best Parameters\n\n")
            for param, value in study.best_params.items():
                f.write(f"- `{param}`: {value}\n")
            
            f.write("\n## Top 5 Trials\n\n")
            f.write("| Trial | Score | Parameters |\n")
            f.write("|-------|-------|------------|\n")
            
            sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
            for trial in sorted_trials:
                params_str = ', '.join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                       for k, v in trial.params.items()])
                f.write(f"| {trial.number} | {trial.value:.4f} | {params_str} |\n")


def validate_comfyui_connection(api_url: str = "http://127.0.0.1:8188") -> bool:
    """Validate that ComfyUI is running and accessible"""
    import requests
    
    try:
        response = requests.get(f"{api_url}/system_stats", timeout=2)
        return response.status_code == 200
    except:
        return False


def auto_detect_workflow_format(file_path: str) -> str:
    """Detect if workflow is in API format or regular format"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # API format has numeric keys and 'class_type' in nodes
    if isinstance(data, dict):
        first_key = list(data.keys())[0]
        if first_key.isdigit() and 'class_type' in data[first_key]:
            return 'api'
    
    return 'regular'


def convert_workflow_to_api_format(workflow_path: str, output_path: Optional[str] = None) -> str:
    """Convert regular ComfyUI workflow to API format"""
    # This would require ComfyUI's internal conversion logic
    # For now, return the path as-is and note that it needs manual conversion
    if output_path is None:
        output_path = workflow_path.replace('.json', '_api.json')
    
    print(f"Note: Please export your workflow in API format from ComfyUI")
    print(f"  1. Load your workflow in ComfyUI")
    print(f"  2. Click the settings gear icon")
    print(f"  3. Select 'Save (API Format)'")
    print(f"  4. Save as: {output_path}")
    
    return output_path