import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class CheckpointManager:
    """Manages model checkpoints and training state."""
    
    def __init__(self, 
                 save_dir: str = "checkpoints",
                 max_checkpoints: int = 5,
                 save_best_only: bool = True,
                 monitor: str = "loss",
                 mode: str = "min"):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: If True, only save when metric improves
            monitor: Metric to monitor for best model
            mode: 'min' for loss, 'max' for accuracy
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_paths = []
        
        # Load existing checkpoints
        self._load_checkpoint_info()
    
    def _load_checkpoint_info(self):
        """Load information about existing checkpoints."""
        info_file = self.save_dir / "checkpoint_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
                self.best_value = info.get('best_value', self.best_value)
                self.checkpoint_paths = info.get('checkpoint_paths', [])
    
    def _save_checkpoint_info(self):
        """Save checkpoint information."""
        info_file = self.save_dir / "checkpoint_info.json"
        info = {
            'best_value': self.best_value,
            'checkpoint_paths': self.checkpoint_paths
        }
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       filename: Optional[str] = None) -> bool:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Dictionary of metrics
            filename: Optional custom filename
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        current_value = metrics.get(self.monitor, float('inf'))
        
        # Check if we should save
        should_save = True
        if self.save_best_only:
            if self.mode == 'min':
                should_save = current_value < self.best_value
            else:
                should_save = current_value > self.best_value
        
        if not should_save:
            return False
        
        # Update best value
        if self.mode == 'min':
            self.best_value = min(self.best_value, current_value)
        else:
            self.best_value = max(self.best_value, current_value)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_value': self.best_value
        }
        
        # Generate filename
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pth"
        
        checkpoint_path = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update checkpoint list
        self.checkpoint_paths.append(str(checkpoint_path))
        
        # Remove old checkpoints if exceeding max
        if len(self.checkpoint_paths) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_paths.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        # Save checkpoint info
        self._save_checkpoint_info()
        
        return True
    
    def load_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint (uses latest if None)
            
        Returns:
            Dictionary with checkpoint information
        """
        if checkpoint_path is None:
            if not self.checkpoint_paths:
                raise ValueError("No checkpoints available")
            checkpoint_path = self.checkpoint_paths[-1]
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if self.checkpoint_paths:
            return self.checkpoint_paths[-1]
        return None


class TrainingLogger:
    """Handles training logging and TensorBoard integration."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 experiment_name: Optional[str] = None):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.experiment_dir))
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Save experiment config
        self.config = {}
    
    def log_metrics(self, 
                   metrics: Dict[str, float], 
                   step: int, 
                   prefix: str = ""):
        """
        Log metrics to TensorBoard and history.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        for name, value in metrics.items():
            # Add prefix if provided
            full_name = f"{prefix}_{name}" if prefix else name
            
            # Log to TensorBoard
            self.writer.add_scalar(full_name, value, step)
            
            # Add to history
            if full_name in self.history:
                self.history[full_name].append(value)
    
    def log_model_graph(self, model: nn.Module, dummy_input: torch.Tensor):
        """Log model graph to TensorBoard."""
        self.writer.add_graph(model, dummy_input)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.config.update(hparams)
        self.writer.add_hparams(hparams, {})
    
    def log_text(self, text: str, step: int, tag: str = "text"):
        """Log text to TensorBoard."""
        self.writer.add_text(tag, text, step)
    
    def log_images(self, images: torch.Tensor, step: int, tag: str = "images"):
        """Log images to TensorBoard."""
        self.writer.add_images(tag, images, step)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if self.history['train_loss'] and self.history['val_loss']:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy plot
        if self.history['train_accuracy'] and self.history['val_accuracy']:
            axes[0, 1].plot(self.history['train_accuracy'], label='Train Accuracy')
            axes[0, 1].plot(self.history['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate plot
        if self.history['learning_rate']:
            axes[1, 0].plot(self.history['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Loss vs Accuracy
        if (self.history['train_loss'] and self.history['train_accuracy'] and 
            len(self.history['train_loss']) == len(self.history['train_accuracy'])):
            axes[1, 1].scatter(self.history['train_loss'], self.history['train_accuracy'], 
                             alpha=0.6, label='Train')
            if self.history['val_loss'] and self.history['val_accuracy']:
                axes[1, 1].scatter(self.history['val_loss'], self.history['val_accuracy'], 
                                 alpha=0.6, label='Val')
            axes[1, 1].set_xlabel('Loss')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Loss vs Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_history(self, filename: str = "training_history.json"):
        """Save training history to JSON."""
        history_file = self.experiment_dir / filename
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_config(self, filename: str = "config.json"):
        """Save experiment configuration."""
        config_file = self.experiment_dir / filename
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


class TrainingManager:
    """Combines checkpoint management and logging."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 save_dir: str = "checkpoints",
                 log_dir: str = "logs",
                 experiment_name: Optional[str] = None,
                 max_checkpoints: int = 5,
                 save_best_only: bool = True,
                 monitor: str = "val_loss",
                 mode: str = "min"):
        """
        Initialize training manager.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            save_dir: Directory for checkpoints
            log_dir: Directory for logs
            experiment_name: Name of experiment
            max_checkpoints: Maximum checkpoints to keep
            save_best_only: Only save best model
            monitor: Metric to monitor
            mode: 'min' or 'max'
        """
        self.model = model
        self.optimizer = optimizer
        
        self.checkpoint_manager = CheckpointManager(
            save_dir=save_dir,
            max_checkpoints=max_checkpoints,
            save_best_only=save_best_only,
            monitor=monitor,
            mode=mode
        )
        
        self.logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name=experiment_name
        )
        
        self.epoch = 0
        self.global_step = 0
    
    def log_epoch(self, 
                  train_metrics: Dict[str, float],
                  val_metrics: Optional[Dict[str, float]] = None,
                  learning_rate: Optional[float] = None):
        """
        Log metrics for current epoch.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
        """
        # Log training metrics
        self.logger.log_metrics(train_metrics, self.epoch, prefix="train")
        
        # Log validation metrics
        if val_metrics:
            self.logger.log_metrics(val_metrics, self.epoch, prefix="val")
        
        # Log learning rate
        if learning_rate is not None:
            self.logger.log_metrics({'learning_rate': learning_rate}, self.epoch)
        
        # Combine metrics for checkpoint
        all_metrics = {**train_metrics}
        if val_metrics:
            all_metrics.update(val_metrics)
        if learning_rate is not None:
            all_metrics['learning_rate'] = learning_rate
        
        # Save checkpoint
        saved = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.epoch, all_metrics
        )
        
        if saved:
            print(f"Checkpoint saved at epoch {self.epoch}")
        
        self.epoch += 1
    
    def log_step(self, metrics: Dict[str, float]):
        """Log metrics for current step."""
        self.logger.log_metrics(metrics, self.global_step)
        self.global_step += 1
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            self.model, self.optimizer, checkpoint_path
        )
        
        self.epoch = checkpoint['epoch'] + 1
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint path."""
        return self.checkpoint_manager.get_latest_checkpoint()
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        self.logger.plot_training_history(save_path)
    
    def save_experiment(self):
        """Save experiment data."""
        self.logger.save_history()
        self.logger.save_config()
    
    def close(self):
        """Close logger."""
        self.logger.close()


# Utility functions
def create_training_manager(model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          **kwargs) -> TrainingManager:
    """Create a training manager with default settings."""
    return TrainingManager(model, optimizer, **kwargs)


def load_experiment(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   experiment_dir: str) -> TrainingManager:
    """Load an existing experiment."""
    experiment_path = Path(experiment_dir)
    
    # Load config
    config_file = experiment_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Create training manager
    manager = TrainingManager(
        model=model,
        optimizer=optimizer,
        save_dir=str(experiment_path / "checkpoints"),
        log_dir=str(experiment_path.parent),
        experiment_name=experiment_path.name,
        **config
    )
    
    # Load latest checkpoint
    latest_checkpoint = manager.get_latest_checkpoint()
    if latest_checkpoint:
        manager.load_checkpoint(latest_checkpoint)
        print(f"Loaded checkpoint from {latest_checkpoint}")
    
    return manager 