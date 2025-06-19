import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json, time, logging, gc
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
import biosaic as bio

from .model import DNA_VQVAE, VQConfig
from ._dataset import Dataset

class EarlyStopping:
  """Early stopping utility"""
  def __init__(self, patience: int = 7, min_delta: float = 0.001):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_loss = float('inf')

  def __call__(self, val_loss: float) -> bool:
    if val_loss < self.best_loss - self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      return False
    else:
      self.counter += 1
      return self.counter >= self.patience

class MultiDatasetTrainer:
  """Improved trainer for DNA VQ-VAE with better monitoring and stability"""

  def __init__(self, config: VQConfig, dataset_paths: List[str], dataset_names: List[str],  save_dir: str = "checkpoints", log_dir: str = "logs", kmer_size: int = 4, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    self.config = config
    self.dataset_paths = dataset_paths
    self.dataset_names = dataset_names
    self.save_dir = Path(save_dir)
    self.log_dir = Path(log_dir)
    self.kmer_size = kmer_size
    self.device = device

    # Create directories
    self.save_dir.mkdir(parents=True, exist_ok=True)
    self.log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    self.setup_logging()

    # Load datasets and initialize model
    self.datasets = {}
    self.current_dataset_idx = 0
    self.load_datasets()
    
    # Initialize model with correct vocab size
    vocab_size = bio.Tokenizer("dna", kmer_size, True)._tokenizer.vocab_size
    self.model = DNA_VQVAE(config).to(device)
    self.n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
    
    self.logger.info(f"Model initialized with {self.n_params:.2f}M parameters, vocab size: {vocab_size}")

    # Training state
    self.optimizer = None
    self.scheduler = None
    self.global_step = 0
    self.epoch = 0
    self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    self.training_history = []

    # Initialize tensorboard
    self.writer = SummaryWriter(log_dir=str(self.log_dir))

  def setup_logging(self):
    """Setup logging configuration"""
    log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    self.logger = logging.getLogger(__name__)

  def load_datasets(self):
    """Load and initialize all datasets"""
    self.logger.info("Loading datasets...")
    
    for i, (path, name) in enumerate(zip(self.dataset_paths, self.dataset_names)):
      try:
        dataset = Dataset(
          path=path,
          kmer=self.kmer_size,
          ratio=0.2,
          random_seed=42 + i,
          max_data_size=500000
        )
        
        self.datasets[name] = dataset
        stats = dataset.get_data_stats()
        self.logger.info(f"Dataset '{name}' loaded: {stats}")
        
      except Exception as e:
        self.logger.error(f"Failed to load dataset '{name}': {e}")
        continue

    if not self.datasets:
      raise ValueError("No datasets loaded successfully")

  def get_current_dataset(self) -> Dataset:
    """Get the current active dataset"""
    return self.datasets[self.dataset_names[self.current_dataset_idx]]

  def switch_dataset(self):
    """Switch to the next dataset in rotation"""
    self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.dataset_names)
    current_name = self.dataset_names[self.current_dataset_idx]
    self.logger.info(f"Switched to dataset: {current_name}")

  def setup_training(self, learning_rate: float = 1e-4, weight_decay: float = 0.01):
    """Setup optimizer and scheduler with improved settings"""
    self.optimizer = optim.AdamW(
      self.model.parameters(),
      lr=learning_rate,
      weight_decay=weight_decay,
      betas=(0.9, 0.95)  # Better for transformers
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
      warmup_steps = 1000
      if step < warmup_steps:
        return step / warmup_steps
      # Cosine decay after warmup
      return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (10000 - warmup_steps)))
    
    self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    self.logger.info("Training setup completed")

  def train_step(self, batch_size: int = 32, block_size: int = 512) -> Dict[str, float]:
    """Improved training step with better monitoring"""
    self.model.train()
    
    # Get batch from current dataset
    dataset = self.get_current_dataset()
    try:
      x, _ = dataset.get_batch("train", batch_size, block_size, self.device)
    except Exception as e:
      self.logger.warning(f"Failed to get batch, switching dataset: {e}")
      self.switch_dataset()
      dataset = self.get_current_dataset()
      x, _ = dataset.get_batch("train", batch_size, block_size, self.device)

    # Forward pass
    self.optimizer.zero_grad()
    x_recon, vq_loss, indices = self.model(x)
    total_loss, recon_loss = self.model.compute_loss(x, x_recon, vq_loss)
    
    # Backward pass with gradient clipping
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
    self.scheduler.step()

    # Compute metrics
    with torch.no_grad():
      # Reconstruction accuracy (for one-hot encoded inputs)
      pred_tokens = torch.argmax(x_recon, dim=-1)
      true_tokens = torch.argmax(x, dim=-1)
      accuracy = (pred_tokens == true_tokens).float().mean()
      
      # Codebook usage metrics
      unique_codes = len(torch.unique(indices))
      codebook_usage = self.model.get_codebook_usage()
      
      # Perplexity
      perplexity = torch.exp(recon_loss)

    return {
      'total_loss': total_loss.item(),
      'recon_loss': recon_loss.item(),
      'vq_loss': vq_loss.item(),
      'accuracy': accuracy.item(),
      'perplexity': perplexity.item(),
      'unique_codes': unique_codes,
      'codebook_usage': codebook_usage,
      'grad_norm': grad_norm.item(),
      'lr': self.optimizer.param_groups[0]['lr']
    }

  def validate(self, batch_size: int = 32, block_size: int = 512, num_batches: int = 10) -> Dict[str, float]:
    """Validation across all datasets"""
    self.model.eval()
    
    val_metrics = {
      'total_loss': 0.0, 'recon_loss': 0.0, 'vq_loss': 0.0,
      'accuracy': 0.0, 'perplexity': 0.0, 'unique_codes': 0.0, 'codebook_usage': 0.0
    }
    total_batches = 0

    with torch.no_grad():
      for dataset_name, dataset in self.datasets.items():
        for _ in range(num_batches):
          try:
            x, _ = dataset.get_batch("val", batch_size, block_size, self.device)
            x_recon, vq_loss, indices = self.model(x)
            total_loss, recon_loss = self.model.compute_loss(x, x_recon, vq_loss)
            
            # Compute metrics
            pred_tokens = torch.argmax(x_recon, dim=-1)
            true_tokens = torch.argmax(x, dim=-1)
            accuracy = (pred_tokens == true_tokens).float().mean()
            
            unique_codes = len(torch.unique(indices))
            codebook_usage = self.model.get_codebook_usage()
            perplexity = torch.exp(recon_loss)
            
            # Accumulate metrics
            val_metrics['total_loss'] += total_loss.item()
            val_metrics['recon_loss'] += recon_loss.item()
            val_metrics['vq_loss'] += vq_loss.item()
            val_metrics['accuracy'] += accuracy.item()
            val_metrics['perplexity'] += perplexity.item()
            val_metrics['unique_codes'] += unique_codes
            val_metrics['codebook_usage'] += codebook_usage
            
            total_batches += 1
            
          except Exception as e:
            self.logger.warning(f"Validation batch failed for {dataset_name}: {e}")
            continue

    # Average metrics
    if total_batches > 0:
      for key in val_metrics:
        val_metrics[key] /= total_batches

    return val_metrics

  def save_checkpoint(self, is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
      'epoch': self.epoch,
      'global_step': self.global_step,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'scheduler_state_dict': self.scheduler.state_dict(),
      'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config,
      'training_history': self.training_history,
      'current_dataset_idx': self.current_dataset_idx
    }

    filename = "best_model.pth" if is_best else f"checkpoint_epoch_{self.epoch:03d}.pth"
    filepath = self.save_dir / filename

    try:
      torch.save(checkpoint, filepath)
      self.logger.info(f"Checkpoint saved: {filepath}")
      
      # Save safetensors if available
      try:
        from safetensors.torch import save_file
        safetensors_path = filepath.with_suffix('.safetensors')
        save_file(self.model.state_dict(), safetensors_path)
        self.logger.info(f"Safetensors saved: {safetensors_path}")
      except ImportError:
        pass
        
      return str(filepath)
      
    except Exception as e:
      self.logger.error(f"Failed to save checkpoint: {e}")
      return None

  def load_checkpoint(self, checkpoint_path: str) -> bool:
    """Load model checkpoint"""
    try:
      checkpoint = torch.load(checkpoint_path, map_location=self.device)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      
      if self.optimizer:
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if self.scheduler:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
      self.epoch = checkpoint.get('epoch', 0)
      self.global_step = checkpoint.get('global_step', 0)
      self.training_history = checkpoint.get('training_history', [])
      self.current_dataset_idx = checkpoint.get('current_dataset_idx', 0)
      
      self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
      return True
      
    except Exception as e:
      self.logger.error(f"Failed to load checkpoint: {e}")
      return False

  def train(self, num_epochs: int = 100, batch_size: int = 32, block_size: int = 512,
            eval_interval: int = 500, save_interval: int = 1000, learning_rate: float = 1e-4,
            resume_from: Optional[str] = None):
    """Main training loop with improved monitoring"""
    
    # Setup training
    self.setup_training(learning_rate)
    
    # Resume from checkpoint if provided
    if resume_from:
      self.load_checkpoint(resume_from)

    self.logger.info("Starting training...")
    self.logger.info(f"Parameters: epochs={num_epochs}, batch_size={batch_size}, "f"block_size={block_size}, lr={learning_rate}")

    try:
      start_time = time.time()
      best_val_loss = float('inf')

      for epoch in range(self.epoch, num_epochs):
        self.epoch = epoch
        epoch_start_time = time.time()
        epoch_losses = []
        
        # Training phase
        steps_per_epoch = 200  # Adjust based on your needs
        
        for step in range(steps_per_epoch):
          self.global_step += 1
          
          # Training step
          metrics = self.train_step(batch_size, block_size)
          epoch_losses.append(metrics)
          
          # Log to tensorboard
          for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, self.global_step)
          
          # Print progress
          if step % 20 == 0:
            current_dataset = self.dataset_names[self.current_dataset_idx]
            self.logger.info(
              f"Step {self.global_step}: Dataset={current_dataset}, "
              f"Loss={metrics['total_loss']:.4f}, "
              f"Acc={metrics['accuracy']:.3f}, "
              f"Usage={metrics['codebook_usage']:.3f}, "
              f"Codes={metrics['unique_codes']}"
            )

          # Validation
          if self.global_step % eval_interval == 0:
            val_metrics = self.validate(batch_size, block_size)
            
            # Log validation metrics
            for key, value in val_metrics.items():
              self.writer.add_scalar(f'val/{key}', value, self.global_step)
            
            self.logger.info(
              f"Validation - Loss: {val_metrics['total_loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.3f}, "
              f"Usage: {val_metrics['codebook_usage']:.3f}"
            )
            
            # Check for improvement and early stopping
            if val_metrics['total_loss'] < best_val_loss:
              best_val_loss = val_metrics['total_loss']
              self.save_checkpoint(is_best=True)
              self.logger.info("New best model saved!")
            
            # Early stopping check
            if self.early_stopping(val_metrics['total_loss']):
              self.logger.info("Early stopping triggered")
              break

          # Regular checkpoint
          if self.global_step % save_interval == 0:
            self.save_checkpoint()

          # Dataset rotation
          if self.global_step % 500 == 0:
            self.switch_dataset()

          # Memory cleanup
          if self.global_step % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
              torch.cuda.empty_cache()

        # End of epoch
        avg_epoch_loss = np.mean([m['total_loss'] for m in epoch_losses])
        avg_accuracy = np.mean([m['accuracy'] for m in epoch_losses])
        avg_codebook_usage = np.mean([m['codebook_usage'] for m in epoch_losses])
        epoch_time = time.time() - epoch_start_time

        self.training_history.append({
          'epoch': epoch,
          'avg_loss': avg_epoch_loss,
          'avg_accuracy': avg_accuracy,
          'avg_codebook_usage': avg_codebook_usage,
          'epoch_time': epoch_time,
          'global_step': self.global_step
        })

        self.logger.info(
          f"Epoch {epoch + 1} completed in {epoch_time:.2f}s, "
          f"Avg Loss: {avg_epoch_loss:.4f}, "
          f"Avg Acc: {avg_accuracy:.3f}, "
          f"Codebook Usage: {avg_codebook_usage:.3f}"
        )

        # Early stopping check
        if self.early_stopping.counter >= self.early_stopping.patience:
          break

    except KeyboardInterrupt:
      self.logger.info("Training interrupted by user")
    except Exception as e:
      self.logger.error(f"Training failed: {e}")
      raise
    finally:
      # Final save and cleanup
      self.save_checkpoint()
      total_time = time.time() - start_time
      
      self.logger.info(f"Training completed in {total_time:.2f}s")
      self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
      
      self.writer.close()
      
      # Save training summary
      summary = {
        'total_time': total_time,
        'best_val_loss': best_val_loss,
        'total_steps': self.global_step,
        'final_epoch': self.epoch,
        'model_params': f"{self.n_params:.2f}M",
        'training_history': self.training_history
      }
      
      with open(self.save_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

# Example usage
def main():
  """Example training setup"""
  
  # Improved configuration
  config = VQConfig(
    vocab_size=256,  # Will be set automatically based on tokenizer
    d_model=512,
    codebook_size=1024,
    beta=0.25,
    gamma=0.99,  # EMA decay
    n_heads=8,
    n_layers=6,  # Reduced to prevent overfitting
    dropout=0.15,
    max_seq_len=1024,
    label_smoothing=0.1
  )
  
  dataset_paths = [
    "data/dataset1.txt",
    "data/dataset2.txt",
    "data/dataset3.txt"
  ]
  dataset_names = ["dataset1", "dataset2", "dataset3"]
  
  # Initialize trainer
  trainer = MultiDatasetTrainer(
    config=config,
    dataset_paths=dataset_paths,
    dataset_names=dataset_names,
    kmer_size=4,
    save_dir="checkpoints",
    log_dir="logs"
  )

  # Start training with improved settings
  trainer.train(
    num_epochs=50,
    batch_size=16,  # Smaller batch size for stability
    block_size=512,
    eval_interval=100,  # More frequent evaluation
    save_interval=500,
    learning_rate=1e-4  # Lower learning rate
  )

if __name__ == "__main__":
  main()