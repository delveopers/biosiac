import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json, time, logging, gc
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

# Google Drive integration
try:
  from google.colab import drive
  from google.colab import files
  COLAB_ENV = True
except ImportError:
  COLAB_ENV = False

# Import your model and dataset classes
from model import DNA_VQVAE, VQConfig
from ._dataset import Dataset  # Assuming the fixed dataset class is in dataset.py

class MultiDatasetTrainer:
  """
  Advanced trainer for DNA VQ-VAE with multi-dataset support, 
  automatic checkpointing, and robust error handling
  """
  
  def __init__(
    self,
    config: VQConfig,
    dataset_paths: List[str],
    dataset_names: List[str],
    save_dir: str = "/content/drive/MyDrive/checkpoints",
    log_dir: str = "/content/drive/MyDrive/logs",
    kmer_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
  ):
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
    
    # Initialize model
    self.model = DNA_VQVAE(config).to(device)
    self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    # Initialize datasets
    self.datasets = {}
    self.current_dataset_idx = 0
    self.load_datasets()
    
    # Training state
    self.optimizer = None
    self.scheduler = None
    self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    # Tracking variables
    self.global_step = 0
    self.epoch = 0
    self.best_val_loss = float('inf')
    self.patience_counter = 0
    self.training_history = []
    
    # Mount Google Drive if in Colab
    if COLAB_ENV:
      self.mount_drive()
    
    # Initialize tensorboard
    self.writer = SummaryWriter(log_dir=str(self.log_dir))
    
    self.logger.info("Trainer initialized successfully")

  def setup_logging(self):
    """Setup logging configuration"""
    log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
      ]
    )
    self.logger = logging.getLogger(__name__)

  def mount_drive(self):
    """Mount Google Drive in Colab environment"""
    if COLAB_ENV:
      try:
        drive.mount('/content/drive')
        self.logger.info("Google Drive mounted successfully")
      except Exception as e:
        self.logger.error(f"Failed to mount Google Drive: {e}")

  def load_datasets(self):
    """Load and initialize all datasets"""
    self.logger.info("Loading datasets...")
    
    for i, (path, name) in enumerate(zip(self.dataset_paths, self.dataset_names)):
      try:
        dataset = Dataset(
          path=path,
          kmer=self.kmer_size,
          ratio=0.2,  # 20% validation
          random_seed=42 + i,  # Different seed for each dataset
          max_data_size=500000  # Limit data size per dataset
        )
        
        self.datasets[name] = dataset
        stats = dataset.get_data_stats()
        self.logger.info(f"Dataset '{name}' loaded: {stats}")
        
      except Exception as e:
        self.logger.error(f"Failed to load dataset '{name}' from {path}: {e}")
        continue
    
    if not self.datasets:
      raise ValueError("No datasets loaded successfully")
    
    self.dataset_names = list(self.datasets.keys())
    self.logger.info(f"Successfully loaded {len(self.datasets)} datasets")

  def get_current_dataset(self) -> Dataset:
    """Get the current active dataset"""
    return self.datasets[self.dataset_names[self.current_dataset_idx]]

  def switch_dataset(self):
    """Switch to the next dataset in rotation"""
    self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.dataset_names)
    current_name = self.dataset_names[self.current_dataset_idx]
    self.logger.info(f"Switched to dataset: {current_name}")

  def setup_training(self, learning_rate: float = 1e-4, weight_decay: float = 1e-5):
    """Setup optimizer and scheduler"""
    self.optimizer = optim.AdamW(
      self.model.parameters(),
      lr=learning_rate,
      weight_decay=weight_decay,
      betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
      self.optimizer,
      T_0=1000,  # Initial restart period
      T_mult=2,  # Multiply restart period by this factor
      eta_min=1e-6
    )
    
    self.logger.info("Training setup completed")

  def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, vq_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute total loss with components"""
    # Reconstruction loss (cross-entropy for one-hot encoded sequences)
    recon_loss = nn.functional.cross_entropy(
      x_recon.reshape(-1, x_recon.size(-1)),
      x.argmax(dim=-1).reshape(-1),
      reduction='mean'
    )
    
    # Total loss
    total_loss = recon_loss + vq_loss
    
    return {
      'total_loss': total_loss,
      'recon_loss': recon_loss,
      'vq_loss': vq_loss,
      'perplexity': torch.exp(vq_loss)  # VQ perplexity
    }

  def train_step(self, batch_size: int = 32, block_size: int = 512) -> Dict[str, float]:
    """Single training step"""
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
    
    self.optimizer.zero_grad()
    
    # Forward pass with mixed precision if available
    if self.scaler is not None:
      with torch.cuda.amp.autocast():
        x_recon, vq_loss, indices = self.model(x)
        losses = self.compute_loss(x, x_recon, vq_loss)
        total_loss = losses['total_loss']
      
      # Backward pass
      self.scaler.scale(total_loss).backward()
      
      # Gradient clipping
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
      
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      x_recon, vq_loss, indices = self.model(x)
      losses = self.compute_loss(x, x_recon, vq_loss)
      total_loss = losses['total_loss']
      
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
      self.optimizer.step()
    
    self.scheduler.step()
    
    # Convert to float for logging
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

  def validate(self, batch_size: int = 32, block_size: int = 512, num_batches: int = 10) -> Dict[str, float]:
    """Validation step across all datasets"""
    self.model.eval()
    
    val_losses = {
      'total_loss': 0.0,
      'recon_loss': 0.0,
      'vq_loss': 0.0,
      'perplexity': 0.0
    }
    
    total_batches = 0
    
    with torch.no_grad():
      for dataset_name, dataset in self.datasets.items():
        dataset_batches = 0
        
        for _ in range(num_batches):
          try:
            x, _ = dataset.get_batch("val", batch_size, block_size, self.device)
            
            if self.scaler is not None:
              with torch.cuda.amp.autocast():
                x_recon, vq_loss, indices = self.model(x)
                losses = self.compute_loss(x, x_recon, vq_loss)
            else:
              x_recon, vq_loss, indices = self.model(x)
              losses = self.compute_loss(x, x_recon, vq_loss)
            
            # Accumulate losses
            for key in val_losses:
              val_losses[key] += losses[key].item()
            
            dataset_batches += 1
            total_batches += 1
            
          except Exception as e:
            self.logger.warning(f"Validation batch failed for {dataset_name}: {e}")
            continue
    
    # Average losses
    if total_batches > 0:
      for key in val_losses:
        val_losses[key] /= total_batches
    
    return val_losses

  def save_checkpoint(self, is_best: bool = False, is_emergency: bool = False):
    """Save model checkpoint"""
    checkpoint = {
      'epoch': self.epoch,
      'global_step': self.global_step,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
      'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
      'config': self.config,
      'best_val_loss': self.best_val_loss,
      'training_history': self.training_history,
      'dataset_names': self.dataset_names,
      'current_dataset_idx': self.current_dataset_idx
    }
    
    # Determine filename
    if is_emergency:
      filename = f"emergency_checkpoint_step_{self.global_step}.pth"
    elif is_best:
      filename = "best_model.pth"
    else:
      filename = f"consolidated_{self.epoch:02d}.pth"
    
    filepath = self.save_dir / filename
    
    try:
      torch.save(checkpoint, filepath)
      self.logger.info(f"Checkpoint saved: {filepath}")
      
      # Also save as safetensors if available
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
      self.epoch = checkpoint.get('epoch', 0)
      self.global_step = checkpoint.get('global_step', 0)
      self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
      self.training_history = checkpoint.get('training_history', [])
      self.current_dataset_idx = checkpoint.get('current_dataset_idx', 0)
      
      if self.optimizer and 'optimizer_state_dict' in checkpoint:
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      
      if self.scheduler and 'scheduler_state_dict' in checkpoint:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      
      self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
      return True
      
    except Exception as e:
      self.logger.error(f"Failed to load checkpoint: {e}")
      return False

  def emergency_save(self):
    """Emergency save when training is interrupted"""
    self.logger.warning("Emergency save triggered!")
    try:
      checkpoint_path = self.save_checkpoint(is_emergency=True)
      if checkpoint_path and COLAB_ENV:
        # Download the checkpoint in Colab
        files.download(checkpoint_path)
      return checkpoint_path
    except Exception as e:
      self.logger.error(f"Emergency save failed: {e}")
      return None

  def train(
    self,
    num_epochs: int = 100,
    batch_size: int = 32,
    block_size: int = 512,
    eval_interval: int = 500,
    save_interval: int = 1000,
    patience: int = 10,
    learning_rate: float = 1e-4,
    resume_from: Optional[str] = None
  ):
    """Main training loop"""

    # Setup training
    self.setup_training(learning_rate)
    
    # Resume from checkpoint if provided
    if resume_from:
      self.load_checkpoint(resume_from)
    
    self.logger.info("Starting training...")
    self.logger.info(f"Training parameters: epochs={num_epochs}, batch_size={batch_size}, " f"block_size={block_size}, lr={learning_rate}")
    
    try:
      start_time = time.time()
      
      for epoch in range(self.epoch, num_epochs):
        self.epoch = epoch
        epoch_start_time = time.time()
        
        # Training phase
        self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = []
        steps_per_epoch = 100  # Number of steps per epoch
        
        for step in range(steps_per_epoch):
          self.global_step += 1
          
          # Training step
          losses = self.train_step(batch_size, block_size)
          epoch_losses.append(losses)
          
          # Log to tensorboard
          for key, value in losses.items():
            self.writer.add_scalar(f'train/{key}', value, self.global_step)
          
          self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
          
          # Print progress
          if step % 20 == 0:
            current_dataset = self.dataset_names[self.current_dataset_idx]
            self.logger.info(
              f"Step {self.global_step}: Dataset={current_dataset}, "
              f"Loss={losses['total_loss']:.4f}, "
              f"Recon={losses['recon_loss']:.4f}, "
              f"VQ={losses['vq_loss']:.4f}"
            )
          
          # Evaluation
          if self.global_step % eval_interval == 0:
            val_losses = self.validate(batch_size, block_size)
            
            # Log validation losses
            for key, value in val_losses.items():
              self.writer.add_scalar(f'val/{key}', value, self.global_step)
            
            self.logger.info(f"Validation - Loss: {val_losses['total_loss']:.4f}")
            
            # Check for improvement
            if val_losses['total_loss'] < self.best_val_loss:
              self.best_val_loss = val_losses['total_loss']
              self.patience_counter = 0
              self.save_checkpoint(is_best=True)
              self.logger.info("New best model saved!")
            else:
              self.patience_counter += 1
            
            # Early stopping check
            if self.patience_counter >= patience:
              self.logger.info(f"Early stopping after {patience} evaluations without improvement")
              break
          
          # Regular checkpoint saving
          if self.global_step % save_interval == 0:
            self.save_checkpoint()
          
          # Dataset rotation every 50 steps
          if self.global_step % 50 == 0:
            self.switch_dataset()
          
          # Memory cleanup
          if self.global_step % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
              torch.cuda.empty_cache()
        
        # End of epoch logging
        avg_epoch_loss = np.mean([losses['total_loss'] for losses in epoch_losses])
        epoch_time = time.time() - epoch_start_time
        
        self.training_history.append({
          'epoch': epoch,
          'avg_loss': avg_epoch_loss,
          'epoch_time': epoch_time,
          'global_step': self.global_step
        })
        
        self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s, " f"Avg Loss: {avg_epoch_loss:.4f}")
        
        # Early stopping check
        if self.patience_counter >= patience:
          break
    
    except KeyboardInterrupt:
      self.logger.info("Training interrupted by user")
      self.emergency_save()
    
    except Exception as e:
      self.logger.error(f"Training failed with error: {e}")
      self.emergency_save()
      raise
    
    finally:
      # Final save
      final_checkpoint = self.save_checkpoint()
      total_time = time.time() - start_time
      
      self.logger.info(f"Training completed in {total_time:.2f}s")
      self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
      self.logger.info(f"Final checkpoint: {final_checkpoint}")
      
      # Close tensorboard writer
      self.writer.close()
      
      # Save training summary
      summary = {
        'total_time': total_time,
        'best_val_loss': self.best_val_loss,
        'total_steps': self.global_step,
        'final_epoch': self.epoch,
        'training_history': self.training_history
      }
      
      summary_path = self.save_dir / 'training_summary.json'
      with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

# Example usage function
def main():
  """Example usage of the trainer"""
  
  # Configuration
  config = VQConfig(
    vocab_size=256,  # 4^4 for 4-mer tokenization
    d_model=512,
    codebook_size=1024,
    beta=0.25,
    n_heads=8,
    n_layers=6,
    dropout=0.1,
    max_seq_len=512
  )
  
  # Dataset paths (modify these to your actual paths)
  dataset_paths = [
    "/content/drive/MyDrive/data1.txt",
    "/content/drive/MyDrive/data2.txt", 
    "/content/drive/MyDrive/data3.txt"
  ]

  dataset_names = ["human", "mouse", "plant"]
  
  # Initialize trainer
  trainer = MultiDatasetTrainer(
    config=config,
    dataset_paths=dataset_paths,
    dataset_names=dataset_names,
    kmer_size=4
  )
  
  # Start training
  trainer.train(
    num_epochs=50,
    batch_size=16,  # Adjust based on GPU memory
    block_size=256,  # Adjust based on GPU memory
    eval_interval=200,
    save_interval=500,
    patience=5,
    learning_rate=1e-4
  )

if __name__ == "__main__":
  main()