import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class CGR:
  def __init__(self, size=1000):
    self.size = size
    self.corners = {'A': (0, 0), 'T': (1, 0), 'G': (0, 1), 'C': (1, 1)}
    self.reset()

  def reset(self): self.x, self.y = 0.5, 0.5

  def step(self, base): 
    corner = self.corners.get(base.upper(), (0.5, 0.5))
    self.x, self.y = (self.x + corner[0]) / 2, (self.y + corner[1]) / 2
    return (self.x, self.y)

  def generate_points(self, sequence):
    self.reset()
    return [self.step(base) for base in sequence if base.upper() in self.corners]

  def create_heatmap(self, points):
    grid = np.zeros((self.size, self.size))
    coords = np.array([(int(y * (self.size - 1)), int(x * (self.size - 1))) for x, y in points])
    valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < self.size) & (coords[:, 1] >= 0) & (coords[:, 1] < self.size)
    coords = coords[valid_mask]

    if len(coords) > 0:
      # Vectorized point spreading using broadcasting
      offsets = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
      weights = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5])

      expanded_coords = coords[:, None, :] + offsets[None, :, :]
      expanded_coords = expanded_coords.reshape(-1, 2)
      expanded_weights = np.repeat(weights, len(coords))

      # Filter valid coordinates
      valid = (expanded_coords[:, 0] >= 0) & (expanded_coords[:, 0] < self.size) & (expanded_coords[:, 1] >= 0) & (expanded_coords[:, 1] < self.size)
      # Use np.add.at for fast accumulation
      np.add.at(grid, (expanded_coords[valid, 0], expanded_coords[valid, 1]), expanded_weights[valid])
    
    return grid

  def create_multicolor_heatmap(self, sequence):
    base_grids = {base: np.zeros((self.size, self.size)) for base in 'ATGC'}
    self.reset()
    
    for base in sequence:
      if base.upper() in self.corners:
        x, y = self.step(base)
        i, j = int(y * (self.size - 1)), int(x * (self.size - 1))
        if 0 <= i < self.size and 0 <= j < self.size:
          base_grids[base.upper()][i, j] += 1
    
    # Apply Gaussian blur for smoother visualization
    from scipy.ndimage import gaussian_filter
    sigma = max(1, self.size // 2000)  # Adaptive blur based on size
    for base in base_grids:
      if base_grids[base].max() > 0:
        base_grids[base] = gaussian_filter(base_grids[base], sigma=sigma)

    # Enhanced color mapping with better contrast
    rgb_image = np.zeros((self.size, self.size, 3))
    rgb_image[:, :, 0] = base_grids['A'] * 1.5 + base_grids['C'] * 0.8  # Red: A dominant, C mixed
    rgb_image[:, :, 1] = base_grids['T'] * 1.5 + base_grids['C'] * 0.8  # Green: T dominant, C mixed
    rgb_image[:, :, 2] = base_grids['G'] * 1.5 + base_grids['A'] * 0.3  # Blue: G dominant, A slight mix

    # Enhanced normalization with gamma correction for brightness
    for channel in range(3):
      if rgb_image[:, :, channel].max() > 0:
        rgb_image[:, :, channel] = rgb_image[:, :, channel] / rgb_image[:, :, channel].max()
        rgb_image[:, :, channel] = np.power(rgb_image[:, :, channel], 0.7)  # Gamma correction

    # Boost overall brightness and saturation
    rgb_image = np.clip(rgb_image * 2.5, 0, 1)

    return rgb_image

  def visualize(self, sequence, title="CGR Visualization", cmap='hot', figsize=(10, 10), label: bool=False, multicolor: bool=True):
    if multicolor:
      rgb_heatmap = self.create_multicolor_heatmap(sequence)
      plt.figure(figsize=figsize)
      plt.imshow(rgb_heatmap, origin='lower', extent=[0, 1, 0, 1])
      plt.title(f"{title}\nSequence length: {len(sequence)} (A=Red, T=Green, G=Blue, C=Purple)")
    else:
      points = self.generate_points(sequence)
      heatmap = self.create_heatmap(points)
      plt.figure(figsize=figsize)
      plt.imshow(heatmap, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
      plt.colorbar(label='Frequency')
      plt.title(f"{title}\nSequence length: {len(sequence)}")
    
    plt.xlabel('X'); plt.ylabel('Y')

    # Add corner labels
    if label:
      colors = ['red', 'lime', 'cyan', 'magenta'] if multicolor else ['white'] * 4
      corners_text = [('A', 0.02, 0.02, colors[0]), ('T', 0.98, 0.02, colors[1]), ('G', 0.02, 0.98, colors[2]), ('C', 0.98, 0.98, colors[3])]
      for text, x, y, color in corners_text:
        plt.text(x, y, text, fontsize=14, fontweight='bold', color=color, bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    return plt