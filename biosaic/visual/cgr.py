import numpy as np
import matplotlib.pyplot as plt

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
    coords = [(int(y * (self.size - 1)), int(x * (self.size - 1))) for x, y in points]
    for i, j in coords:
      if 0 <= i < self.size and 0 <= j < self.size: grid[i, j] += 1
    return grid

  def create_multicolor_heatmap(self, sequence):
    # create separate point lists for each base
    base_points = {'A': [], 'T': [], 'G': [], 'C': []}
    self.reset()
    
    for base in sequence:
      if base.upper() in self.corners:
        x, y = self.step(base)
        base_points[base.upper()].append((x, y))
    
    # create rgb image
    rgb_image = np.zeros((self.size, self.size, 3))

    # map each base to color channels with proper intensity
    for base, color_channel in [('A', 0), ('T', 1), ('G', 2)]:  # red, green, blue
      if base_points[base]:
        coords = [(int(y * (self.size - 1)), int(x * (self.size - 1))) for x, y in base_points[base]]
        for i, j in coords:
          if 0 <= i < self.size and 0 <= j < self.size:
            rgb_image[i, j, color_channel] += 1

    # handle C (cytosine) as magenta (red + blue)
    if base_points['C']:
      coords = [(int(y * (self.size - 1)), int(x * (self.size - 1))) for x, y in base_points['C']]
      for i, j in coords:
        if 0 <= i < self.size and 0 <= j < self.size:
          rgb_image[i, j, 0] += 0.8  # red component
          rgb_image[i, j, 2] += 0.8  # blue component

    # normalize each channel independently
    for c in range(3):
      if rgb_image[:, :, c].max() > 0:
        rgb_image[:, :, c] = rgb_image[:, :, c] / rgb_image[:, :, c].max()

    # apply gamma correction and brightness boost
    rgb_image = np.power(rgb_image, 0.8) * 1.5
    return np.clip(rgb_image, 0, 1)

  def visualize(self, sequence, title="CGR Visualization", cmap='hot', figsize=(12, 10), label=False, multicolor=True):
    plt.figure(figsize=figsize)

    if multicolor:
      rgb_heatmap = self.create_multicolor_heatmap(sequence)
      plt.imshow(rgb_heatmap, origin='lower', extent=[0, 1, 0, 1])
      plt.title(f"{title}\nSequence length: {len(sequence)} (A=Red, T=Green, G=Blue, C=Magenta)")
    else:
      points = self.generate_points(sequence)
      heatmap = self.create_heatmap(points)
      # apply log scaling for better visualization
      heatmap_log = np.log1p(heatmap)  # log(1+x) to handle zeros
      plt.imshow(heatmap_log, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
      plt.colorbar(label='log(frequency + 1)')
      plt.title(f"{title}\nSequence length: {len(sequence)}")

    plt.xlabel('X'); plt.ylabel('Y')

    # add corner labels with better visibility
    if label:
      corner_props = [('A', 0.02, 0.02, 'red'), ('T', 0.98, 0.02, 'lime'), ('G', 0.02, 0.98, 'cyan'), ('C', 0.98, 0.98, 'magenta')]
      for text, x, y, color in corner_props:
        plt.text(x, y, text, fontsize=16, fontweight='bold', color=color, bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.8, edgecolor=color))

    plt.tight_layout()
    return plt