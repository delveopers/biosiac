import matplotlib.pyplot as plt
import numpy as np
import random
import math

class CMap:
  def __init__(self, sequence, window_size=1000):
    self.seq = sequence.upper()
    self.window_size = window_size
    self.length = len(sequence)
    self.features = []
    self.orfs = []
    self.repeats = []
    
  def add_feature(self, start, end, name, feature_type='gene', strand='+'):
    colors = {'gene': '#4CAF50', 'cds': '#2196F3', 'rrna': '#FF9800', 'trna': '#9C27B0', 'repeat': '#F44336'}
    self.features.append({'start': start, 'end': end, 'name': name, 'type': feature_type, 'strand': strand, 'color': colors.get(feature_type, '#757575')})
    
  def add_orf(self, start, end, strand='+'): self.orfs.append({'start': start, 'end': end, 'strand': strand})
  def add_repeat(self, start, end, repeat_type='tandem'): self.repeats.append({'start': start, 'end': end, 'type': repeat_type})
  def gc_content(self, seq_chunk): return (seq_chunk.count('G') + seq_chunk.count('C')) / len(seq_chunk) if seq_chunk else 0
  def get_gc_windows(self): return [(self.gc_content(self.seq[i:i+self.window_size]), i) for i in range(0, self.length-self.window_size, self.window_size//8)]
  def pos_to_angle(self, pos): return 2 * np.pi * pos / self.length

  def draw_track_separators(self, ax, radii):
    for r in radii: ax.add_patch(plt.Circle((0, 0), r, fill=False, color='#E0E0E0', linewidth=0.3))

  def draw_feature_labels(self, ax, labels, radius=1.0):
    if not labels: return
    
    # Sort labels by angle
    labels.sort(key=lambda x: x['angle'])
    
    # Resolve overlaps
    min_sep = 0.4  # minimum angular separation
    for _ in range(8):  # max iterations
      overlaps = False
      for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          angle_diff = abs(labels[i]['angle'] - labels[j]['angle'])
          angle_diff = min(angle_diff, 2*np.pi - angle_diff)
          if angle_diff < min_sep:
            overlaps = True
            push = (min_sep - angle_diff) / 2
            labels[i]['angle'] = (labels[i]['angle'] - push) % (2*np.pi)
            labels[j]['angle'] = (labels[j]['angle'] + push) % (2*np.pi)
      if not overlaps: break
    
    # Draw labels
    for label in labels:
      x = label['radius'] * np.cos(label['angle'] - np.pi/2)
      y = label['radius'] * np.sin(label['angle'] - np.pi/2)
      rotation = math.degrees(label['angle']) - 90 if label['angle'] > np.pi else math.degrees(label['angle']) + 90
      if abs(rotation) > 90: rotation += 180
      ax.text(x, y, label['name'], ha='center', va='center', fontsize=6, rotation=rotation, 
              weight='bold', color=label['color'],
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor=label['color'], linewidth=0.8))
    major_ticks = max(1, self.length // 12)
    for i in range(0, self.length, major_ticks):
      angle = self.pos_to_angle(i)
      x, y = (radius + 0.08) * np.cos(angle - np.pi/2), (radius + 0.08) * np.sin(angle - np.pi/2)
      label = f"{i//1000000}M" if i >= 1000000 else f"{i//1000}k" if i >= 1000 else str(i)

  def draw_position_labels(self, ax, radius=1.2):
    major_ticks = max(12, self.length // 12)
    for i in range(0, self.length, major_ticks):
      angle = self.pos_to_angle(i)
      x, y = (radius + 0.08) * np.cos(angle - np.pi/2), (radius + 0.08) * np.sin(angle - np.pi/2)
      label = f"{i//1000000}M" if i >= 1000000 else f"{i//1000}k" if i >= 1000 else str(i)
      ax.text(x, y, label, ha='center', va='center', fontsize=6, rotation=math.degrees(angle) - 90 if angle > np.pi else math.degrees(angle) + 90, weight='light')

  def draw_outer_features_track(self, ax, inner_r=2.0, height=0.08):
    labels = []
    for feature in self.features:
      if feature['type'] == 'gene':
        start_angle, end_angle = self.pos_to_angle(feature['start']), self.pos_to_angle(feature['end'])
        radius = inner_r + height/2 + (height/2 if feature['strand'] == '+' else -height/2)
        angles = np.linspace(start_angle, end_angle, max(3, int((end_angle-start_angle)*50)))
        x, y = radius * np.cos(angles - np.pi/2), radius * np.sin(angles - np.pi/2)
        color = '#4169E1' if feature['strand'] == '+' else '#32CD32'
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        
        # Add label info
        mid_angle = (start_angle + end_angle) / 2
        labels.append({'angle': mid_angle, 'name': feature['name'], 'color': color, 'radius': inner_r + height + 0.15})
    
    self.draw_feature_labels(ax, labels)

  def draw_inner_features_track(self, ax, inner_r=1.85, height=0.08):
    labels = []
    for feature in self.features:
      if feature['type'] in ['rrna', 'trna']:
        start_angle, end_angle = self.pos_to_angle(feature['start']), self.pos_to_angle(feature['end'])
        radius = inner_r + height/2 + (height/2 if feature['strand'] == '+' else -height/2)
        angles = np.linspace(start_angle, end_angle, max(3, int((end_angle-start_angle)*50)))
        x, y = radius * np.cos(angles - np.pi/2), radius * np.sin(angles - np.pi/2)
        color = '#FF6B35' if feature['type'] == 'rrna' else '#9C27B0'
        ax.plot(x, y, color=color, linewidth=2, alpha=0.9)
        
        # Add label info for rRNA only (tRNA too small/numerous)
        if feature['type'] == 'rrna':
          mid_angle = (start_angle + end_angle) / 2
          labels.append({'angle': mid_angle, 'name': feature['name'], 'color': color, 'radius': inner_r + height + 0.12})
    
    self.draw_feature_labels(ax, labels)

  def draw_gc_skew_track(self, ax, inner_r=1.4, height=0.3):
    gc_data = self.get_gc_windows()
    mean_gc = np.mean([gc for gc, _ in gc_data])
    center_r = inner_r + height/2
    
    angles, values = [], []
    for gc, pos in gc_data:
      angle = self.pos_to_angle(pos)
      skew = (gc - mean_gc) * 2  # amplify for visibility
      angles.append(angle)
      values.append(skew)
    
    # Create continuous line plot
    for i in range(len(angles)-1):
      r1 = center_r + values[i] * height/2
      r2 = center_r + values[i+1] * height/2
      x1, y1 = r1 * np.cos(angles[i] - np.pi/2), r1 * np.sin(angles[i] - np.pi/2)
      x2, y2 = r2 * np.cos(angles[i+1] - np.pi/2), r2 * np.sin(angles[i+1] - np.pi/2)
      color = '#1976D2' if values[i] > 0 else '#D32F2F'
      ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5, alpha=0.7)

  def draw_gc_content_track(self, ax, inner_r=1.0, height=0.3):
    gc_data = self.get_gc_windows()
    for gc, pos in gc_data:
      angle = self.pos_to_angle(pos)
      gc_height = (gc - 0.3) * height / 0.4
      gc_height = max(0, min(height, gc_height))
      
      if gc > 0.55: color = '#2E7D32'
      elif gc < 0.45: color = '#C62828' 
      else: color = '#1565C0'
        
      x1, y1 = inner_r * np.cos(angle - np.pi/2), inner_r * np.sin(angle - np.pi/2)
      x2, y2 = (inner_r + gc_height) * np.cos(angle - np.pi/2), (inner_r + gc_height) * np.sin(angle - np.pi/2)
      ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.8)

  def draw_repeat_density_track(self, ax, inner_r=0.7, height=0.2):
    density_windows = []
    window_size = self.length // 100
    for i in range(0, self.length, window_size):
      count = sum(1 for r in self.repeats if r['start'] >= i and r['start'] < i + window_size)
      density_windows.append((count, i))
    
    max_density = max([d for d, _ in density_windows]) if density_windows else 1
    for density, pos in density_windows:
      angle = self.pos_to_angle(pos)
      bar_height = (density / max_density) * height
      x1, y1 = inner_r * np.cos(angle - np.pi/2), inner_r * np.sin(angle - np.pi/2)
      x2, y2 = (inner_r + bar_height) * np.cos(angle - np.pi/2), (inner_r + bar_height) * np.sin(angle - np.pi/2)
      ax.plot([x1, x2], [y1, y2], color='#FF5722', linewidth=1.5, alpha=0.6)

  def draw_histogram_tracks(self, ax):
    # Multiple histogram-like tracks with different data
    tracks = [
      {'inner_r': 0.4, 'height': 0.15, 'color': '#E91E63', 'density': 0.3},
      {'inner_r': 0.2, 'height': 0.12, 'color': '#FF9800', 'density': 0.5},
      {'inner_r': 0.05, 'height': 0.1, 'color': '#4CAF50', 'density': 0.7}
    ]
    
    for track in tracks:
      n_bars = int(200 * track['density'])
      for i in range(n_bars):
        angle = 2 * np.pi * i / n_bars + random.uniform(0, 0.1)
        bar_height = random.uniform(0.3, 1.0) * track['height']
        x1, y1 = track['inner_r'] * np.cos(angle - np.pi/2), track['inner_r'] * np.sin(angle - np.pi/2)
        x2, y2 = (track['inner_r'] + bar_height) * np.cos(angle - np.pi/2), (track['inner_r'] + bar_height) * np.sin(angle - np.pi/2)
        ax.plot([x1, x2], [y1, y2], color=track['color'], linewidth=0.8, alpha=0.6)

  def draw_radial_lines(self, ax, inner_r=0.05, outer_r=2.2):
    n_lines = 48
    for i in range(n_lines):
      angle = 2 * np.pi * i / n_lines
      x1, y1 = inner_r * np.cos(angle - np.pi/2), inner_r * np.sin(angle - np.pi/2)
      x2, y2 = outer_r * np.cos(angle - np.pi/2), outer_r * np.sin(angle - np.pi/2)
      ax.plot([x1, x2], [y1, y2], color='#E0E0E0', linewidth=0.2, alpha=0.5)

  def generate_realistic_features(self):
    feature_positions = []
    def is_position_free(start, end, min_gap=200): return all(end < pos_start - min_gap or start > pos_end + min_gap for pos_start, pos_end in feature_positions)

    # Generate genes
    for _ in range(max(15, self.length // 6000)):
      for _ in range(50):  # max attempts
        start = random.randint(0, self.length - 2000)
        length = random.randint(300, 2000)
        end = min(start + length, self.length)
        if is_position_free(start, end):
          self.add_feature(start, end, f"gene_{len(self.features)+1}", 'gene', random.choice(['+', '-']))
          self.add_orf(start, end, random.choice(['+', '-']))
          feature_positions.append((start, end))
          break
    
    # Generate rRNA
    for i, name in enumerate(['16S_rRNA', '23S_rRNA', '5S_rRNA']):
      for _ in range(20):
        start = random.randint(0, self.length - 1200)
        end = min(start + random.randint(800, 1200), self.length)
        if is_position_free(start, end, 500):
          self.add_feature(start, end, name, 'rrna', '+')
          feature_positions.append((start, end))
          break
    
    # Generate tRNA
    aa_list = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
    for _ in range(random.randint(20, 30)):
      for _ in range(30):
        start = random.randint(0, self.length - 80)
        end = min(start + random.randint(70, 85), self.length)
        if is_position_free(start, end, 100):
          self.add_feature(start, end, f"tRNA-{random.choice(aa_list)}", 'trna', random.choice(['+', '-']))
          feature_positions.append((start, end))
          break
    
    # Generate repeats
    for _ in range(random.randint(8, 15)):
      start = random.randint(0, self.length - 200)
      end = min(start + random.randint(50, 200), self.length)
      self.add_repeat(start, end)

  def plot(self, figsize=(16, 16), title=None, generate_features=True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    if generate_features and not self.features: self.generate_realistic_features()
    
    # Draw all tracks from inside out
    track_radii = [0.05, 0.2, 0.4, 0.7, 1.0, 1.4, 1.7, 1.85, 2.0, 2.2]
    self.draw_track_separators(ax, track_radii)
    self.draw_radial_lines(ax)
    self.draw_histogram_tracks(ax)
    self.draw_repeat_density_track(ax)
    self.draw_gc_content_track(ax)
    self.draw_gc_skew_track(ax)
    self.draw_inner_features_track(ax)
    self.draw_outer_features_track(ax)
    self.draw_position_labels(ax)
    
    # Add center circle
    ax.add_patch(plt.Circle((0, 0), 0.05, fill=True, color='white', zorder=10))
    
    plt.tight_layout()
    return fig, ax