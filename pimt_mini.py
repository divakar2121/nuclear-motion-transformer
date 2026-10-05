"""
PiMT Mini - Educational Version
================================
Small transformer with clear architecture, probability outputs, and confidence analysis.

CRITICAL CLARIFICATION:
  In particle simulation: gradient updates ATOM POSITIONS (atoms move)
  In transformer: gradient updates NEURAL NETWORK WEIGHTS (network learns)
  
  The transformer does NOT move nuclei. It learns to CLASSIFY nuclei motion.
  Nuclei positions are FIXED input data; neural network weights are what change.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from collections import defaultdict

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# 1. SMALL TRANSFORMER WITH CLEAR ARCHITECTURE
# ============================================================================

class PiMTMini(nn.Module):
    """
    Physics-Informed Motion Transformer (Mini Version)
    
    Clear architecture for educational purposes:
      - Small: 32-dim embeddings, 1 transformer layer
      - Explicit: physics features as separate input
      - Probabilistic: outputs softmax probabilities
    
    Data Flow:
      trajectory (50×3) → Conv1D → Transformer → Embedding (32-dim)
      physics (5-dim)   → MLP    → Embedding (32-dim)
                                    ↓
                              Concatenate (64-dim)
                                    ↓
                              Classification (4 classes) + Alpha regression
    """
    
    def __init__(self, d_model=32, nhead=4, num_layers=1):
        super().__init__()
        
        # Trajectory encoder
        self.traj_conv1 = nn.Conv1d(3, 16, kernel_size=5, padding=2)
        self.traj_conv2 = nn.Conv1d(16, d_model, kernel_size=3, padding=1)
        
        # Transformer encoder
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Physics encoder
        self.physics_fc1 = nn.Linear(5, 16)
        self.physics_fc2 = nn.Linear(16, d_model)
        
        # Fusion and output
        self.fusion_fc = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, 4)
        self.alpha_head = nn.Linear(d_model, 1)
    
    def forward(self, trajectory, physics):
        """
        Forward pass with explicit data flow.
        
        Args:
            trajectory: (batch, 50, 3) - x,y,z positions
            physics: (batch, 5) - [alpha, confinement, msd, msd_std, vacf]
        
        Returns:
            logits: (batch, 4) - raw scores before softmax
            probabilities: (batch, 4) - softmax probabilities
            alpha_pred: (batch, 1) - predicted alpha value
        """
        batch_size = trajectory.shape[0]
        
        # === TRAJECTORY ENCODING ===
        # Step 1: Conv1D layers
        traj_t = trajectory.transpose(1, 2)  # (batch, 3, 50)
        x = F.relu(self.traj_conv1(traj_t))  # (batch, 16, 50)
        x = F.relu(self.traj_conv2(x))       # (batch, 32, 50)
        traj_feat = x.transpose(1, 2)        # (batch, 50, 32)
        
        # Step 2: Add positional encoding
        traj_feat = traj_feat + self.pos_encoding[:, :50, :]
        
        # Step 3: Transformer attention
        traj_encoded = self.transformer(traj_feat)  # (batch, 50, 32)
        traj_emb = traj_encoded.mean(dim=1)         # (batch, 32) - global pooling
        
        # === PHYSICS ENCODING ===
        phys_x = F.relu(self.physics_fc1(physics))  # (batch, 16)
        phys_emb = F.relu(self.physics_fc2(phys_x)) # (batch, 32)
        
        # === FUSION ===
        # Concatenate trajectory and physics embeddings
        fused = torch.cat([traj_emb, phys_emb], dim=1)  # (batch, 64)
        fused = F.relu(self.fusion_fc(fused))            # (batch, 32)
        
        # === OUTPUT HEADS ===
        logits = self.classifier(fused)      # (batch, 4) - raw scores
        alpha_pred = self.alpha_head(fused)  # (batch, 1)
        
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)  # (batch, 4)
        
        return logits, probabilities, alpha_pred


# ============================================================================
# 2. PHYSICS FEATURES (same as before)
# ============================================================================

def compute_msd(traj, max_lag=20):
    n = len(traj)
    msd = []
    for lag in range(1, min(max_lag + 1, n)):
        diff = traj[lag:] - traj[:-lag]
        msd.append(np.mean(np.sum(diff**2, axis=1)))
    return np.array(msd) if msd else np.array([0.0])

def compute_vacf(traj, max_lag=10):
    velocity = np.diff(traj, axis=0)
    n = len(velocity)
    vacf = []
    for lag in range(min(max_lag + 1, n)):
        if lag == 0:
            corr = np.mean(np.sum(velocity**2, axis=1))
        else:
            v1, v2 = velocity[:-lag], velocity[lag:]
            corr = np.mean(np.sum(v1 * v2, axis=1))
        vacf.append(corr)
    return np.array(vacf) if vacf else np.array([0.0])

def compute_alpha(traj, max_lag=15):
    msd = compute_msd(traj, max_lag)
    if len(msd) < 5:
        return 1.0
    lags = np.arange(1, len(msd) + 1)
    mask = msd > 0
    if np.sum(mask) < 3:
        return 1.0
    log_lags, log_msd = np.log(lags[mask]), np.log(msd[mask])
    slope, _ = np.polyfit(log_lags, log_msd, 1)
    return slope

def compute_confinement(traj):
    if len(traj) < 3:
        return 1.0
    max_disp = np.max(np.sqrt(np.sum((traj - traj[0])**2, axis=1)))
    path_length = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1)))
    return max_disp / (path_length + 1e-10)

def extract_physics_features(traj):
    alpha = compute_alpha(traj)
    confinement = compute_confinement(traj)
    msd = compute_msd(traj, max_lag=15)
    vacf = compute_vacf(traj, max_lag=10)
    return {
        'alpha': alpha,
        'confinement': confinement,
        'msd_mean': np.mean(msd) if len(msd) > 0 else 0.0,
        'msd_std': np.std(msd) if len(msd) > 0 else 0.0,
        'vacf_decay': vacf[-1] / (vacf[0] + 1e-10) if len(vacf) > 1 else 0.0,
    }


# ============================================================================
# 3. DATASET
# ============================================================================

class MotionDataset(Dataset):
    def __init__(self, trajectories, physics_features, labels, seq_len=50):
        self.trajs, self.physics, self.labels = [], [], []
        
        for traj, feat, label in zip(trajectories, physics_features, labels):
            if len(traj) < seq_len:
                padded = np.pad(traj, ((0, seq_len - len(traj)), (0, 0)))
            else:
                start = np.random.randint(0, len(traj) - seq_len + 1)
                padded = traj[start:start + seq_len]
            
            self.trajs.append(padded)
            self.physics.append(feat)
            self.labels.append(label)
        
        self.trajs = torch.FloatTensor(np.array(self.trajs))
        self.physics = torch.FloatTensor(np.array(self.physics))
        self.labels = torch.LongTensor(np.array(self.labels))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.trajs[idx], self.physics[idx], self.labels[idx]


def load_real_data(data_dir='data/zenodo', max_traj=100):
    """Load subset of real data for quick demonstration."""
    trajectories = []
    csv_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    csv_files.sort()
    
    for csv_path in csv_files:
        if len(trajectories) >= max_traj:
            break
        
        try:
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
            hl = [h.strip().lower() for h in header]
            col = {}
            for i, h in enumerate(hl):
                if h in ('x_um', 'x'): col['x'] = i
                if h in ('y_um', 'y'): col['y'] = i
                if h in ('z_um', 'z'): col['z'] = i
                if h in ('frame', 't'): col['frame'] = i
                if h in ('pid', 'particle', 'track_id'): col['pid'] = i
            
            if 'x' not in col or 'y' not in col:
                continue
            
            rows_by_pid = defaultdict(list)
            with open(csv_path, 'r') as f:
                f.readline()
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) <= max(col.values()):
                        continue
                    try:
                        x = float(parts[col['x']])
                        y = float(parts[col['y']])
                        z = float(parts[col['z']]) if 'z' in col else 0.0
                        frame = float(parts[col['frame']]) if 'frame' in col else 0
                        pid = int(parts[col['pid']]) if 'pid' in col else 0
                        rows_by_pid[pid].append((frame, x, y, z))
                    except:
                        continue
            
            for pid, rows in rows_by_pid.items():
                if len(rows) < 30:
                    continue
                rows.sort(key=lambda r: r[0])
                traj = np.array([(r[1], r[2], r[3]) for r in rows])
                trajectories.append(traj)
                
                if len(trajectories) >= max_traj:
                    break
        except:
            continue
    
    return trajectories


# ============================================================================
# 4. TRAINING WITH PROBABILITY ANALYSIS
# ============================================================================

def train_and_analyze(trajectories, n_epochs=15, batch_size=16):
    """Train PiMT Mini and perform probability/confidence analysis."""
    
    print("="*70)
    print("PiMT MINI - Training with Probability Analysis")
    print("="*70)
    
    # Extract physics features
    print("\nExtracting physics features...")
    physics_features = []
    labels = []
    
    for traj in trajectories:
        feat_dict = extract_physics_features(traj)
        feat_vec = [
            feat_dict['alpha'],
            feat_dict['confinement'],
            feat_dict['msd_mean'],
            feat_dict['msd_std'],
            feat_dict['vacf_decay'],
        ]
        physics_features.append(feat_vec)
        
        # Classify based on alpha
        alpha = feat_dict['alpha']
        if alpha < 0.5:
            labels.append(0)  # Confined
        elif alpha < 0.8:
            labels.append(1)  # Sub-diff
        elif alpha < 1.2:
            labels.append(2)  # Normal
        else:
            labels.append(3)  # Active
    
    physics_features = np.array(physics_features)
    labels = np.array(labels)
    
    print(f"  Loaded {len(trajectories)} trajectories")
    print(f"  Class distribution:")
    class_names = ['Confined', 'Sub-diff', 'Normal', 'Active']
    for i, name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"    {name}: {count} ({100*count/len(labels):.1f}%)")
    
    # Split train/test
    n_train = int(0.8 * len(trajectories))
    indices = np.random.permutation(len(trajectories))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_ds = MotionDataset(
        [trajectories[i] for i in train_idx],
        physics_features[train_idx],
        labels[train_idx],
        seq_len=50
    )
    
    test_ds = MotionDataset(
        [trajectories[i] for i in test_idx],
        physics_features[test_idx],
        labels[test_idx],
        seq_len=50
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PiMTMini(d_model=32, nhead=4, num_layers=1).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: PiMT Mini")
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture: Conv1D(2 layers) + Transformer(1 layer, 4 heads) + Physics MLP")
    print(f"  Device: {device}")
    
    # Loss and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for trajs, physics, labs in train_loader:
            trajs = trajs.to(device)
            physics = physics.to(device)
            labs = labs.to(device)
            
            logits, probs, alpha_pred = model(trajs, physics)
            
            loss_cls = criterion_cls(logits, labs)
            loss_reg = criterion_reg(alpha_pred.squeeze(), physics[:, 0])
            loss = loss_cls + 0.1 * loss_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(probs, 1)
            train_correct += (predicted == labs).sum().item()
            train_total += labs.size(0)
        
        train_acc = train_correct / train_total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: Train acc={train_acc:.3f}, Loss={train_loss/len(train_loader):.4f}")
    
    # === PROBABILITY AND CONFIDENCE ANALYSIS ===
    print("\n" + "="*70)
    print("PROBABILITY AND CONFIDENCE ANALYSIS")
    print("="*70)
    
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    all_confidences = []
    all_alphas_true = []
    all_alphas_pred = []
    
    with torch.no_grad():
        for trajs, physics, labs in test_loader:
            trajs = trajs.to(device)
            physics = physics.to(device)
            
            logits, probs, alpha_pred = model(trajs, physics)
            
            confidence, predicted = torch.max(probs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labs.numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_alphas_true.extend(physics[:, 0].cpu().numpy())
            all_alphas_pred.extend(alpha_pred.cpu().numpy().squeeze())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_alphas_true = np.array(all_alphas_true)
    all_alphas_pred = np.array(all_alphas_pred)
    
    correct_mask = (all_preds == all_labels)
    
    # Accuracy
    accuracy = correct_mask.mean()
    print(f"\nOverall Accuracy: {accuracy:.3f} ({correct_mask.sum()}/{len(all_labels)})")
    
    # Confidence analysis
    print(f"\nConfidence Analysis:")
    print(f"  Average confidence (correct):   {all_confidences[correct_mask].mean():.3f}")
    print(f"  Average confidence (wrong):     {all_confidences[~correct_mask].mean():.3f}")
    print(f"  Confidence gap:                 {all_confidences[correct_mask].mean() - all_confidences[~correct_mask].mean():.3f}")
    
    if all_confidences[correct_mask].mean() > all_confidences[~correct_mask].mean():
        print(f"  ✓ Model is MORE confident when correct (good calibration)")
    else:
        print(f"  ✗ Model is LESS confident when correct (poor calibration)")
    
    # Per-class confidence
    print(f"\nPer-Class Confidence:")
    for i, name in enumerate(class_names):
        class_mask = (all_labels == i)
        if class_mask.sum() > 0:
            class_correct = correct_mask & class_mask
            class_wrong = (~correct_mask) & class_mask
            
            print(f"  {name} (n={class_mask.sum()}):")
            if class_correct.sum() > 0:
                print(f"    Correct predictions: {class_correct.sum()}, avg confidence: {all_confidences[class_correct].mean():.3f}")
            if class_wrong.sum() > 0:
                print(f"    Wrong predictions:   {class_wrong.sum()}, avg confidence: {all_confidences[class_wrong].mean():.3f}")
    
    # Alpha correlation
    corr = np.corrcoef(all_alphas_true, all_alphas_pred)[0, 1]
    print(f"\nAlpha Prediction:")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Mean absolute error: {np.abs(all_alphas_true - all_alphas_pred).mean():.3f}")
    
    # Sample probability distributions
    print(f"\nSample Probability Distributions (first 5 test samples):")
    print(f"  {'True':<12} {'Pred':<12} {'Conf':<6} {'Prob[Conf]':<10} {'Prob[Sub]':<10} {'Prob[Norm]':<10} {'Prob[Act]':<10}")
    print(f"  {'-'*70}")
    for i in range(min(5, len(all_labels))):
        true_name = class_names[all_labels[i]]
        pred_name = class_names[all_preds[i]]
        conf = all_confidences[i]
        probs_i = all_probs[i]
        print(f"  {true_name:<12} {pred_name:<12} {conf:.3f}  {probs_i[0]:.3f}      {probs_i[1]:.3f}      {probs_i[2]:.3f}      {probs_i[3]:.3f}")
    
    return model, all_probs, all_confidences, correct_mask, all_alphas_true, all_alphas_pred


# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

def create_visualizations(all_probs, all_confidences, correct_mask, 
                         all_alphas_true, all_alphas_pred, 
                         output_dir='output_pimt_mini'):
    """Create visualizations showing probability distributions."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot 1: Confidence distribution (correct vs wrong)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_confidences[correct_mask], bins=20, alpha=0.6, label='Correct predictions', color='green')
    ax.hist(all_confidences[~correct_mask], bins=20, alpha=0.6, label='Wrong predictions', color='red')
    ax.set_xlabel('Model Confidence (max probability)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Wrong Predictions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/confidence_distribution.png")
    
    # Plot 2: Alpha prediction scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(all_alphas_true, all_alphas_pred, alpha=0.6, s=50, c=all_confidences, cmap='viridis')
    ax.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True Alpha', fontsize=12)
    ax.set_ylabel('Predicted Alpha', fontsize=12)
    ax.set_title('Alpha Prediction (color = confidence)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    corr = np.corrcoef(all_alphas_true, all_alphas_pred)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(f'{output_dir}/alpha_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/alpha_prediction.png")
    
    # Plot 3: Architecture diagram
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    arch_text = """
    PiMT MINI ARCHITECTURE - Clear Data Flow
    ═══════════════════════════════════════════════════════════════════════
    
    INPUTS:
      Trajectory: (batch, 50, 3)    Physics: (batch, 5)
      [x,y,z positions]             [alpha, confinement, msd, msd_std, vacf]
           ↓                              ↓
    
    TRAJECTORY ENCODER:                  PHYSICS ENCODER:
      Conv1D(3→16, k=5)                   Linear(5→16)
           ↓                              ↓
      ReLU                               ReLU
           ↓                              ↓
      Conv1D(16→32, k=3)                  Linear(16→32)
           ↓                              ↓
      ReLU                               ReLU
           ↓                              ↓
      + Positional Encoding                    ↓
           ↓                              (32-dim embedding)
      Transformer(1 layer, 4 heads)
           ↓
      Global Average Pooling
           ↓
      (32-dim embedding)
    
                    ↓                              ↓
                    +────────── Concatenate ──────────+
                                    ↓
                              (64-dim vector)
                                    ↓
                              Linear(64→32) + ReLU
                                    ↓
                              (32-dim fused)
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              Linear(32→4)                   Linear(32→1)
              Classification                 Alpha Regression
                    ↓                               ↓
              Softmax → Probabilities        Predicted Alpha
              [P(conf), P(sub), P(norm), P(act)]
                    ↓
              Argmax → Predicted Class
    
    KEY FEATURES:
      • Small: 32-dim embeddings, 1 transformer layer
      • Explicit physics features (not learned from data)
      • Probability outputs (not just logits)
      • Multi-task: classification + regression
    
    GRADIENT DESCRIPTON:
      • Updates NEURAL NETWORK WEIGHTS (not nuclei positions)
      • Nuclei positions are FIXED input data
      • Network learns to classify motion patterns
    """
    
    ax.text(0.02, 0.98, arch_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    
    plt.savefig(f'{output_dir}/architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/architecture.png")
    
    print(f"\nAll visualizations saved to: {output_dir}/")


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("="*70)
    print("PiMT MINI - Educational Transformer with Probability Analysis")
    print("="*70)
    
    # Load data
    print("\nLoading real Drosophila tracking data...")
    trajectories = load_real_data(max_traj=100)
    print(f"  Loaded {len(trajectories)} trajectories")
    
    # Train and analyze
    model, all_probs, all_confidences, correct_mask, all_alphas_true, all_alphas_pred = \
        train_and_analyze(trajectories, n_epochs=15, batch_size=16)
    
    # Visualizations
    create_visualizations(all_probs, all_confidences, correct_mask,
                         all_alphas_true, all_alphas_pred)
    
    print("\n" + "="*70)
    print("CRITICAL CLARIFICATION")
    print("="*70)
    print("""
    In particle simulation (atomic_motion.py):
      • Gradient descent updates ATOM POSITIONS
      • Atoms physically move in 3D space
      • Gradient = force on atom
    
    In transformer (PiMT):
      • Gradient descent updates NEURAL NETWORK WEIGHTS
      • Nuclei positions are FIXED input data (they don't move)
      • Gradient = how to adjust weights to reduce prediction error
      • Network LEARNS to classify motion patterns
    
    The transformer does NOT move nuclei. It learns from nuclei motion data.
    """)
    print("="*70)


if __name__ == '__main__':
    main()
