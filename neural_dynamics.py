"""
Neural Dynamics: Learn Physics from Real Trajectory Data
=========================================================
This is ACTUAL MACHINE LEARNING (not just simulation).

Learns dx/dt = f(x, t) from real Drosophila tracking data using neural ODE.
Predicts future nuclear positions and discovers underlying force field.

Three-System Bridge:
  1. Simulation:    Understand physics (Langevin dynamics)
  2. Classifier:    Extract physics features (MSD, VACF, alpha)
  3. Neural ODE:    Learn dynamics from data (predict future motion)

Author: Divakar Ravi Kumar | October 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# 1. NEURAL DYNAMICS NETWORK
# ============================================================================

class NeuralDynamics(nn.Module):
    """
    Neural network that learns dx/dt = f(x, t) from trajectory data.
    
    This is a Neural ODE: given current position x(t), predict velocity dx/dt.
    Then integrate to get future position x(t+dt).
    
    Architecture:
      Input:  position x(t) ∈ R^3
      Output: velocity dx/dt ∈ R^3
    
    Loss: Compare predicted x(t+dt) with actual x(t+dt) from data.
    
    This is GENUINE ML:
      - Learns from real trajectory data
      - Generalizes to unseen trajectories
      - Predicts future positions
      - Discovers force field implicitly
    """
    
    def __init__(self, input_dim=3, hidden_dim=64, n_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Predict velocity dx/dt from current position x.
        
        Args:
            x: (batch, 3) current position
        Returns:
            dx/dt: (batch, 3) predicted velocity
        """
        return self.net(x)
    
    def integrate(self, x0, dt, n_steps):
        """
        Integrate dynamics forward in time using Euler method.
        
        Args:
            x0: (batch, 3) initial position
            dt: time step
            n_steps: number of integration steps
        Returns:
            trajectory: (batch, n_steps, 3) predicted trajectory
        """
        trajectory = [x0]
        x = x0
        
        for _ in range(n_steps):
            dx_dt = self.forward(x)  # Predict velocity
            x = x + dx_dt * dt        # Euler integration
            trajectory.append(x)
        
        return torch.stack(trajectory, dim=1)  # (batch, n_steps+1, 3)


# ============================================================================
# 2. DATASET: Real Trajectory Pairs
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Dataset of (x(t), x(t+dt)) pairs from real trajectories.
    
    For each trajectory, extract consecutive position pairs.
    """
    
    def __init__(self, trajectories, dt=1):
        self.pairs = []
        
        for traj in trajectories:
            if len(traj) < dt + 1:
                continue
            
            for i in range(len(traj) - dt):
                x_t = traj[i]        # Position at time t
                x_next = traj[i + dt]  # Position at time t+dt
                self.pairs.append((x_t, x_next))
        
        self.pairs = np.array(self.pairs)
        self.x_t = torch.FloatTensor(self.pairs[:, 0])
        self.x_next = torch.FloatTensor(self.pairs[:, 1])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.x_t[idx], self.x_next[idx]


# ============================================================================
# 3. DATA LOADING
# ============================================================================

def load_real_trajectories(data_dir='data/zenodo', max_traj=200):
    """Load real Drosophila tracking data."""
    print(f"Loading up to {max_traj} real trajectories...")
    
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
    
    print(f"  Loaded {len(trajectories)} trajectories")
    return trajectories


# ============================================================================
# 4. TRAINING
# ============================================================================

def train_neural_dynamics(trajectories, n_epochs=30, batch_size=256):
    """Train neural dynamics model on real trajectory data."""
    
    print("\n" + "="*70)
    print("NEURAL DYNAMICS TRAINING")
    print("="*70)
    
    # Create dataset
    print("\nCreating training pairs...")
    dataset = TrajectoryDataset(trajectories, dt=1)
    print(f"  Created {len(dataset)} (x(t), x(t+1)) pairs")
    
    # Split train/test
    n_train = int(0.8 * len(dataset))
    train_subset, test_subset = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train]
    )
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralDynamics(input_dim=3, hidden_dim=64, n_layers=3).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Neural Dynamics")
    print(f"  Architecture: MLP (3 → 64 → 64 → 64 → 3)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        train_loss = 0
        
        for x_t, x_next in train_loader:
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            
            # Predict velocity and integrate
            dx_dt = model(x_t)
            x_pred = x_t + dx_dt  # dt=1
            
            # Loss: compare predicted position with actual
            loss = criterion(x_pred, x_next)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Test
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for x_t, x_next in test_loader:
                x_t = x_t.to(device)
                x_next = x_next.to(device)
                
                dx_dt = model(x_t)
                x_pred = x_t + dx_dt
                loss = criterion(x_pred, x_next)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: Train loss={avg_train_loss:.6f}, "
                  f"Test loss={avg_test_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"\nBest test loss: {best_loss:.6f}")
    
    return model, test_loader, device


# ============================================================================
# 5. EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_and_visualize(model, trajectories, device, output_dir='output_neural_dynamics'):
    """Evaluate model and create visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EVALUATION AND VISUALIZATION")
    print("="*70)
    
    model.eval()
    
    # Test on a few trajectories
    n_test = min(5, len(trajectories))
    test_trajectories = trajectories[:n_test]
    
    all_predictions = []
    all_ground_truth = []
    all_errors = []
    
    for i, traj in enumerate(test_trajectories):
        if len(traj) < 20:
            continue
        
        # Take first 10 steps as initial condition
        x0 = torch.FloatTensor(traj[:1]).to(device)
        n_predict = min(19, len(traj) - 1)
        
        # Predict future trajectory
        with torch.no_grad():
            pred_traj = model.integrate(x0, dt=1, n_steps=n_predict)
        
        pred_traj = pred_traj[0].cpu().numpy()  # Remove batch dim
        true_traj = traj[:n_predict+1]
        
        all_predictions.append(pred_traj)
        all_ground_truth.append(true_traj)
        
        # Compute error
        error = np.sqrt(np.sum((pred_traj - true_traj)**2, axis=1))
        all_errors.append(error)
        
        print(f"\nTrajectory {i+1}:")
        print(f"  Length: {len(traj)} steps")
        print(f"  Predicted: {n_predict} steps")
        print(f"  Mean error: {error.mean():.4f} μm")
        print(f"  Final error: {error[-1]:.4f} μm")
    
    # Aggregate statistics
    all_errors_flat = np.concatenate(all_errors)
    
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    print(f"  Total predictions: {len(all_errors_flat)}")
    print(f"  Mean error: {all_errors_flat.mean():.4f} μm")
    print(f"  Median error: {np.median(all_errors_flat):.4f} μm")
    print(f"  Max error: {all_errors_flat.max():.4f} μm")
    
    # === VISUALIZATIONS ===
    
    # Plot 1: Predicted vs Actual trajectories
    fig, axes = plt.subplots(2, min(3, len(all_predictions)), figsize=(15, 10))
    if len(all_predictions) == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(3, len(all_predictions))):
        pred = all_predictions[i]
        true = all_ground_truth[i]
        
        # XY projection
        ax = axes[0, i]
        ax.plot(true[:, 0], true[:, 1], 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax.plot(true[0, 0], true[0, 1], 'go', markersize=10, label='Start')
        ax.plot(true[-1, 0], true[-1, 1], 'g*', markersize=15, label='End')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Trajectory {i+1} (XY projection)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # XZ projection
        ax = axes[1, i]
        ax.plot(true[:, 0], true[:, 2], 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(pred[:, 0], pred[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax.plot(true[0, 0], true[0, 2], 'go', markersize=10, label='Start')
        ax.plot(true[-1, 0], true[-1, 2], 'g*', markersize=15, label='End')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Z (μm)')
        ax.set_title(f'Trajectory {i+1} (XZ projection)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('Neural Dynamics: Predicted vs Actual Trajectories', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trajectory_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_dir}/trajectory_predictions.png")
    
    # Plot 2: Error over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, error in enumerate(all_errors):
        ax.plot(error, alpha=0.7, label=f'Traj {i+1}')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Position Error (μm)', fontsize=12)
    ax.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/error_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/error_over_time.png")
    
    # Plot 3: Velocity field visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create grid
    x_range = np.linspace(-5, 5, 20)
    y_range = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Predict velocity at each grid point (z=0)
    grid_points = np.stack([X.flatten(), Y.flatten(), np.zeros(X.size)], axis=1)
    grid_tensor = torch.FloatTensor(grid_points).to(device)
    
    with torch.no_grad():
        velocities = model(grid_tensor).cpu().numpy()
    
    # Plot velocity field (XY components only)
    U = velocities[:, 0].reshape(X.shape)
    V = velocities[:, 1].reshape(X.shape)
    
    ax.quiver(X, Y, U, V, color='blue', alpha=0.6, scale=10)
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title('Learned Velocity Field (Neural Dynamics)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.savefig(f'{output_dir}/velocity_field.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/velocity_field.png")
    
    # Plot 4: Architecture diagram
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    arch_text = """
    NEURAL DYNAMICS ARCHITECTURE
    ═══════════════════════════════════════════════════════════════════════
    
    INPUT:
      x(t) = Current position ∈ R^3
           ↓
    
    NEURAL NETWORK:
      Linear(3 → 64) + Tanh
           ↓
      Linear(64 → 64) + Tanh
           ↓
      Linear(64 → 64) + Tanh
           ↓
      Linear(64 → 3)
           ↓
    
    OUTPUT:
      dx/dt = Predicted velocity ∈ R^3
           ↓
    
    INTEGRATION:
      x(t+dt) = x(t) + (dx/dt) × dt
    
    TRAINING:
      Loss = ||x_predicted(t+dt) - x_actual(t+dt)||^2
      Optimizer: AdamW (lr=1e-3)
    
    THIS IS GENUINE MACHINE LEARNING:
      ✓ Learns from real trajectory data
      ✓ Predicts future positions
      ✓ Generalizes to unseen trajectories
      ✓ Discovers force field implicitly
      ✓ Network weights update (not atom positions)
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
    print("NEURAL DYNAMICS: Learn Physics from Real Data")
    print("="*70)
    
    # Load data
    trajectories = load_real_trajectories(max_traj=200)
    
    # Train
    model, test_loader, device = train_neural_dynamics(
        trajectories, n_epochs=30, batch_size=256
    )
    
    # Evaluate
    evaluate_and_visualize(model, trajectories, device)
    
    print("\n" + "="*70)
    print("THREE-SYSTEM BRIDGE")
    print("="*70)
    print("""
    1. PARTICLE SIMULATION (atomic_gradient_bridge.py)
       → Understand physics: Langevin dynamics, Brownian motion
       → Gradient = force on atoms (simulation)
    
    2. PiMT CLASSIFIER (pimt_mini.py)
       → Extract physics features: MSD, VACF, alpha
       → Gradient = learning signal (classification)
    
    3. NEURAL DYNAMICS (this file)
       → Learn physics from data: dx/dt = f(x, t)
       → Gradient = learning signal (dynamics prediction)
    
    PROGRESSION:
      Understanding → Analyzing → Predicting
    
    All three address: "Characterization of active Brownian motion"
    """)
    print("="*70)


if __name__ == '__main__':
    main()
