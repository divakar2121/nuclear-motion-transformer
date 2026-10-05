"""
Atomic Motion via Gradient Descent — Full Physics-ML Bridge
==============================================================
Explicitly shows:
  - Gradient = Force (arrows on particles)
  - Weight update = Atomic displacement
  - Bias update = Center-of-mass drift
  - Loss reduction = Entropy/enthalpy decrease

Each gradient step IS a physical force-driven atomic jump.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# 1. PHYSICS LOSS WITH BIASES
# ============================================================================

class PhysicsPotential(nn.Module):
    """
    Physics loss with explicit biases:
      - Lennard-Jones pairwise potential
      - Global bias = center-of-mass offset (external field)
      - Per-particle bias = individual energy offset
    """
    
    def __init__(self, n_particles=30, epsilon=1.0, sigma=1.2):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        
        # Bias: external field (shifts all particles uniformly)
        self.external_field = nn.Parameter(torch.zeros(3))
        
        # Bias: per-particle charge (affects interaction strength)
        self.charge = nn.Parameter(torch.ones(n_particles) * 1.0)
    
    def forward(self, positions):
        n = positions.shape[0]
        
        # Apply external field (bias shifts all positions)
        effective_pos = positions + self.external_field
        
        # Pairwise distances
        diff = effective_pos.unsqueeze(1) - effective_pos.unsqueeze(0)
        r = torch.norm(diff, dim=2)
        
        # Mask self-interaction
        r = r + torch.eye(n, device=r.device) * 1e10
        
        # Cutoff
        mask = r < 5.0
        
        # LJ with per-particle charge modulation
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 ** 2
        
        # Charge-weighted interaction: q_i * q_j * V(r)
        q_ij = self.charge.unsqueeze(1) * self.charge.unsqueeze(0)
        energy = 4 * self.epsilon * q_ij * (sr12 - sr6) * mask
        
        return energy.sum() / 2


# ============================================================================
# 2. ATOMIC SYSTEM WITH FULL TRACKING
# ============================================================================

class AtomicSystem:
    """
    Gradient descent = physical forces.
    Tracks gradients, weight updates, and bias updates explicitly.
    """
    
    def __init__(self, n_particles=30, device='cpu'):
        self.n_particles = n_particles
        self.device = device
        
        # Particles = weights (positions are learnable parameters)
        theta = np.random.uniform(0, 2*np.pi, n_particles)
        phi = np.random.uniform(0, np.pi, n_particles)
        r = np.random.uniform(2, 4, n_particles)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        init_pos = np.stack([x, y, z], axis=1)
        self.positions = nn.Parameter(torch.FloatTensor(init_pos).to(device))
        
        # Physics model (with biases)
        self.physics = PhysicsPotential(n_particles=n_particles).to(device)
        
        # Optimizer with separate lr for positions vs biases
        self.optimizer = torch.optim.SGD([
            {'params': [self.positions], 'lr': 0.05, 'momentum': 0.8},
            {'params': [self.physics.external_field], 'lr': 0.02, 'momentum': 0.5},
            {'params': [self.physics.charge], 'lr': 0.01, 'momentum': 0.5},
        ])
        
        # Full history
        self.history = {
            'positions': [],
            'gradients': [],       # d(loss)/d(positions) = force
            'weight_updates': [],  # actual displacement per step
            'bias_field': [],      # external field values
            'bias_grad': [],       # gradient of external field
            'charges': [],         # per-particle charges
            'charge_grad': [],     # gradient of charges
            'loss': [],
            'lr_effective': [],    # effective learning rate * gradient
        }
        
        self.prev_pos = None
    
    def step(self, step_num):
        """One gradient descent step with full tracking."""
        self.optimizer.zero_grad()
        
        loss = self.physics(self.positions)
        loss.backward()
        
        # ---- RECORD GRADIENTS (forces) ----
        pos_grad = self.positions.grad.detach().cpu().numpy().copy()
        field_grad = self.physics.external_field.grad.detach().cpu().numpy().copy()
        charge_grad = self.physics.charge.grad.detach().cpu().numpy().copy()
        
        self.history['gradients'].append(pos_grad)
        self.history['bias_grad'].append(field_grad)
        self.history['charge_grad'].append(charge_grad)
        
        # ---- RECORD PRE-UPDATE STATE ----
        pre_pos = self.positions.detach().cpu().numpy().copy()
        pre_field = self.physics.external_field.detach().cpu().numpy().copy()
        pre_charge = self.physics.charge.detach().cpu().numpy().copy()
        
        # ---- GRADIENT UPDATE (force → displacement) ----
        torch.nn.utils.clip_grad_norm_(
            [self.positions, self.physics.external_field, self.physics.charge],
            max_norm=0.5
        )
        self.optimizer.step()
        
        # ---- RECORD POST-UPDATE STATE ----
        post_pos = self.positions.detach().cpu().numpy().copy()
        post_field = self.physics.external_field.detach().cpu().numpy().copy()
        post_charge = self.physics.charge.detach().cpu().numpy().copy()
        
        # Weight update = displacement
        displacement = post_pos - pre_pos
        field_update = post_field - pre_field
        charge_update = post_charge - pre_charge
        
        self.history['positions'].append(post_pos)
        self.history['weight_updates'].append(displacement)
        self.history['bias_field'].append(post_field)
        self.history['charges'].append(post_charge)
        self.history['loss'].append(loss.item())
        self.history['lr_effective'].append(np.linalg.norm(displacement, axis=1))
        
        # Print detailed info periodically
        if step_num % 25 == 0 or step_num < 3:
            grad_norm = np.linalg.norm(pos_grad, axis=1)
            disp_norm = np.linalg.norm(displacement, axis=1)
            
            print(f"\n  --- Step {step_num:3d} ---")
            print(f"  Loss:          {loss.item():.4f}")
            print(f"  Gradient (force):")
            print(f"    mean |grad|: {grad_norm.mean():.6f}")
            print(f"    max  |grad|: {grad_norm.max():.6f}")
            print(f"    particle 0:  [{pos_grad[0,0]:+.6f}, {pos_grad[0,1]:+.6f}, {pos_grad[0,2]:+.6f}]")
            print(f"    particle 1:  [{pos_grad[1,0]:+.6f}, {pos_grad[1,1]:+.6f}, {pos_grad[1,2]:+.6f}]")
            print(f"  Weight update (displacement):")
            print(f"    mean |Δpos|: {disp_norm.mean():.6f}")
            print(f"    max  |Δpos|: {disp_norm.max():.6f}")
            print(f"    particle 0:  [{displacement[0,0]:+.6f}, {displacement[0,1]:+.6f}, {displacement[0,2]:+.6f}]")
            print(f"    particle 1:  [{displacement[1,0]:+.6f}, {displacement[1,1]:+.6f}, {displacement[1,2]:+.6f}]")
            print(f"  Bias (external field):")
            print(f"    value:       [{post_field[0]:+.6f}, {post_field[1]:+.6f}, {post_field[2]:+.6f}]")
            print(f"    gradient:    [{field_grad[0]:+.6f}, {field_grad[1]:+.6f}, {field_grad[2]:+.6f}]")
            print(f"    update:      [{field_update[0]:+.6f}, {field_update[1]:+.6f}, {field_update[2]:+.6f}]")
            print(f"  Charges (first 5): {post_charge[:5]}")
            print(f"  Charge grad (first 5): {charge_grad[:5]}")
        
        return loss.item()
    
    def simulate(self, n_steps=100):
        """Run full simulation."""
        for i in range(n_steps):
            self.step(i)
        
        return self.history


# ============================================================================
# 3. ANIMATION WITH GRADIENT ARROWS
# ============================================================================

def create_animation(history, output_path='atomic_gradient_motion.gif', fps=12):
    """
    Animation showing:
      Left:  3D particles with gradient arrows (force vectors)
      Right: Loss curve + gradient magnitude + displacement magnitude
    """
    
    positions = np.array(history['positions'])
    gradients = np.array(history['gradients'])
    updates = np.array(history['weight_updates'])
    loss_hist = np.array(history['loss'])
    bias_field = np.array(history['bias_field'])
    lr_eff = np.array(history['lr_effective'])
    
    n_steps, n_particles, _ = positions.shape
    
    # Subsample for cleaner arrows (show every 5th particle's gradient)
    show_idx = np.arange(0, n_particles, 5)
    
    fig = plt.figure(figsize=(18, 8))
    
    # ---- LEFT: 3D with gradient arrows ----
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.view_init(elev=25, azim=45)
    
    all_pos = positions.reshape(-1, 3)
    bounds = np.percentile(all_pos, [5, 95], axis=0)
    margin = (bounds[1] - bounds[0]) * 0.15
    ax3d.set_xlim(bounds[0, 0] - margin[0], bounds[1, 0] + margin[0])
    ax3d.set_ylim(bounds[0, 1] - margin[1], bounds[1, 1] + margin[1])
    ax3d.set_zlim(bounds[0, 2] - margin[2], bounds[1, 2] + margin[2])
    ax3d.set_xlabel('X (μm)')
    ax3d.set_ylabel('Y (μm)')
    ax3d.set_zlabel('Z (μm)')
    ax3d.set_title('Particles + Gradient Arrows (Force Vectors)', fontsize=13, fontweight='bold')
    
    # ---- RIGHT: Metrics ----
    ax_loss = fig.add_subplot(222)
    ax_loss.set_ylabel('Loss (Potential Energy)', fontsize=11)
    ax_loss.set_title('Energy Minimization', fontsize=12, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlim(0, n_steps)
    
    ax_grad = fig.add_subplot(224)
    ax_grad.set_xlabel('Timestep', fontsize=11)
    ax_grad.set_ylabel('|∇L| (Force)', fontsize=11)
    ax_grad.set_title('Gradient Magnitude & Displacement', fontsize=12, fontweight='bold')
    ax_grad.grid(True, alpha=0.3)
    ax_grad.set_xlim(0, n_steps)
    
    # Initialize elements
    particles = ax3d.scatter([], [], [], c='steelblue', s=60, alpha=0.8, depthshade=False)
    
    # Gradient arrows (quiver)
    arrow_scale = 5.0  # Scale factor for visibility
    quiver = None
    
    loss_line, = ax_loss.plot([], [], 'r-', linewidth=2, label='Loss')
    ax_loss.legend(loc='upper right')
    
    grad_mean_line, = ax_grad.plot([], [], 'b-', linewidth=2, label='Mean |∇L|')
    disp_mean_line, = ax_grad.plot([], [], 'g--', linewidth=2, label='Mean |Δpos|')
    ax_grad.legend(loc='upper right')
    
    # Pre-compute gradient norms
    grad_norms = np.linalg.norm(gradients, axis=2).mean(axis=1)
    disp_norms = lr_eff.mean(axis=1)
    
    ax_grad.set_ylim(0, max(grad_norms.max(), disp_norms.max()) * 1.2)
    ax_loss.set_ylim(min(loss_hist) * 0.9, max(loss_hist) * 1.1)
    
    def update(frame):
        nonlocal quiver
        
        pos = positions[frame]
        
        # Remove old quiver
        if quiver is not None:
            quiver.remove()
            quiver = None
        
        # Update particles
        particles._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # Add gradient arrows (negative gradient = force direction)
        grad = gradients[frame]
        # Show subset of particles for clarity
        if frame < n_steps:
            u = -grad[show_idx, 0] * arrow_scale
            v = -grad[show_idx, 1] * arrow_scale
            w = -grad[show_idx, 2] * arrow_scale
            
            quiver = ax3d.quiver(
                pos[show_idx, 0], pos[show_idx, 1], pos[show_idx, 2],
                u, v, w,
                color='red', alpha=0.7, arrow_length_ratio=0.15, linewidth=1.5
            )
        
        # Update loss curve
        loss_line.set_data(range(frame + 1), loss_hist[:frame + 1])
        
        # Update gradient/displacement curves
        grad_mean_line.set_data(range(frame + 1), grad_norms[:frame + 1])
        disp_mean_line.set_data(range(frame + 1), disp_norms[:frame + 1])
        
        return particles, loss_line, grad_mean_line, disp_mean_line
    
    print(f"\nRendering {n_steps} frames...")
    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000//fps, blit=False)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()
    print(f"  Saved: {output_path}")
    
    return anim


# ============================================================================
# 4. STATIC SUMMARY PLOTS
# ============================================================================

def create_summary_plots(history, output_dir='.'):
    """Create detailed static plots showing gradient/update evolution."""
    
    positions = np.array(history['positions'])
    gradients = np.array(history['gradients'])
    updates = np.array(history['weight_updates'])
    loss_hist = np.array(history['loss'])
    bias_field = np.array(history['bias_field'])
    charges = np.array(history['charges'])
    lr_eff = np.array(history['lr_effective'])
    
    n_steps, n_particles, _ = positions.shape
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # [0,0] Initial positions
    ax = axes[0, 0]
    ax.scatter(positions[0, :, 0], positions[0, :, 1], c='blue', s=40, alpha=0.7)
    ax.set_title('Initial Positions (t=0)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # [0,1] Final positions
    ax = axes[0, 1]
    ax.scatter(positions[-1, :, 0], positions[-1, :, 1], c='red', s=40, alpha=0.7)
    ax.set_title('Final Positions (t=100)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # [0,2] Loss
    ax = axes[0, 2]
    ax.plot(loss_hist, 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss = Potential Energy', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # [1,0] Gradient magnitude over time
    ax = axes[1, 0]
    grad_norms = np.linalg.norm(gradients, axis=2).mean(axis=1)
    ax.plot(grad_norms, 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean |∇L|')
    ax.set_title('Gradient Magnitude (Force)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # [1,1] Displacement magnitude over time
    ax = axes[1, 1]
    disp_norms = lr_eff.mean(axis=1)
    ax.plot(disp_norms, 'g-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean |Δpos|')
    ax.set_title('Weight Update (Displacement)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # [1,2] Bias (external field) over time
    ax = axes[1, 2]
    ax.plot(bias_field[:, 0], 'r-', label='Field X')
    ax.plot(bias_field[:, 1], 'g-', label='Field Y')
    ax.plot(bias_field[:, 2], 'b-', label='Field Z')
    ax.set_xlabel('Step')
    ax.set_ylabel('Bias Value')
    ax.set_title('Bias Update (External Field)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [2,0] Per-particle gradient at step 0
    ax = axes[2, 0]
    g0 = gradients[0]
    ax.quiver(positions[0, :, 0], positions[0, :, 1],
             -g0[:, 0], -g0[:, 1],
             color='red', alpha=0.6, scale=20)
    ax.scatter(positions[0, :, 0], positions[0, :, 1], c='blue', s=30)
    ax.set_title('Force Vectors at t=0', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # [2,1] Per-particle gradient at step 50
    ax = axes[2, 1]
    mid = n_steps // 2
    gm = gradients[mid]
    ax.quiver(positions[mid, :, 0], positions[mid, :, 1],
             -gm[:, 0], -gm[:, 1],
             color='orange', alpha=0.6, scale=20)
    ax.scatter(positions[mid, :, 0], positions[mid, :, 1], c='green', s=30)
    ax.set_title(f'Force Vectors at t={mid}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # [2,2] Charges over time (first 5 particles)
    ax = axes[2, 2]
    for i in range(min(5, n_particles)):
        ax.plot(charges[:, i], label=f'q_{i}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Charge')
    ax.set_title('Per-Particle Charge (Bias)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Descent = Physical Force\n'
                 'Weights = Positions | ∇L = Force | ΔW = Displacement | Bias = External Field',
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    out_path = f'{output_dir}/gradient_physics_summary.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot: {out_path}")


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  Atomic Motion via Gradient Descent — Full Physics-ML Bridge")
    print("=" * 70)
    print("\n  PHYSICS ↔ ML MAPPING:")
    print("    Positions (weights)  ←→  Atom coordinates")
    print("    Gradient (∇L)        ←→  Force on each atom")
    print("    Weight update (ΔW)   ←→  Atomic displacement")
    print("    Bias (external field)←→  External potential")
    print("    Loss reduction       ←→  Entropy/enthalpy decrease")
    print("    Learning rate        ←→  Time step / mobility")
    print("    Momentum             ←→  Inertia")
    print("=" * 70)
    
    print("\n[1] Initializing system...")
    system = AtomicSystem(n_particles=30, device='cpu')
    
    print("\n[2] Simulating (100 steps)...\n")
    history = system.simulate(n_steps=100)
    
    # Final summary
    loss_hist = np.array(history['loss'])
    positions = np.array(history['positions'])
    gradients = np.array(history['gradients'])
    updates = np.array(history['weight_updates'])
    
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Loss: {loss_hist[0]:.2f} → {loss_hist[-1]:.2f} "
          f"(reduction: {(1-loss_hist[-1]/loss_hist[0])*100:.1f}%)")
    print(f"  Final gradient: mean |∇L| = {np.linalg.norm(gradients[-1], axis=1).mean():.6f}")
    print(f"  Final displacement: mean |Δpos| = {np.linalg.norm(updates[-1], axis=1).mean():.6f}")
    print(f"  Final bias field: {history['bias_field'][-1]}")
    
    print("\n[3] Creating animation...")
    create_animation(history, output_path='atomic_gradient_motion.gif', fps=12)
    
    print("\n[4] Creating summary plots...")
    create_summary_plots(history, output_dir='.')
    
    # Export raw data
    print("\n[5] Exporting raw data...")
    np.save('gradient_history.npy', gradients)
    np.save('position_history.npy', positions)
    np.save('update_history.npy', updates)
    np.save('loss_history.npy', loss_hist)
    np.save('bias_history.npy', np.array(history['bias_field']))
    np.save('charge_history.npy', np.array(history['charges']))
    print("  Saved: gradient_history.npy, position_history.npy, update_history.npy,")
    print("         loss_history.npy, bias_history.npy, charge_history.npy")
    
    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
