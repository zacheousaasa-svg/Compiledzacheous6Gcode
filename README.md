import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# METRIC PLOTTING STYLING (SCIENTIFIC REPORTS)
# ==========================================
plt.rcParams.update({
    'font.size': 14, 
    'axes.labelsize': 16, 
    'axes.titlesize': 16,
    'xtick.labelsize': 13, 
    'ytick.labelsize': 13, 
    'figure.titlesize': 18,
    'font.family': 'sans-serif', 
    'font.weight': 'bold', 
    'text.usetex': False
})

print("Generating high-visibility manuscript figures...")

# --- FIGURE 1: RIS-assisted SAGIN Architecture ---
fig1, ax1 = plt.subplots(figsize=(9, 7))
bs, ris, user_k, user_other = (0.1, 0.5), (0.45, 0.8), (0.7, 0.3), (0.85, 0.4)
ax1.plot(bs[0], bs[1], '^k', markersize=18, label='Base Station (BS)')
ax1.plot(ris[0], ris[1], 'sb', markersize=18, label=r'RIS ($M$ elements)')
ax1.plot(user_k[0], user_k[1], 'or', markersize=14, label='User $k$')
ax1.plot(user_other[0], user_other[1], 'or', markersize=9, alpha=0.5)
ax1.annotate('', xy=ris, xytext=bs, arrowprops=dict(arrowstyle="-|>", color='blue', lw=3, mutation_scale=15))
ax1.annotate('', xy=user_k, xytext=ris, arrowprops=dict(arrowstyle="-|>", color='blue', lw=3, mutation_scale=15))
ax1.annotate('', xy=user_k, xytext=bs, arrowprops=dict(arrowstyle="-|>", color='gray', linestyle='--', lw=2, mutation_scale=15))
ax1.text(0.24, 0.70, r'$\mathbf{G}$', color='blue', fontsize=18, weight='bold')
ax1.text(0.61, 0.60, r'$\mathbf{h}_{r,k}^H$', color='blue', fontsize=18, weight='bold')
ax1.text(0.38, 0.34, r'$\mathbf{h}_{d,k}^H$', color='black', fontsize=18, weight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.legend(loc='lower left', frameon=False, prop={'weight': 'bold', 'size': 13})
fig1.tight_layout()
fig1.savefig('Fig1_SAGIN_Architecture.png', dpi=1200, bbox_inches='tight')
plt.close(fig1)

# --- FIGURE 2: Lightweight GFM Distillation ---
fig2, ax2 = plt.subplots(figsize=(11, 4.5))
ax2.add_patch(patches.Rectangle((0.05, 0.25), 0.25, 0.5, linewidth=2, edgecolor='black', facecolor='#e6f2ff'))
ax2.add_patch(patches.Rectangle((0.65, 0.25), 0.25, 0.5, linewidth=2, edgecolor='black', facecolor='#fff2e6'))
ax2.text(0.175, 0.5, 'Teacher GFM\n(Cloud-Scale)', ha='center', va='center', weight='bold', fontsize=14)
ax2.text(0.775, 0.5, 'Student GFM\n(Edge-Distilled)', ha='center', va='center', weight='bold', fontsize=14)
ax2.annotate('', xy=(0.65, 0.5), xytext=(0.30, 0.5), arrowprops=dict(arrowstyle="-|>", color='#ff8c00', lw=3.5, mutation_scale=20))
ax2.text(0.475, 0.54, r'KL Divergence $\mathcal{L}_{KL}$' + '\n' + r'Soft Targets (Logits)', ha='center', va='bottom', color='#b36200', weight='bold', fontsize=13)
ax2.annotate('', xy=(0.98, 0.5), xytext=(0.90, 0.5), arrowprops=dict(arrowstyle="-|>", color='black', lw=2, mutation_scale=12))
ax2.text(0.94, 0.54, 'Edge\nDeploy', ha='center', va='bottom', fontsize=11, weight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
fig2.tight_layout()
fig2.savefig('Fig2_GFM_Distillation.png', dpi=1200, bbox_inches='tight')
plt.close(fig2)

# --- FIGURE 3a: PIDT Synchronization Framework ---
fig3, ax3 = plt.subplots(figsize=(11, 6.5))
ax3.add_patch(patches.Rectangle((0.05, 0.20), 0.35, 0.45, linewidth=2.5, edgecolor='#000080', facecolor='#e6f2ff'))
ax3.add_patch(patches.Rectangle((0.58, 0.20), 0.35, 0.45, linewidth=2.5, edgecolor='#006400', facecolor='#e2f0d9'))
ax3.add_patch(patches.Rectangle((0.58, 0.75), 0.35, 0.20, linewidth=2.5, edgecolor='#800000', facecolor='#fce4d6'))
ax3.text(0.225, 0.425, 'Physical 6G\nEnvironment\n(RIS + Sensors)', ha='center', va='center', weight='bold', color='#000080', fontsize=14)
ax3.text(0.755, 0.425, 'Digital Twin\n(Virtual State)', ha='center', va='center', weight='bold', color='#006400', fontsize=14)
ax3.text(0.755, 0.85, 'Distilled GFM\nControl Plane', ha='center', va='center', weight='bold', color='#800000', fontsize=14)
ax3.annotate('', xy=(0.58, 0.46), xytext=(0.40, 0.46), arrowprops=dict(arrowstyle="-|>", color='black', lw=3, mutation_scale=18))
ax3.text(0.49, 0.48, 'Real-time Sensing', ha='center', va='bottom', weight='bold', fontsize=12)
ax3.annotate('', xy=(0.58, 0.32), xytext=(0.40, 0.32), arrowprops=dict(arrowstyle="-|>", color='blue', linestyle='--', lw=2.5, mutation_scale=18))
ax3.text(0.49, 0.24, r'Physics Loss $\mathcal{L}_{phys}$', ha='center', va='top', color='blue', weight='bold', fontsize=13)
ax3.annotate('', xy=(0.755, 0.75), xytext=(0.755, 0.65), arrowprops=dict(arrowstyle="-|>", color='#800000', lw=3, mutation_scale=18))
ax3.text(0.755, 0.70, r'Optimization ($\mathbf{\Phi}, \mathbf{w}$)', ha='center', va='center', backgroundcolor='white', weight='bold', fontsize=12)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1.0)
ax3.axis('off')
fig3.tight_layout()
fig3.savefig('Fig3a_PIDT_Synchronization.png', dpi=1200, bbox_inches='tight')
plt.close(fig3)

# --- FIGURE 3b: Algorithmic Architecture Flowchart ---
fig6, ax6 = plt.subplots(figsize=(11, 8.5))
b_style = dict(linewidth=2.5, edgecolor='black', boxstyle='round,pad=0.4')
ax6.text(0.15, 0.85, "Phase 1: Perception\n\nReal-time CSI Capture\n" + r"($\mathbf{H}_{real}$)", ha='center', va='center', weight='bold', bbox=dict(facecolor='#e6f2ff', **b_style))
ax6.text(0.50, 0.85, r"Phase 2: PIDT Sync" + "\n\nWavefront continuity\n" + r"Enforce $\mathcal{L}_{phys}$ bounds", ha='center', va='center', weight='bold', bbox=dict(facecolor='#e2f0d9', **b_style))
ax6.text(0.50, 0.52, "Phase 3: Edge Inference\n\n3-Layer Distilled Student\nTransformer Pass\n(Latency < 0.9 ms)", ha='center', va='center', weight='bold', bbox=dict(facecolor='#fff2e6', **b_style))
ax6.text(0.88, 0.52, "Self-Attention\nLatent Refinement\n" + r"$\mathbf{Q}, \mathbf{K}, \mathbf{V}$ Block", ha='center', va='center', weight='bold', bbox=dict(facecolor='#fff2e6', linestyle='--', **b_style), fontsize=12)
ax6.text(0.50, 0.22, r"Phase 4: Pareto Check" + "\n\nEvaluate Rate-Energy\n" + r"Threshold ($\eta$)", ha='center', va='center', weight='bold', bbox=dict(facecolor='#fffab3', **b_style))
ax6.text(0.88, 0.22, "Phase 5: Action\n\nDeploy Optimal\n" + r"($\mathbf{w}_k, \mathbf{\Phi}$)" + "\nto Controllers", ha='center', va='center', weight='bold', bbox=dict(facecolor='#fce4d6', **b_style))
arr_t = dict(arrowstyle="-|>", color='black', lw=3, mutation_scale=20)
ax6.annotate('', xy=(0.32, 0.85), xytext=(0.25, 0.85), arrowprops=arr_t)
ax6.annotate('', xy=(0.50, 0.67), xytext=(0.50, 0.73), arrowprops=arr_t)
ax6.annotate('', xy=(0.50, 0.36), xytext=(0.50, 0.40), arrowprops=arr_t)
ax6.annotate('', xy=(0.74, 0.52), xytext=(0.67, 0.52), arrowprops=dict(arrowstyle="-|>", color='red', lw=2.5, mutation_scale=15))
ax6.text(0.705, 0.54, r"$\eta < \eta_{th}$", color='red', ha='center', va='bottom', weight='bold', fontsize=12)
ax6.annotate('', xy=(0.67, 0.56), xytext=(0.74, 0.56), arrowprops=dict(arrowstyle="<-", color='green', lw=2.5, mutation_scale=15))
ax6.text(0.705, 0.58, "Refined", color='green', ha='center', va='bottom', weight='bold', fontsize=12)
ax6.annotate('', xy=(0.73, 0.22), xytext=(0.64, 0.22), arrowprops=arr_t)
ax6.text(0.685, 0.24, r"$\eta \geq \eta_{th}$", color='green', ha='center', va='bottom', weight='bold', fontsize=12)
ax6.set_xlim(0, 1.05)
ax6.set_ylim(0.05, 1.0)
ax6.axis('off')
fig6.tight_layout()
fig6.savefig('Fig3b_Methodology_Flowchart.png', dpi=1200, bbox_inches='tight')
plt.close(fig6)

# --- FIGURE 4: Zero-Shot Adaptability Analysis ---
fig4, ax4 = plt.subplots(figsize=(8, 6))
topologies = np.arange(1, 11)
madrl_baseline = np.array([0.52, 0.55, 0.50, 0.53, 0.56, 0.54, 0.51, 0.57, 0.52, 0.54])
gfm_edge = np.array([0.75, 0.79, 0.72, 0.77, 0.81, 0.78, 0.74, 0.83, 0.75, 0.78])
ax4.plot(topologies, madrl_baseline, marker='o', markersize=8, linewidth=2.5, label='MADRL (Baseline)', color='#1f77b4')
ax4.plot(topologies, gfm_edge, marker='s', markersize=8, linewidth=2.5, label='Proposed GFM-Edge', color='#ff7f0e')
ax4.set_xlabel('Unseen Network Topology Index', weight='bold', labelpad=10)
ax4.set_ylabel('Adaptability Score (Normalized)', weight='bold', labelpad=10)
ax4.set_xticks(topologies)
ax4.set_ylim(0.45, 0.90)
for label in (ax4.get_xticklabels() + ax4.get_yticklabels()): label.set_weight('bold')
ax4.grid(True, which='both', linestyle='-', linewidth=0.75, alpha=0.7)
ax4.legend(loc='lower right', prop={'weight': 'bold', 'size': 13}, framealpha=0.9)
fig4.tight_layout()
fig4.savefig('Fig4_Zero_Shot_Adaptability.png', dpi=1200, bbox_inches='tight')
plt.close(fig4)

# --- FIGURE 5: Rate-Energy Pareto Frontier ---
fig5, ax5 = plt.subplots(figsize=(8, 6))
achievable_rate = np.linspace(2, 10, 100)
baseline_drl_energy = 10.0 - 0.5 * achievable_rate
gfm_edge_energy = 7.77 - 0.485 * achievable_rate
ax5.plot(achievable_rate, baseline_drl_energy, label='Baseline DRL', color='#1f77b4', linewidth=3.0)
ax5.plot(achievable_rate, gfm_edge_energy, label='Proposed GFM-Edge', color='#ff7f0e', linewidth=3.0)
ax5.set_xlabel(r'Achievable Rate $R$ (bps/Hz)', weight='bold', labelpad=10)
ax5.set_ylabel(r'Energy Consumption $E$ (Joule)', weight='bold', labelpad=10)
for label in (ax5.get_xticklabels() + ax5.get_yticklabels()): label.set_weight('bold')
ax5.grid(True, which='both', linestyle='-', linewidth=0.75, alpha=0.7)
ax5.legend(loc='upper right', prop={'weight': 'bold', 'size': 13}, framealpha=0.9)
fig5.tight_layout()
fig5.savefig('Fig5_Pareto_Frontier.png', dpi=1200, bbox_inches='tight')
plt.close(fig5)

print("All 6 figures generated locally and saved to your workspace successfully!")
