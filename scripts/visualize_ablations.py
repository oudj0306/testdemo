import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Find a good font
available = [f.name for f in fm.fontManager.ttflist]
font_name = None
for f in ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']:
    if f in available:
        font_name = f
        break
if font_name is None:
    font_name = available[0]
print(f"Using font: {font_name}")

plt.rcParams.update({
    'font.family': font_name,
    'font.size': 18,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 28,
    'axes.unicode_minus': False,
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

with open('data/results/ablation_results.json') as f:
    data = json.load(f)

# --- Build clean data ---
fine_data = {
    'labels': ['w/ MLP refine\n(baseline)', 'w/o refine\n(coarse only)'],
    'auc5': [data['XFeat-star w/ refine (baseline)']['auc']['auc@5'] * 100,
             data['XFeat-star w/o refine (coarse only)']['auc']['auc@5'] * 100],
    'auc10': [data['XFeat-star w/ refine (baseline)']['auc']['auc@10'] * 100,
              data['XFeat-star w/o refine (coarse only)']['auc']['auc@10'] * 100],
    'auc20': [data['XFeat-star w/ refine (baseline)']['auc']['auc@20'] * 100,
              data['XFeat-star w/o refine (coarse only)']['auc']['auc@20'] * 100],
}

mnn_data = {
    'labels': ['MNN\n(baseline)', 'Single-NN\n(no mutual)'],
    'auc5': [data['XFeat MNN (baseline)']['auc']['auc@5'] * 100,
             data['XFeat single-NN (no mutual)']['auc']['auc@5'] * 100],
    'auc10': [data['XFeat MNN (baseline)']['auc']['auc@10'] * 100,
              data['XFeat single-NN (no mutual)']['auc']['auc@10'] * 100],
    'auc20': [data['XFeat MNN (baseline)']['auc']['auc@20'] * 100,
              data['XFeat single-NN (no mutual)']['auc']['auc@20'] * 100],
}

dim_data = {
    'dims': [16, 32, 48, 64],
    'auc5': [data['XFeat 16D']['auc']['auc@5'] * 100,
             data['XFeat 32D']['auc']['auc@5'] * 100,
             data['XFeat 48D']['auc']['auc@5'] * 100,
             data['XFeat 64D (baseline)']['auc']['auc@5'] * 100],
    'auc10': [data['XFeat 16D']['auc']['auc@10'] * 100,
              data['XFeat 32D']['auc']['auc@10'] * 100,
              data['XFeat 48D']['auc']['auc@10'] * 100,
              data['XFeat 64D (baseline)']['auc']['auc@10'] * 100],
    'auc20': [data['XFeat 16D']['auc']['auc@20'] * 100,
              data['XFeat 32D']['auc']['auc@20'] * 100,
              data['XFeat 48D']['auc']['auc@20'] * 100,
              data['XFeat 64D (baseline)']['auc']['auc@20'] * 100],
}

degradation_data = {
    'labels': ['Fine Matcher\n(MLP removed)', 'Matching Dir\n(MNN removed)',
               'Desc Dim\n(64D -> 48D)', 'Desc Dim\n(64D -> 32D)',
               'Desc Dim\n(64D -> 16D)'],
    'vals': [],
    'colors': ['#C62828', '#C62828', '#2E7D32', '#E65100', '#C62828'],
}

# Calculate degradations
degradation_data['vals'] = [
    round((fine_data['auc5'][1] - fine_data['auc5'][0]) / fine_data['auc5'][0] * 100, 1),
    round((mnn_data['auc5'][1] - mnn_data['auc5'][0]) / mnn_data['auc5'][0] * 100, 1),
    round((dim_data['auc5'][2] - dim_data['auc5'][3]) / dim_data['auc5'][3] * 100, 1),
    round((dim_data['auc5'][1] - dim_data['auc5'][3]) / dim_data['auc5'][3] * 100, 1),
    round((dim_data['auc5'][0] - dim_data['auc5'][3]) / dim_data['auc5'][3] * 100, 1),
]

os.makedirs('data/results', exist_ok=True)

# ============================================================
# FIGURE 1: Fine Matcher + Matching Direction (side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

def plot_double_bar(ax, data, title):
    x = [0, 1]
    w = 0.25
    c5, c10, c20 = '#1565C0', '#42A5F5', '#90CAF9'

    b1 = ax.bar([i - w for i in x], data['auc5'], w, color=c5, label='AUC@5', edgecolor='white', lw=1)
    b2 = ax.bar(x, data['auc10'], w, color=c10, label='AUC@10', edgecolor='white', lw=1)
    b3 = ax.bar([i + w for i in x], data['auc20'], w, color=c20, label='AUC@20', edgecolor='white', lw=1)

    for i, (v5, v10, v20) in enumerate(zip(data['auc5'], data['auc10'], data['auc20'])):
        ax.text(i - w, v5 + 2, f'{v5:.1f}', ha='center', fontsize=18, fontweight='bold', color='#0D47A1')
        ax.text(i, v10 + 2, f'{v10:.1f}', ha='center', fontsize=18, fontweight='bold', color='#1565C0')
        ax.text(i + w, v20 + 2, f'{v20:.1f}', ha='center', fontsize=18, fontweight='bold', color='#1976D2')

    # Degradation arrow
    drop = (data['auc5'][0] - data['auc5'][1]) / data['auc5'][0] * 100
    mid_x = 0.5
    mid_y = (data['auc5'][0] + data['auc5'][1]) / 2
    ax.annotate(f'  -{drop:.1f}%  ', xy=(0.75, data['auc5'][1]),
                xytext=(1.2, mid_y + 10),
                fontsize=22, fontweight='bold', color='#C62828',
                arrowprops=dict(arrowstyle='fancy', color='#C62828', lw=2.5,
                               connectionstyle='arc3,rad=0.2'))

    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'], fontsize=20)
    ax.set_ylabel('AUC (%)', fontsize=22)
    ax.set_title(title, fontsize=26, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=16, framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plot_double_bar(ax1, fine_data, 'Ablation 1: Fine Matcher (MLP) Contribution')
plot_double_bar(ax2, mnn_data, 'Ablation 2: Matching Direction (MNN vs NN)')

plt.tight_layout()
plt.savefig('data/results/ablation_fig1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: ablation_fig1.png (Fine Matcher + Matching Direction)")

# ============================================================
# FIGURE 2: Descriptor Dimensionality (standalone, big)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 8))

dims = dim_data['dims']
ax.plot(dims, dim_data['auc5'], 'D-', color='#1565C0', linewidth=4, markersize=16, label='AUC@5', zorder=5)
ax.plot(dims, dim_data['auc10'], 's-', color='#42A5F5', linewidth=4, markersize=16, label='AUC@10', zorder=4)
ax.plot(dims, dim_data['auc20'], 'o--', color='#90CAF9', linewidth=4, markersize=16, label='AUC@20', zorder=3)

# Value labels
for d, v5, v10, v20 in zip(dims, dim_data['auc5'], dim_data['auc10'], dim_data['auc20']):
    offset = 12 if d < 50 else -25
    ax.annotate(f'{v5:.1f}', (d, v5), textcoords="offset points", xytext=(offset, 6),
                fontsize=20, fontweight='bold', color='#0D47A1')
    ax.annotate(f'{v10:.1f}', (d, v10), textcoords="offset points", xytext=(offset, -18),
                fontsize=16, color='#1565C0')
    ax.annotate(f'{v20:.1f}', (d, v20), textcoords="offset points", xytext=(offset, -35),
                fontsize=16, color='#1976D2')

# Degradation callout boxes
bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', lw=2, alpha=0.9)
ax.annotate('  -6.3%  \n(near-lossless)', xy=(48, dim_data['auc5'][2]),
            xytext=(52, dim_data['auc5'][2] + 15),
            fontsize=20, fontweight='bold', color='#2E7D32',
            bbox=bbox_props, arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5))

bbox_props2 = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', lw=2, alpha=0.9)
ax.annotate('  -11.8%  \n(acceptable)', xy=(32, dim_data['auc5'][1]),
            xytext=(36, dim_data['auc5'][1] + 15),
            fontsize=20, fontweight='bold', color='#E65100',
            bbox=bbox_props2, arrowprops=dict(arrowstyle='->', color='#E65100', lw=2.5))

bbox_props3 = dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#C62828', lw=2, alpha=0.9)
ax.annotate('  -38.4%  \n(collapse)', xy=(16, dim_data['auc5'][0]),
            xytext=(20, dim_data['auc5'][0] + 15),
            fontsize=20, fontweight='bold', color='#C62828',
            bbox=bbox_props3, arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5))

ax.set_xlabel('Descriptor Dimension', fontsize=24)
ax.set_ylabel('AUC (%)', fontsize=24)
ax.set_title('Ablation 3: Descriptor Dimensionality (64D -> 16D)', fontsize=28, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=18, framealpha=0.9)
ax.set_xticks(dims)
ax.set_xticklabels([f'{d}D' for d in dims], fontsize=20)
ax.set_ylim(25, 100)
ax.set_xlim(10, 72)
ax.grid(alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('data/results/ablation_fig2.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: ablation_fig2.png (Descriptor Dimensionality)")

# ============================================================
# FIGURE 3: AUC@5 Degradation Summary (standalone, big)
# ============================================================
fig, ax = plt.subplots(figsize=(18, 8))

x_pos = range(len(degradation_data['labels']))
bars = ax.bar(x_pos, degradation_data['vals'], color=degradation_data['colors'],
              edgecolor='white', lw=2, width=0.55)

for i, (bar, deg) in enumerate(zip(bars, degradation_data['vals'])):
    y_pos = deg + 1.5 if deg > -15 else deg - 5
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{deg}%', ha='center',
            fontsize=24, fontweight='bold', color=degradation_data['colors'][i])

# Reference lines with labels
for y, label, color in [(-6.3, 'Near-lossless', '#2E7D32'),
                          (-11.8, 'Acceptable', '#E65100'),
                          (-24.0, 'Severe', '#C62828'),
                          (-36.7, 'Critical', '#C62828')]:
    ax.axhline(y=y, color=color, linestyle='--', alpha=0.3, lw=1.5)
    ax.text(4.7, y, f'  {label}', fontsize=13, color=color, va='center', fontstyle='italic')

ax.set_xticks(x_pos)
ax.set_xticklabels(degradation_data['labels'], fontsize=18)
ax.set_ylabel('AUC@5 Change (%)', fontsize=24)
ax.set_title('Ablation Summary: AUC@5 Degradation by Component', fontsize=28, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.2)
ax.set_ylim(-45, 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('data/results/ablation_fig3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: ablation_fig3.png (Degradation Summary)")

print("\nAll 3 figures generated successfully!")
