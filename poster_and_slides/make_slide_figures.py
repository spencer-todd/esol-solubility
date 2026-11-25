# Draws slide-sized copies of five poster figures, from the same result tables as make_poster_figures.py.
# The poster figures are made for a 48 in poster, so on a 13.3 in slide some labels came out too small to read on a
# projector (the figure review of Nov 22). The data, colours and labels are the same as on the poster. Changes:
#   s_split: the two grids are stacked, so the figure fills its half of the slide (labels 20 pt instead of about 10 pt)
#   s_importance: drawn at the size it has on the slide, so the input names are 17 pt instead of about 14 pt,
#     with the value of each bar printed (as on the poster)
#   s_parity: larger legend and axis labels
#   s_models: Figure 4 (test RMSE and the splits won against kNN) at its size on slide 6, labels 16 pt or more
#   s_logp, s_parity: the scatter points are saved at 600 dpi in the PDF (the default 100 dpi looked soft)
# No model is fitted here. The poster figures in results/figures/poster are not changed.
# Run from the esol-solubility folder, after the notebooks:  uv run python poster_and_slides/make_slide_figures.py
# The figures go to results/figures/slides (PNG, and a PDF copy that stays sharp in the slides PDF).
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')   # the esol-solubility folder
TAB = os.path.join(REPO, 'results', 'tables')
OUT = os.path.join(REPO, 'results', 'figures', 'slides')
os.makedirs(OUT, exist_ok=True)

# the poster's colours and style
NAVY, MID, LIGHT, BRIGHT = '#002A5C', '#5B7FB0', '#C9D6E8', '#3D8FD1'
DARK = '#1A1A1A'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 24, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 1.8, 'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'axes.edgecolor': DARK,
                     'text.color': DARK, 'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=600, facecolor='white')
    # vector copy for the PDF (no creation date, so every run gives the same file)
    fig.savefig(os.path.join(OUT, name.replace('.png', '.pdf')), dpi=600, facecolor='white', metadata={'CreationDate': None})
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------- what a scaffold split is (a schematic, no data)
# Source: the poster's Figure 1 (make_poster_figures.py, section 6) at slide size
fig, axes = plt.subplots(2, 1, figsize=(5.7, 5.75))   # tall enough to leave space between the rows of molecules (the molecules stay the same size)
families = ['family A', 'family B', 'family C', 'family D']
random_test = {(0, 3), (1, 0), (1, 4), (2, 2), (3, 1)}
for ax, title, test in [(axes[0], 'Random split', random_test), (axes[1], 'Scaffold split', {(3, k) for k in range(5)})]:
    for row in range(4):
        for k in range(5):
            is_test = (row, k) in test
            ax.scatter(k, 3 - row, s=420, marker='s' if is_test else 'o', color=BRIGHT if is_test else LIGHT,
                       edgecolor=NAVY, linewidth=2)
        ax.text(-0.75, 3 - row, families[row], ha='right', va='center', fontsize=20)
    ax.set_xlim(-2.9, 4.5)
    # title centred over the 5 columns of molecules (x = 0 to 4). The family labels are left out of the centring
    ax.set_title(title, fontsize=24, fontweight='bold', color=NAVY, pad=6, x=(2 + 2.9) / (4.5 + 2.9))
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
training = axes[0].scatter([], [], s=300, marker='o', color=LIGHT, edgecolor=NAVY, linewidth=2, label='training molecule')
test = axes[0].scatter([], [], s=300, marker='s', color=BRIGHT, edgecolor=NAVY, linewidth=2, label='test molecule')
fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.2)
fig.legend(handles=[training, test], frameon=False, loc='lower center', ncol=2, fontsize=20, handletextpad=0.3,
           columnspacing=1.5, bbox_to_anchor=(0.5, 0))
save(fig, 's_split.png')

# ---------------------------------------------------------------- predicted against measured, scaffold splits
# Source: the poster's Figure 7 (make_poster_figures.py, section 4) at slide size
# the axes go to -12.5 so that the least soluble molecule (log S -11.6) and its large errors are shown
pred = pd.read_csv(os.path.join(TAB, 'scaffold_test_predictions.csv'))
part = pred[(pred['model'] == 'XGBoost') & (pred['split'] == 'scaffold')]
fig, ax = plt.subplots(figsize=(5.35, 5.0))
ax.scatter(part['y_true'], part['y_pred'], s=14, color=NAVY, alpha=0.25, linewidths=0, rasterized=True)
lims = [-12.5, 2.5]
ax.plot(lims, lims, color=BRIGHT, lw=2.5, label='y = x')
# a straight-line fit of predicted on measured, drawn over the range of the data (the same line as on the poster)
slope, intercept = np.polyfit(part['y_true'], part['y_pred'], 1)
xs = np.array([part['y_true'].min(), part['y_true'].max()])
ax.plot(xs, slope * xs + intercept, color=DARK, lw=2.5, ls='--', label='best fit')
print('parity plot: best-fit slope', round(slope, 3))
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks([-12, -8, -4, 0])
ax.set_yticks([-12, -8, -4, 0])
ax.set_xlabel('Measured log S (log mol/L)', fontsize=22)
ax.set_ylabel('Predicted log S (log mol/L)', fontsize=22)
ax.set_aspect('equal')
ax.legend(frameon=False, loc='lower right', fontsize=21, handlelength=1.4)   # 16 pt at its size on the slide
ax.text(0.04, 0.96, 'XGBoost, all 10\nscaffold test sets', transform=ax.transAxes, ha='left', va='top', fontsize=21)   # the empty top left corner
fig.tight_layout()
save(fig, 's_parity.png')

# ---------------------------------------------------------------- test RMSE and the splits won against kNN (Figure 4)
# Source: the poster's Figure 4 (make_poster_figures.py, section 1) at slide size
# the poster's Figure 4 drawn at its size on slide 6 (the left part of the slide, the text is on the right), so every
# label is at least 16 pt. It replaces the bar chart and the table slide. Same data: paper_table1.csv for the bars,
# scaffold_per_split.csv for the marks
t1 = pd.read_csv(os.path.join(TAB, 'paper_table1.csv'))
per_split = pd.read_csv(os.path.join(TAB, 'scaffold_per_split.csv'))
models = [('ESOL equation (Delaney, 2004)', 'ESOL equation'), ('Ridge (8 descriptors)', 'Ridge'), ('kNN (8 descriptors)', 'kNN'),
          ('Kernel ridge', 'Kernel ridge'), ('Random forest', 'Random forest'), ('XGBoost', 'XGBoost')]
LOST, BAND = '#6E6E6E', '#E6ECF4'   # grey for the lost splits, the poster's light box colour behind XGBoost
fig, (ax, ad) = plt.subplots(1, 2, figsize=(8.3, 4.9), gridspec_kw={'width_ratios': [2.05, 1.3]}, layout='constrained')
for i, (key, label) in enumerate(models):
    bold = 'bold' if key == 'XGBoost' else 'normal'
    for split, bar_colour, dot_colour, dy in [('random', LIGHT, MID, -0.2), ('scaffold', NAVY, NAVY, 0.2)]:
        row = t1[(t1['model'] == key) & (t1['test set'] == split)].iloc[0]
        ax.barh(i + dy, row['rmse_mean'], height=0.36, color=bar_colour, edgecolor=NAVY, linewidth=1.2, xerr=row['rmse_sd'],
                error_kw=dict(lw=1.4, capsize=2.5, ecolor=DARK), zorder=2)
        ax.text(row['rmse_mean'] + row['rmse_sd'] + 0.03, i + dy, f"{row['rmse_mean']:.3f}", va='center', fontsize=16, fontweight=bold)
        if key.startswith('kNN'):
            continue   # kNN is the model the others are compared with
        wins = per_split[(per_split['model'] == key) & (per_split['split'] == split)].sort_values('seed')['win']
        assert len(wins) == 10 and wins.sum() == int(row['wins_vs_knn'])   # the marks must give the counts of the paper's Table 1
        for k, won in enumerate(wins):
            if won:
                ad.scatter(k, i + dy, s=65, marker='o', color=dot_colour, edgecolor=NAVY, linewidth=1.0, zorder=3)
            else:
                ad.scatter(k, i + dy, s=55, marker='x', color=LOST, linewidth=2.0, zorder=3)
        ad.text(10.0, i + dy, str(wins.sum()), va='center', fontsize=16, fontweight=bold)
    if key.startswith('kNN'):
        ad.text(4.5, i, 'baseline', ha='center', va='center', fontsize=16, style='italic', color='#555555')
ax.set_yticks(range(len(models)))
ax.set_yticklabels([label for _, label in models], fontsize=18)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.tick_params(axis='y', length=0, pad=8)   # a little room between the model names and the axis
ax.invert_yaxis()
ax.set_xlim(0, 1.32)
ax.set_xticks([0, 0.4, 0.8, 1.2])
ax.tick_params(axis='x', labelsize=16)
ax.set_xlabel('Mean test RMSE (log mol/L)', fontsize=17)
ax.legend(handles=[Patch(facecolor=LIGHT, edgecolor=NAVY, label='random splits'), Patch(facecolor=NAVY, edgecolor=NAVY, label='scaffold splits')],
          loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=16, handletextpad=0.4, columnspacing=0.8,
          handlelength=1.6)
ad.set_xlim(-0.6, 10.9)
ad.set_ylim(ax.get_ylim())
ad.axis('off')
ad.set_title('Splits won\nagainst kNN', fontsize=17, fontweight='bold', color=NAVY, pad=8)   # two lines, so it stays clear of the legend
won_random = Line2D([], [], ls='', marker='o', markersize=9, color=MID, markeredgecolor=NAVY)
won_scaffold = Line2D([], [], ls='', marker='o', markersize=9, color=NAVY, markeredgecolor=NAVY)
lost = Line2D([], [], ls='', marker='x', markersize=9, color=LOST, markeredgewidth=2.0)
ad.legend([(won_random, won_scaffold), lost], ['won', 'lost'], handler_map={tuple: HandlerTuple(ndivide=None)},
          loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=16, columnspacing=1.5)
for a in (ax, ad):
    a.set_facecolor('none')   # see-through, so the band below shows
fig.canvas.draw()
fig.set_layout_engine('none')   # keep this layout, so the band lines up with both panels
to_fig = fig.transFigure.inverted()   # from pixels to figure coordinates (0 to 1)
# top and bottom of the XGBoost row, from data units to figure coordinates
y0 = to_fig.transform(ax.transData.transform((0, len(models) - 1 + 0.46)))[1]
y1 = to_fig.transform(ax.transData.transform((0, len(models) - 1 - 0.46)))[1]
# the band runs across the whole figure (x from 0 to 1)
fig.patches.append(Rectangle((0, y0), 1, y1 - y0, transform=fig.transFigure, color=BAND, zorder=-1))
save(fig, 's_models.png')

# ---------------------------------------------------------------- permutation importance of XGBoost (random splits)
# Source: the poster's Figure 8 (make_poster_figures.py, section 5) at slide size
perm = pd.read_csv(os.path.join(TAB, 'interpretation_permutation.csv'))
perm = perm[perm['model'] == 'XGBoost'].sort_values('mean')
labels = {'logP': 'logP', 'MW': 'Molecular weight', 'TPSA + HBA': 'TPSA + H-bond acceptors', 'Fingerprints (2,048 bits)': 'Fingerprints',
          'AromRings + AromProp': 'Aromatic rings + proportion', 'Halogens': 'Halogens', 'RotBonds': 'Rotatable bonds',
          'HBD': 'H-bond donors'}
fig, ax = plt.subplots(figsize=(6.22, 3.85))   # the size it has on the slide (it starts further left than the right half, so the bars have more room)
colours = [BRIGHT if g == 'logP' else NAVY for g in perm['group']]
ax.barh([labels[g] for g in perm['group']], perm['mean'], xerr=perm['std'], color=colours, error_kw=dict(lw=2, capsize=4, ecolor=DARK))
# the mean of each bar printed after its error bar, as on the poster (values under 0.01 shown as <0.01)
for k, (m, sd) in enumerate(zip(perm['mean'], perm['std'])):
    ax.text(m + sd + 0.03, k, f'{m:.2f}' if m >= 0.01 else '<0.01', va='center', fontsize=16)
ax.set_xlim(0, 1.4)
ax.set_xlabel('Rise in test MAE\n(log mol/L), mean ± SD\nover 10 random splits', fontsize=17)   # three short lines fit under the bars
ax.set_xticks([0, 0.5, 1.0])
ax.tick_params(axis='y', labelsize=17)
ax.tick_params(axis='x', labelsize=18)
ax.grid(axis='x', alpha=0.35)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)   # a small outer margin, so the bars get the room
save(fig, 's_importance.png')

# ---------------------------------------------------------------- log S against logP (training set of split 1, as in notebook 01)
# Source: the poster's Figure 2 (make_poster_figures.py, section 7) at slide size
sys.path.insert(0, os.path.join(REPO, 'src'))
from esol_data import RANDOM_SEED, TARGET, prepare_esol, stratified_split

os.chdir(os.path.join(REPO, 'notebooks'))   # the data path in src is relative to the notebooks folder
df_train, df_test = stratified_split(prepare_esol(), RANDOM_SEED)
rho = stats.spearmanr(df_train['LogP'], df_train[TARGET])[0]
fig, ax = plt.subplots(figsize=(5.35, 5.7))
ax.scatter(df_train['LogP'], df_train[TARGET], s=16, color=NAVY, alpha=0.35, linewidths=0, rasterized=True)
ax.set_xlabel('logP')
ax.set_ylabel('Measured log S (log mol/L)', fontsize=21)
ax.set_xticks([-5, 0, 5, 10])
ax.set_yticks([-10, -8, -6, -4, -2, 0])
# the note goes in the empty top right corner (high logP and high log S do not occur together)
ax.text(0.97, 0.97, f'{len(df_train)} training\nmolecules\nSpearman\n{rho:.2f}'.replace('-', '−'), transform=ax.transAxes,
        ha='right', va='top', fontsize=19)
ax.grid(alpha=0.35)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, 's_logp.png')
