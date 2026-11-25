# Draws the figures of the poster at poster size, from the result tables of the repository.
# No model is fitted here. Every number is read from results/tables
# (the log S against logP figure uses the training set of split 1, built with src/esol_data.py as in notebook 01).
# Run from the esol-solubility folder, after the notebooks:  uv run python poster_and_slides/make_poster_figures.py
# The figures go to results/figures/poster (PNG, and a PDF copy that stays sharp in the poster PDF).
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
OUT = os.path.join(REPO, 'results', 'figures', 'poster')
os.makedirs(OUT, exist_ok=True)

# blues only: navy (the ChemE signature), a mid blue, a light blue and a bright blue for the accent
NAVY, MID, LIGHT, BRIGHT = '#002A5C', '#5B7FB0', '#C9D6E8', '#3D8FD1'
DARK = '#1A1A1A'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 24, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 1.8, 'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'axes.edgecolor': DARK,
                     'text.color': DARK, 'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
                     'hatch.linewidth': 1.6})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=600, facecolor='white')
    # vector copy for the poster PDF (no creation date, so every run gives the same file)
    fig.savefig(os.path.join(OUT, name.replace('.png', '.pdf')), facecolor='white', metadata={'CreationDate': None})
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------- 1. test RMSE of six models and the splits each won against kNN
# Source: new figure for the poster (Figure 4). Horizontal bars as in Tutorial 4 (11.4), one mark per split
# Figure 4 (it replaces the bar chart and Table 1): the mean test RMSE ± SD over 10 splits (paper_table1.csv), and one
# mark per split (scaffold_per_split.csv): a dot where the model beat kNN on that test set, a grey cross where it lost
t1 = pd.read_csv(os.path.join(TAB, 'paper_table1.csv'))
per_split = pd.read_csv(os.path.join(TAB, 'scaffold_per_split.csv'))
models = [('ESOL equation (Delaney, 2004)', 'ESOL equation'), ('Ridge (8 descriptors)', 'Ridge'), ('kNN (8 descriptors)', 'kNN'),
          ('Kernel ridge', 'Kernel ridge'), ('Random forest', 'Random forest'), ('XGBoost', 'XGBoost')]
LOST, BAND = '#6E6E6E', '#E6ECF4'   # grey for the lost splits, the poster's light box colour behind XGBoost
fig, (ax, ad) = plt.subplots(1, 2, figsize=(11.3, 9.3), gridspec_kw={'width_ratios': [2.1, 1.35]}, layout='constrained')
for i, (key, label) in enumerate(models):
    bold = 'bold' if key == 'XGBoost' else 'normal'
    # the random split above the scaffold split, as in the legend. Dots of the random splits in mid blue, so they read as filled
    for split, bar_colour, dot_colour, dy in [('random', LIGHT, MID, -0.2), ('scaffold', NAVY, NAVY, 0.2)]:
        row = t1[(t1['model'] == key) & (t1['test set'] == split)].iloc[0]
        ax.barh(i + dy, row['rmse_mean'], height=0.36, color=bar_colour, edgecolor=NAVY, linewidth=1.5, xerr=row['rmse_sd'],
                error_kw=dict(lw=1.6, capsize=3, ecolor=DARK), zorder=2)
        ax.text(row['rmse_mean'] + row['rmse_sd'] + 0.035, i + dy, f"{row['rmse_mean']:.3f}", va='center', fontsize=20, fontweight=bold)
        if key.startswith('kNN'):
            continue   # kNN is the model the others are compared with
        wins = per_split[(per_split['model'] == key) & (per_split['split'] == split)].sort_values('seed')['win']
        assert len(wins) == 10 and wins.sum() == int(row['wins_vs_knn'])   # the dots must give the counts of the paper's Table 1
        for k, won in enumerate(wins):
            if won:
                ad.scatter(k, i + dy, s=170, marker='o', color=dot_colour, edgecolor=NAVY, linewidth=1.2, zorder=3)
            else:
                ad.scatter(k, i + dy, s=130, marker='x', color=LOST, linewidth=2.4, zorder=3)
        ad.text(10.0, i + dy, str(wins.sum()), va='center', fontsize=20, fontweight=bold)
    if key.startswith('kNN'):
        ad.text(4.5, i, 'baseline', ha='center', va='center', fontsize=20, style='italic', color='#555555')
ax.set_yticks(range(len(models)))
ax.set_yticklabels([label for _, label in models], fontsize=22)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.tick_params(axis='y', length=0, pad=8)   # a little room between the model names and the axis
ax.invert_yaxis()
ax.set_xlim(0, 1.32)
ax.set_xticks([0, 0.4, 0.8, 1.2])
ax.tick_params(axis='x', labelsize=20)
ax.set_xlabel('Mean test RMSE (log mol/L)', fontsize=21)
ax.legend(handles=[Patch(facecolor=LIGHT, edgecolor=NAVY, label='random splits'), Patch(facecolor=NAVY, edgecolor=NAVY, label='scaffold splits')],
          loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=20, handletextpad=0.4, columnspacing=1.2)
ad.set_xlim(-0.6, 10.9)
ad.set_ylim(ax.get_ylim())
ad.axis('off')
ad.set_title('Splits won against kNN', fontsize=21, fontweight='bold', color=NAVY, pad=22)   # level with the legend
# the key under the marks: a dot for a split won (both blues), a cross for a split lost
won_random = Line2D([], [], ls='', marker='o', markersize=13, color=MID, markeredgecolor=NAVY)
won_scaffold = Line2D([], [], ls='', marker='o', markersize=13, color=NAVY, markeredgecolor=NAVY)
lost = Line2D([], [], ls='', marker='x', markersize=12, color=LOST, markeredgewidth=2.4)
ad.legend([(won_random, won_scaffold), lost], ['won', 'lost'], handler_map={tuple: HandlerTuple(ndivide=None)},
          loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=20, columnspacing=1.5)
for a in (ax, ad):
    a.set_facecolor('none')   # see-through, so the band below shows
fig.canvas.draw()
fig.set_layout_engine('none')   # keep this layout, so the band lines up with both panels
# one light band behind the XGBoost rows, across both panels (as the XGBoost row of the table was shaded)
to_fig = fig.transFigure.inverted()   # from pixels to figure coordinates (0 to 1)
# top and bottom of the XGBoost row, from data units to figure coordinates
y0 = to_fig.transform(ax.transData.transform((0, len(models) - 1 + 0.46)))[1]
y1 = to_fig.transform(ax.transData.transform((0, len(models) - 1 - 0.46)))[1]
# the band runs across the whole figure (x from 0 to 1)
fig.patches.append(Rectangle((0, y0), 1, y1 - y0, transform=fig.transFigure, color=BAND, zorder=-1))
save(fig, 'p_models.png')

# ---------------------------------------------------------------- 2. descriptor twins (random splits)
# Source: Figure 2a of the paper (notebook 04, section 9) at poster size. Bars as in Tutorial 4 (11.4)
# one colour pair for both models: hatched = with a twin, solid mid blue = no twin
tw = pd.read_csv(os.path.join(TAB, 'models_twins.csv'))
fig, ax = plt.subplots(figsize=(5.35, 4.85))
names = [('kNN (8 descriptors)', 'kNN'), ('XGBoost', 'XGBoost')]
for k, (key, label) in enumerate(names):
    part = tw[tw['model'] == key]
    for shift, column, colour, hatch in [(-0.2, 'rmse_twins', 'white', '///'), (0.2, 'rmse_no_twin', MID, None)]:
        ax.bar(k + shift, part[column].mean(), 0.4, yerr=part[column].std(), color=colour, edgecolor=NAVY, hatch=hatch,
               linewidth=2, error_kw=dict(lw=2, capsize=5, ecolor=DARK))
        ax.text(k + shift, part[column].mean() + part[column].std() + 0.02, f'{part[column].mean():.2f}', ha='center', fontsize=20)
ax.bar(0, 0, color='white', edgecolor=NAVY, hatch='///', label='with a twin')
ax.bar(0, 0, color=MID, edgecolor=NAVY, label='no twin')
ax.set_xticks([0, 1])
ax.set_xticklabels([label for _, label in names])
ax.set_ylabel('Test RMSE (log mol/L)', fontsize=22)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.5, 1.0])
ax.grid(axis='y', alpha=0.35)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=20, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.3, columnspacing=1.2)
fig.tight_layout(pad=0.3)   # a small outer margin, so no wide white band is left above the legend
save(fig, 'p_twins.png')

# ---------------------------------------------------------------- 3. RQ3 step chart
# Source: Figure 2b of the paper (notebook 04, section 9) at poster size. Each bar starts where the one before ends
xgb = {s: t1[(t1['model'] == 'XGBoost') & (t1['test set'] == s)]['rmse_mean'].iloc[0]
       for s in ['random', 'random, other ring', 'scaffold']}
a, b, c = xgb['random'], xgb['random, other ring'], xgb['scaffold']
fig, ax = plt.subplots(figsize=(6.3, 4.8))   # placed wider than the other small figures, so its labels fit
ax.bar(0, a, 0.62, color=LIGHT, edgecolor=NAVY, linewidth=2)
ax.bar(1, b - a, 0.62, bottom=a, color=MID, edgecolor=NAVY, linewidth=2)
ax.bar(2, c - b, 0.62, bottom=b, color=BRIGHT, edgecolor=NAVY, linewidth=2)
ax.bar(3, c, 0.62, color=NAVY, edgecolor=NAVY, linewidth=2)
for xi, top, label in [(0, a, f'{a:.3f}'), (1, b, f'+{b - a:.3f}'), (2, c, f'+{c - b:.3f}'), (3, c, f'{c:.3f}')]:
    ax.text(xi, top + 0.02, label, ha='center', fontsize=20, fontweight='bold')
ax.plot([0.31, 0.69], [a, a], color=DARK, lw=1.5)
ax.plot([1.31, 1.69], [b, b], color=DARK, lw=1.5)
ax.plot([2.31, 2.69], [c, c], color=DARK, lw=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(['random\nsplits', 'harder\ntest set', 'new\nscaffolds', 'scaffold\nsplits'], fontsize=19)
ax.set_ylabel('XGBoost test RMSE (log mol/L)', fontsize=21)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.35)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, 'p_rq3.png')

# ---------------------------------------------------------------- 4. predicted against measured, scaffold splits
# Source: parity plot as in Tutorial 4 (6.), XGBoost on the 10 scaffold test sets as in Figure A6.2b of the paper
# the axes go to -12.5 so that the least soluble molecule (log S -11.6) and its large errors are shown
pred = pd.read_csv(os.path.join(TAB, 'scaffold_test_predictions.csv'))
part = pred[(pred['model'] == 'XGBoost') & (pred['split'] == 'scaffold')]
fig, ax = plt.subplots(figsize=(5.35, 5.0))
ax.scatter(part['y_true'], part['y_pred'], s=14, color=NAVY, alpha=0.25, linewidths=0, rasterized=True)
lims = [-12.5, 2.5]
ax.plot(lims, lims, color=BRIGHT, lw=2.5, label='y = x')
# a straight-line fit of predicted on measured, drawn over the range of the data. It is flatter than y = x,
# so the most and least soluble molecules are predicted too close to the mean
slope, intercept = np.polyfit(part['y_true'], part['y_pred'], 1)
xs = np.array([part['y_true'].min(), part['y_true'].max()])
ax.plot(xs, slope * xs + intercept, color=DARK, lw=2.5, ls='--', label='best fit')
print('parity plot: best-fit slope', round(slope, 3))
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks([-12, -8, -4, 0])
ax.set_yticks([-12, -8, -4, 0])
ax.set_xlabel('Measured log S (log mol/L)', fontsize=20)
ax.set_ylabel('Predicted log S (log mol/L)', fontsize=20)
ax.set_aspect('equal')
ax.legend(frameon=False, loc='lower right', fontsize=18, handlelength=1.4)
ax.text(0.04, 0.96, 'XGBoost, all 10\nscaffold test sets', transform=ax.transAxes, ha='left', va='top', fontsize=18)   # the empty top left corner
fig.tight_layout()
save(fig, 'p_parity.png')

# ---------------------------------------------------------------- 5. permutation importance of XGBoost (random splits)
# Source: Figure 4a of the paper (notebook 05, section 13) at poster size. Horizontal bars as in Tutorial 4 (11.4)
perm = pd.read_csv(os.path.join(TAB, 'interpretation_permutation.csv'))
perm = perm[perm['model'] == 'XGBoost'].sort_values('mean')
labels = {'logP': 'logP', 'MW': 'Molecular weight', 'TPSA + HBA': 'TPSA + H-bond acceptors', 'Fingerprints (2,048 bits)': 'Fingerprints',
          'AromRings + AromProp': 'Aromatic rings + proportion', 'Halogens': 'Halogens', 'RotBonds': 'Rotatable bonds',
          'HBD': 'H-bond donors'}
# wider than the other small figures: on the poster it starts 0.6 in further left (into the white space right of the
# parity plot), so the bars have more room
fig, ax = plt.subplots(figsize=(5.93, 5.0))
colours = [BRIGHT if g == 'logP' else NAVY for g in perm['group']]
ax.barh([labels[g] for g in perm['group']], perm['mean'], xerr=perm['std'], color=colours, error_kw=dict(lw=2, capsize=4, ecolor=DARK))
# the mean of each bar printed after its error bar, as on the other bar charts (values under 0.01 shown as <0.01)
for k, (m, sd) in enumerate(zip(perm['mean'], perm['std'])):
    ax.text(m + sd + 0.05, k, f'{m:.2f}' if m >= 0.01 else '<0.01', va='center', fontsize=18)
ax.set_xlim(0, 1.4)
ax.set_xlabel('Rise in test MAE\n(log mol/L), mean ± SD\nover 10 random splits', fontsize=18)   # three short lines fit under the bars
ax.set_xticks([0, 0.5, 1.0])
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=19)   # the wider axis brings the ticks closer, so a smaller size keeps them apart
ax.grid(axis='x', alpha=0.35)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, 'p_importance.png')

# ---------------------------------------------------------------- 6. what a scaffold split is (a schematic, no data)
# Source: new schematic, drawn with matplotlib markers
fig, axes = plt.subplots(1, 2, figsize=(10.9, 5.0))
families = ['family A', 'family B', 'family C', 'family D']
random_test = {(0, 3), (1, 0), (1, 4), (2, 2), (3, 1)}
for ax, title, test in [(axes[0], 'Random split', random_test), (axes[1], 'Scaffold split', {(3, k) for k in range(5)})]:
    for row in range(4):
        for k in range(5):
            is_test = (row, k) in test
            ax.scatter(k, 3 - row, s=1150, marker='s' if is_test else 'o', color=BRIGHT if is_test else LIGHT,
                       edgecolor=NAVY, linewidth=2)
        ax.text(-0.75, 3 - row, families[row], ha='right', va='center', fontsize=20)
    ax.set_xlim(-2.6, 4.6)
    # title centred over the 5 columns of molecules (x = 0 to 4). The family labels are left out of the centring
    ax.set_title(title, fontsize=26, fontweight='bold', color=NAVY, pad=12, x=(2 + 2.6) / (4.6 + 2.6))
    ax.set_ylim(-0.6, 3.6)
    ax.axis('off')
training = axes[0].scatter([], [], s=420, marker='o', color=LIGHT, edgecolor=NAVY, linewidth=2, label='training molecule')
test = axes[0].scatter([], [], s=420, marker='s', color=BRIGHT, edgecolor=NAVY, linewidth=2, label='test molecule')
fig.tight_layout(rect=(0, 0.1, 1, 1))
# one key under each grid of molecules, so the keys line up with the grids and the titles above them
for ax, handle in [(axes[0], training), (axes[1], test)]:
    middle = fig.transFigure.inverted().transform((ax.transData.transform((2, 0))[0], 0))[0]
    fig.legend(handles=[handle], frameon=False, loc='lower center', bbox_to_anchor=(middle, 0), fontsize=21, handletextpad=0.3)
save(fig, 'p_split.png')

# ---------------------------------------------------------------- 7. log S against logP (training set of split 1, as in notebook 01)
# Source: scatter plot as in Tutorial 3 (Mini Exercise 3), the data of Figure A1.2 of the paper
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
save(fig, 'p_logp.png')
