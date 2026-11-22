# Predicting aqueous solubility (log S) of the ESOL molecules

CHE1147 (Chemical Data Science and Engineering), University of Toronto, Fall 2026. Team 9: Rojin Ostadsharif Memar and Spencer Todd.

This repository holds all the code, data and results for our final project. We predict the aqueous solubility (log S, in log mol/L) of the molecules in the ESOL dataset (Delaney, 2004) from their structure. The file has 1,128 entries, which are 1,115 unique molecules after we merge duplicates. We compare kernel ridge, a random forest and XGBoost with simple baselines, mainly a k-nearest neighbours (kNN) model on eight molecular descriptors. We test the baselines and the three models on 10 stratified random splits, and all of them except the logP-only baseline on 10 scaffold splits. In a scaffold split, every test molecule has a Bemis–Murcko scaffold (its ring systems and the linkers between them) that is not in the training set. We then look at which inputs the models rely on, which molecules they predict worst, and how well they can flag an unreliable prediction.

## Main results

- On the 10 random splits, XGBoost (descriptors + fingerprints) has a test root mean squared error (RMSE) of 0.607 ± 0.036 log mol/L against 0.803 ± 0.032 for kNN, and it is lower in 10 of 10 splits (`03_models.ipynb`).
- On the random splits, kNN looks strong partly because of descriptor twins, training molecules with the same 8 descriptors as a test molecule. kNN's test RMSE is 0.468 on molecules with a twin in the training set and 0.876 on molecules without one (mean of 10 splits, notebook 03, section 6).
- On the 10 scaffold splits, XGBoost has 0.834 ± 0.051 against 1.030 ± 0.057 for kNN, again lower in 10 of 10 splits (`04_scaffold_split_and_uncertainty.ipynb`).
- The RMSE of XGBoost rises from 0.607 (all random test molecules) to 0.690 (the molecules of the random splits that have a ring scaffold other than benzene) to 0.834 (scaffold splits). So part of the rise comes from which molecules are tested and part from the new scaffolds (notebook 04, section 6).
- These results do not depend on how we tuned the models on the scaffold splits. When we tune again with cross-validation folds grouped by scaffold, XGBoost still beats kNN in 10 of 10 scaffold splits, and its rise from 0.690 is about the same (notebook 05, section 4.4). Kernel ridge improves the most with these folds (scaffold test RMSE 0.962 to 0.835) and then reaches about the same RMSE as XGBoost tuned the same way (0.839).

The ± values are the standard deviation over the 10 splits.

## Folder structure

```
esol-solubility/
├── README.md            this file
├── pyproject.toml       Python version and package versions
├── uv.lock              exact versions of every package (for uv)
├── requirements.txt     the same versions, for pip
├── data/
│   └── delaney-processed.csv     the ESOL dataset (MoleculeNet version)
├── src/
│   ├── esol_data.py         loading, cleaning, descriptors, fingerprints, random and scaffold splits, molecule families
│   ├── esol_models.py       the model pipelines and their parameter ranges
│   ├── esol_evaluation.py   regression metrics, the parity plot and the figure style
│   └── esol_predict.py      predict_log_s(): log S of new molecules with the final model
├── notebooks/
│   ├── 01_data_and_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_models.ipynb
│   ├── 04_scaffold_split_and_uncertainty.ipynb
│   └── 05_interpretation.ipynb
└── results/
    ├── eda_report.html   automatic exploratory data analysis (EDA) report (ydata-profiling)
    ├── figures/          every figure, as PNG (Figures 1, 3 and 4 of the paper also as PDF)
    ├── tables/           every table and all test-set predictions, as CSV
    └── models/
        └── xgboost_final.joblib   the final XGBoost model
```

The functions in `src/` are shared by all notebooks, so every notebook uses exactly the same data, descriptors, fingerprints, splits and models. The notebooks contain the analysis. Each notebook starts with a short description and an outline. Each code cell starts with a `Source` line that names the tutorial or lecture its code comes from, or says that the code is new.

## Setting up the environment

We use [uv](https://docs.astral.sh/uv/) to install Python and the packages. `uv.lock` pins every package, so `uv sync --frozen` installs exactly the versions we used (Python 3.11, numpy 1.26.4, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2, xgboost 3.2.0, RDKit 2026.03.6, shap 0.49.1). uv downloads Python 3.11 itself if it is not installed. Install uv (see its website), then run from a terminal:

```bash
git clone https://github.com/spencer-todd/esol-solubility.git
cd esol-solubility
uv sync --frozen
```

This creates a virtual environment in `.venv/`. We use Python 3.11 rather than the course's Python 3.10, because the scipy version of our proposal (1.15.3) no longer loads on current macOS. The comment in `pyproject.toml` explains this.

Without uv, use Python 3.11 and pip with `requirements.txt` (the same versions):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, the second line is `.venv\Scripts\activate`. With pip, leave out `uv run` in the commands below (for example, start JupyterLab with `jupyter lab`).

RDKit 2026.03.6 has ready-made packages for Apple Silicon Macs, Linux (x86_64 and aarch64) and Windows (x64), but not for Intel Macs. On an Intel Mac, please use one of the other systems or a cloud machine.

## How to run the training and analysis

Run the notebooks in order, 01 to 05, because notebooks 03 to 05 read tables that the earlier ones save in `results/tables/`. You can open them in JupyterLab:

```bash
uv run jupyter lab
```

or run all five from the terminal, from the `esol-solubility` folder (the notebooks run in the order 01 to 05):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/0*.ipynb
```

| Notebook | What it does | Reads | Run time |
|---|---|---|---|
| `01_data_and_eda.ipynb` | Cleans the data (merges duplicates, 1,128 entries to 1,115 molecules), calculates the descriptors and explores the data (EDA, principal component analysis) | `data/` only | about 1 min |
| `02_baselines.ipynb` | Tests the baselines on 10 stratified random splits: the mean predictor, logP only, the ESOL equation (as published and refitted), ridge and kNN, and looks at which molecules are descriptor twins | `data/` only | about 30 s |
| `03_models.ipynb` | Tunes and tests kernel ridge, the random forest and XGBoost on the same 10 random splits, and compares them with kNN | tables of notebook 02 | about 1 h |
| `04_scaffold_split_and_uncertainty.ipynb` | Builds 10 scaffold splits and tests all models on them. Compares the error with the similarity to the training set, estimates the uncertainty (random forest trees and a Gaussian process) and makes Table 1 of the paper | tables of notebooks 02 and 03 | about 20 min |
| `05_interpretation.ipynb` | Runs the bootstrap, learning and validation curves, checks of the tuning (budget, folds grouped by scaffold, default XGBoost, number of trees), permutation importance, SHAP (SHapley Additive exPlanations) and the worst-predicted families, measures the training time, and trains and saves the final model | tables of notebooks 02, 03 and 04 | about 2 h 45 min (about 20 min with `RUN_TUNING_CHECKS = False`) |

The whole run takes about 4 hours. The run times were measured on a Mac with an Apple M5 chip (10 cores, 16 GB of memory). The searches use all cores, so a computer with fewer cores takes longer. Two shorter options:
- In `05_interpretation.ipynb`, set `RUN_TUNING_CHECKS = False` in section 0. The notebook then reads the saved results of its three big tuning checks (sections 4.2, 4.3 and 4.4) instead of running them.
- For a quick check that the environment works (under 2 minutes), run notebooks 01 and 02, then the `predict_log_s()` example below.

All random seeds are fixed (`RANDOM_SEED = 1` in `src/esol_data.py`), so a new run gives the same tables and figures. Only the measured times change (the timing prints and `training_time.csv`).

## Where to find the results

- `results/tables/paper_table1.csv`: Table 1 of the paper. It gives the scores of each model on three test sets: the random splits, the "other ring" molecules of the random splits (molecules that have a ring scaffold other than benzene, the kind of molecules a scaffold test set holds) and the scaffold splits.
- `results/tables/models_test_predictions.csv` and `results/tables/baselines_test_predictions.csv`: the test-set predictions of every model on the 10 random splits.
- `results/tables/scaffold_test_predictions.csv`: the test-set predictions of the 10 models compared in notebook 04, on the 10 random and the 10 scaffold splits, with the kind of each molecule, its descriptor twins and its similarity to the training set.
- `results/tables/tuning_check_*.csv`: the checks of the tuning in notebook 05 (section 4). They cover the bigger searches on the random and scaffold splits, the tuning with folds grouped by scaffold, default against tuned XGBoost, and 100 against 300 trees in the random forest.
- `results/models/xgboost_final.joblib`: the final model, XGBoost on the 9 descriptors (the 8 descriptors plus the aromatic proportion) and Morgan fingerprints. It is trained on all 1,115 molecules with the settings that cross-validation chose most often in notebook 03 (notebook 05, section 12).
- `results/eda_report.html`: the automatic EDA report. It is 10 MB, so GitHub does not show it. Download it and open it in a web browser.
- `results/figures/`: the file names start with the topic (`eda_`, `baselines_`, `models_`, `scaffold_`, `uncertainty_`, `interpretation_`), except `pipeline_diagram.png`.

The table and figures in the main text of the paper:

| In the paper | File | Made in |
|---|---|---|
| Figure 1 | `results/figures/pipeline_diagram.png` (and `.pdf`) | notebook 05, section 13 |
| Table 1 | `results/tables/paper_table1.csv` | notebook 04, section 6.1 |
| Figure 2 | `results/figures/models_twin_vs_no_twin.png` | notebook 03, section 6 |
| Figure 3 | `results/figures/scaffold_random_vs_scaffold.png` (and `.pdf`) | notebook 04, section 9 |
| Figure 4 | `results/figures/interpretation_importance_and_shap.png` (and `.pdf`) | notebook 05, section 13 |

## Predicting new molecules

Start Python from the `esol-solubility` folder (with uv: `uv run python`), then:

```python
import sys
sys.path.append('src')   # from the esol-solubility folder
from esol_predict import predict_log_s

predict_log_s(['CC(=O)Oc1ccccc1C(=O)O'])   # aspirin, returns the predicted log S in log mol/L
```

The model was trained on the ESOL molecules only, so its predictions are less reliable for molecules that are not similar to them (notebook 04, section 7).

## Data

`data/delaney-processed.csv` is the ESOL dataset as distributed by MoleculeNet (Wu et al., 2018), downloaded from https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv. The copy in this repository is the file we used, unchanged. If the file is missing, `load_esol()` in `src/esol_data.py` downloads it from this address.

## References

The sources cited in the notebooks and in this file:

- Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs. 1. Molecular Frameworks. *J. Med. Chem.* 1996, 39 (15), 2887–2893. https://doi.org/10.1021/jm9602928
- Bonett, D. G.; Wright, T. A. Sample Size Requirements for Estimating Pearson, Kendall and Spearman Correlations. *Psychometrika* 2000, 65 (1), 23–28. https://doi.org/10.1007/BF02294183
- Delaney, J. S. ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. *J. Chem. Inf. Comput. Sci.* 2004, 44 (3), 1000–1005. https://doi.org/10.1021/ci034243x
- Jain, N.; Yalkowsky, S. H. Estimation of the Aqueous Solubility I: Application to Organic Nonelectrolytes. *J. Pharm. Sci.* 2001, 90 (2), 234–252. [https://doi.org/10.1002/1520-6017(200102)90:2&lt;234::AID-JPS14&gt;3.0.CO;2-V](https://doi.org/10.1002/1520-6017%28200102%2990%3A2%3C234%3A%3AAID-JPS14%3E3.0.CO%3B2-V)
- Palmer, D. S.; Mitchell, J. B. O. Is Experimental Data Quality the Limiting Factor in Predicting the Aqueous Solubility of Druglike Molecules? *Mol. Pharmaceutics* 2014, 11 (8), 2962–2972. https://doi.org/10.1021/mp500103r
- Sorkun, M. C.; Khetan, A.; Er, S. AqSolDB, a Curated Reference Set of Aqueous Solubility and 2D Descriptors for a Diverse Set of Compounds. *Sci. Data* 2019, 6 (1), 143. https://doi.org/10.1038/s41597-019-0151-1
- Wu, Z.; Ramsundar, B.; Feinberg, E. N.; Gomes, J.; Geniesse, C.; Pappu, A. S.; Leswing, K.; Pande, V. MoleculeNet: A Benchmark for Molecular Machine Learning. *Chem. Sci.* 2018, 9 (2), 513–530. https://doi.org/10.1039/C7SC02664A
- Yang, K.; Swanson, K.; Jin, W.; Coley, C.; Eiden, P.; Gao, H.; Guzman-Perez, A.; Hopper, T.; Kelley, B.; Mathea, M.; Palmer, A.; Settels, V.; Jaakkola, T.; Jensen, K.; Barzilay, R. Analyzing Learned Molecular Representations for Property Prediction. *J. Chem. Inf. Model.* 2019, 59 (8), 3370–3388. https://doi.org/10.1021/acs.jcim.9b00237

Software: RDKit, scikit-learn, XGBoost, SHAP, pandas, NumPy, SciPy, matplotlib and seaborn (versions in `pyproject.toml`).

## Use of generative AI

We used Claude (Anthropic) to help write and check the code. Our generative AI usage statement, submitted with the paper, describes where and how.
