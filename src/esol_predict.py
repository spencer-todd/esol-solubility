# -*- coding: utf-8 -*-
"""esol_predict.py: Predict log S for new molecules with the final model.

The model is trained and saved by notebooks/05_interpretation.ipynb (section 12) as
results/models/xgboost_final.joblib. The descriptors and fingerprints are calculated with the
same functions as for our data (src/esol_data.py).

Example:
    from esol_predict import predict_log_s
    predict_log_s(['CC(=O)Oc1ccccc1C(=O)O'])   # aspirin
"""

import os
import joblib
import pandas as pd
from rdkit import Chem

from esol_data import REPO_DIR, FEATURES_9, FP_COLUMNS, canonicalize_smiles, add_descriptors, morgan_fingerprints

MODEL_FILE = os.path.join(REPO_DIR, 'results', 'models', 'xgboost_final.joblib')


def predict_log_s(smiles_list):
    """Predicted log S (log mol/L) of each SMILES in the list, from the final XGBoost model."""
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]   # one SMILES on its own is also fine
    if len(smiles_list) == 0:
        raise ValueError("The list of SMILES is empty")
    # Source: joblib.load as in Tutorial 4 (10. Saving the model)
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("results/models/xgboost_final.joblib is missing: run 05_interpretation.ipynb first")
    model = joblib.load(MODEL_FILE)

    # the same steps as for our data: canonical SMILES without stereochemistry, then the mol object
    molecules = []
    for smiles in smiles_list:
        if not isinstance(smiles, str):
            raise ValueError(f"{smiles!r} is not a SMILES string")
        # first check that RDKit can read the SMILES at all
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() == 0:   # an empty SMILES gives a molecule with no atoms
            raise ValueError(f"RDKit cannot read the SMILES '{smiles}'")
        molecules.append(Chem.MolFromSmiles(canonicalize_smiles(smiles, isomeric=False)))

    # the 9 descriptors and the Morgan fingerprints, with the same functions as for our data
    df_new = add_descriptors(pd.DataFrame({'molecule': molecules}))
    X = pd.concat([df_new, morgan_fingerprints(df_new)], axis=1)[FEATURES_9 + FP_COLUMNS]
    return model.predict(X)
