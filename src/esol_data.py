# -*- coding: utf-8 -*-
"""esol_data.py: Loading, cleaning and featurizing the ESOL dataset, and the train/test splits
(random stratified and scaffold).

Most functions come from the notebook of our project proposal, which took them from the
course tutorials. The scaffold and family functions are new in the final project. They are kept here so
every notebook uses exactly the same data, descriptors, fingerprints, splits and families.
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

# Source: Tutorial 4 (0.3 Fix the random seed), same value as in the proposal
RANDOM_SEED = 1

# Location of the data file, relative to this file (src/, then the repository root, then data/)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(REPO_DIR, "data", "delaney-processed.csv")

# Source: Tutorial 4 (0.5 Understanding the data), column names stored as global variables
TARGET = "measured log solubility in mols per litre"
DELANEY = "ESOL predicted log solubility in mols per litre"
TARGET_LABEL = "Measured log S (log mol/L)"
# 5 log S bins (quantiles), used to stratify the train/test splits (and as log S groups in notebook 05)
TARGET_BINARY = "target_binned"
# The name comes from Tutorial 4 (1.1), which bins its target into two groups

# The 8 RDKit descriptors of the proposal, and the aromatic proportion (9th descriptor)
FEATURES = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'AromRings', 'Halogens']
FEATURES_9 = FEATURES + ['AromProp']

# the 4 terms of the ESOL equation (Delaney, 2004)
ESOL_TERMS = ['LogP', 'MW', 'RotBonds', 'AromProp']

# Axis labels with units for the plots
LABELS = {
    'MW': 'Molecular weight (g/mol)',
    'LogP': 'Crippen logP (-)',
    'TPSA': 'TPSA (Å$^2$)',
    'HBD': 'H-bond donors (count)',
    'HBA': 'H-bond acceptors (count)',
    'RotBonds': 'Rotatable bonds (count)',
    'AromRings': 'Aromatic rings (count)',
    'Halogens': 'Halogen atoms (count)',
    'AromProp': 'Aromatic proportion (-)',
}

# Short plain names for tick labels and legends (figures should not show the column names of the code)
SHORT_NAMES = {
    'MW': 'Molecular weight',
    'LogP': 'logP',
    'TPSA': 'TPSA',
    'HBD': 'H-bond donors',
    'HBA': 'H-bond acceptors',
    'RotBonds': 'Rotatable bonds',
    'AromRings': 'Aromatic rings',
    'Halogens': 'Halogen atoms',
    'AromProp': 'Aromatic proportion',
}


def load_esol():
    """Load the ESOL CSV file from MoleculeNet (Wu et al., 2018)."""
    # Source: Tutorial 4 (0.4 Import the data), as in the proposal
    # Changes: reads the ESOL CSV, with the MoleculeNet URL as fallback
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded data/delaney-processed.csv, shape = {df.shape}")
    except FileNotFoundError:
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        df = pd.read_csv(url)
        print(f"Loaded data from MoleculeNet, shape = {df.shape}")
    return df


# Source: Tutorial 3 (2. SMILES Canonicalization and 5. Working with Datasets), same code
def canonicalize_smiles(smiles, isomeric:bool=True):
    """Convert any SMILES to its canonical form.

    Args:
        smiles (str): Input SMILES string
        isomeric (bool): Whether to retain stereochemistry information

    Returns:
        canonical_smiles (str): Canonical SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    return canonical_smiles


def merge_duplicates(df, isomeric=False):
    """Average log S over the entries (rows) that have the same canonical SMILES (one row per molecule)."""
    # Source: Lecture 2 (slide 35: remove duplicates because of "bias, inflated importance") and Tutorial 3 (5.:
    #         duplicates with different target values "should be handled carefully"), as in our proposal notebook (2.3).
    #         AddMoleculeColumnToFrame from Tutorial 3 (5.)
    # Changes from the proposal:
    #   - isomeric=False by default: our descriptors can't tell cis/trans isomers apart, so
    #     they are merged too (2 more pairs than in the proposal: 1,115 instead of 1,117 molecules)
    #   - the names of merged molecules are joined (e.g. "mannitol / Sorbitol") instead of
    #     keeping only the first name. Extra spaces at the end of some names are removed
    df = df.copy()
    df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles, isomeric=isomeric)

    df_clean = df.groupby('canonical_smiles', as_index=False).agg({
        'Compound ID': lambda names: ' / '.join(names.str.strip()),
        TARGET: 'mean',
        DELANEY: 'mean',
    })

    # Add a column called 'molecule' with the mol objects
    PandasTools.AddMoleculeColumnToFrame(df_clean, smilesCol='canonical_smiles', molCol='molecule')
    return df_clean


# Source: Tutorial 3 (5. Working with Datasets), same function as in the proposal
# Changes (proposal): added AromRings (NumAromaticRings) and Halogens (fr_halogen)
def calculate_key_descriptors(mol):
    """Calculate key molecular descriptors for a molecule."""

    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'AromRings': Descriptors.NumAromaticRings(mol),
        'Halogens': Descriptors.fr_halogen(mol),
    }

    return descriptors


def aromatic_proportion(mol):
    """Fraction of heavy atoms that are aromatic, one of the 4 terms of the ESOL equation."""
    # Source: proposal notebook (10.5), loop over the atoms of each mol object
    n_aromatic = 0
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            n_aromatic += 1
    return n_aromatic / mol.GetNumHeavyAtoms()


def add_descriptors(df_clean):
    """Add the 8 descriptors and the aromatic proportion as columns."""
    # Source: Tutorial 3 (5. Working with Datasets), same loop as in the proposal
    descriptor_data = []
    for row, mol in df_clean['molecule'].items():
        desc = calculate_key_descriptors(mol)
        desc['AromProp'] = aromatic_proportion(mol)
        descriptor_data.append(desc)

    descriptor_df = pd.DataFrame(descriptor_data, index=df_clean.index)
    return pd.concat([df_clean, descriptor_df], axis=1)


def prepare_esol(isomeric=False):
    """Load, clean and featurize ESOL, and add the log S bins used to stratify the splits."""
    df = load_esol()
    df_clean = merge_duplicates(df, isomeric=isomeric)
    df_clean = add_descriptors(df_clean)

    # Source: Tutorial 4 (1.1 Split with stratification), as in the proposal
    # Changes (proposal): 5 quantile bins of log S instead of one threshold
    df_clean[TARGET_BINARY] = pd.qcut(df_clean[TARGET], q=5, labels=False)

    print(f"Dataset size: {len(df)} rows, {len(df_clean)} unique molecules")
    return df_clean


def stratified_split(df_clean, seed):
    """80/20 train/test split, stratified on the log S bins (same as the proposal)."""
    # Source: Tutorial 4 (1.1 Split with stratification)
    df_train, df_test = train_test_split(
        df_clean,
        train_size=0.8,
        test_size=0.2,
        random_state=seed,
        stratify=df_clean[TARGET_BINARY]
    )
    return df_train, df_test


# Source: new (final project). Bemis-Murcko scaffolds (Bemis and Murcko, 1996) with RDKit, as in
#         the Chemprop code of Yang et al. (2019). Loop over the mol objects as in Tutorial 3 (5.)
def murcko_scaffolds(df_clean):
    """Bemis-Murcko scaffold (ring systems and the chains that link them) of every molecule, as SMILES.

    Molecules without a ring get an empty scaffold ''.
    """
    scaffolds = []
    for mol in df_clean['molecule']:
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    return pd.Series(scaffolds, index=df_clean.index, name='scaffold')


# Source: new (final project). The families of molecules of notebook 05 (section 8), defined by
#         structure rules before any error was computed. Kept here so notebooks 02 and 05 use the same rules.
#         SMARTS substructure search (HasSubstructMatch) is not in the course.
#         NumAromaticRings as in Tutorial 3 (3.)
# steroid skeleton (gonane) as SMARTS: [#6] = any carbon, ~ = any bond, the digits open and close
# the four rings, !a = not aromatic
STEROID = Chem.MolFromSmarts('[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]~[#6]3~[#6]~2~[#6]~[#6]'
                             '~[#6;!a]4~[#6;!a]~[#6;!a]~[#6;!a]~[#6;!a]~3~4')
# the ring scaffold of the PCBs as SMILES (two benzene rings joined by a bond)
BIPHENYL = 'c1ccc(-c2ccccc2)cc1'
FAMILY_ORDER = ['steroid', 'PCB', 'large PAH', 'sugar or polyol', 'other']


def family_of(mol, scaffold):
    """The family of a molecule (the first rule that matches)."""
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if mol.HasSubstructMatch(STEROID):
        return 'steroid'
    if scaffold == BIPHENYL and 'Cl' in symbols:
        return 'PCB'
    # only carbon atoms (the hydrogens are not atoms of the mol object)
    if Descriptors.NumAromaticRings(mol) >= 4 and set(symbols) == {'C'}:
        return 'large PAH'
    # fr_Al_OH: RDKit's count of aliphatic OH groups
    if Descriptors.fr_Al_OH(mol) >= 3 and Descriptors.NumAromaticRings(mol) == 0:
        return 'sugar or polyol'
    return 'other'


def molecule_families(df_clean, scaffolds):
    """The family of every molecule (scaffolds from murcko_scaffolds())."""
    families = []
    for i, mol in df_clean['molecule'].items():
        families.append(family_of(mol, scaffolds[i]))
    return pd.Series(families, index=df_clean.index, name='family')


# Source: new (final project). Adapted from the "balanced" scaffold split of Yang et al. (2019) and their
#         Chemprop code ("Put stuff that's bigger than half the val/test size into train, rest just order
#         randomly"). A group split as in Lecture 3 (slide 17)
# Changes: the rule for the large groups is the same. We fill the test set first, so that every scaffold
#          test set has the same size as the test set of the random split (223 molecules)
def scaffold_split(df_clean, scaffolds, seed, test_size=0.2):
    """80/20 train/test split in which every scaffold is wholly in the training set or the test set.

    1. Scaffold groups larger than half the test set always go to the training set
       (in ESOL: the benzene ring and the empty scaffold of all molecules without a ring).
    2. The other groups are put in a random order (set by the seed) and added to the test set if they
       still fit, until the test set has as many molecules as in the random 80/20 split.
    3. The rows of both sets are shuffled (set by the seed), because df_clean is sorted by SMILES.
    """
    n_test = round(test_size * len(df_clean))
    group_sizes = scaffolds.value_counts()

    # 1. small groups only. Sorted first so that the random order depends only on the seed
    small_groups = sorted(group_sizes[group_sizes <= n_test / 2].index)

    # 2. random order (np.random.default_rng as in Tutorial 4, 4.1), then fill the test set
    rng = np.random.default_rng(seed)
    rng.shuffle(small_groups)
    test_groups = []
    n_in_test = 0
    for group in small_groups:
        if n_in_test + group_sizes[group] <= n_test:
            test_groups.append(group)
            n_in_test += group_sizes[group]

    # 3. split and shuffle the rows
    in_test = scaffolds.isin(test_groups)
    df_train = df_clean[~in_test].sample(frac=1, random_state=seed)
    df_test = df_clean[in_test].sample(frac=1, random_state=seed)
    return df_train, df_test


# Source: Tutorial 3 (Morgan fingerprints for a dataset: generator, loop, fp.ToList(), DataFrame)
# Changes: 2,048 bits (Tutorial 3, section 4) instead of 1,024, written as a function
FP_RADIUS = 2
FP_SIZE = 2048
FP_COLUMNS = [f'Morgan_bit_{i}' for i in range(FP_SIZE)]


def morgan_fingerprints(df_clean):
    """Morgan fingerprints (radius 2, 2,048 bits) of every molecule, as a DataFrame of 0/1 columns."""
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

    fingerprints = []
    for mol in df_clean['molecule']:
        fp = morgan_gen.GetFingerprint(mol)
        fingerprints.append(fp.ToList())

    return pd.DataFrame(fingerprints, columns=FP_COLUMNS, index=df_clean.index)
