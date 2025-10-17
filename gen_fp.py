#from imaplib import IMAP4
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd 
import os
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors       
from map4 import MAP4


def get_fp(smile):
    # RDKit descriptors -->
    nbits = 1024
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    
    fpFunc_dict = {}
    fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
    fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
    fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
    fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
    fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
    fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
    fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
    fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
    fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
    #fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
    fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
    fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
    #fpFunc_dict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
    #fpFunc_dict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
    fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
    fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
    fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
    fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)
    fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
    fpFunc_dict['map4'] = lambda m: MAP4.calculate(m)

    mol = Chem.MolFromSmiles(smile)
     
    morgan = np.array(fpFunc_dict['ecfp4'](mol)).flatten().astype(np.float32)
    hashap = np.array(fpFunc_dict['hashap'](mol)).flatten().astype(np.float32)
    rdkitfp = np.array(fpFunc_dict['rdk5'](mol)).flatten().astype(np.float32)

    return   morgan
    # return   rdkitfp 




dataset = 'ALMANAC'
# dataset = 'ONEIL'
base_dir = os.path.dirname(__file__) 
data_dir = os.path.join(base_dir, "Data")
    
if dataset == 'ONEIL':
    drug_smiles_file = os.path.join(data_dir, "ONEIL-COSMIC", "drug_smiles.csv")
else:
    drug_smiles_file = os.path.join(data_dir, "ALMANAC-COSMIC", "drug_smiles.csv")
    
drug = pd.read_csv(drug_smiles_file, sep=',', header=0)
drugid2smile = dict(zip(drug['pubchemid'], drug['isosmiles']))
drugid2graph = {}
for key, value in drugid2smile.items():
    drugid2graph[key] = get_fp(value)

with open(f'{dataset}_drugfp.pkl', 'wb') as f:
    pickle.dump(drugid2graph, f, protocol=pickle.HIGHEST_PROTOCOL)
