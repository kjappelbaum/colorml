# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from glob import glob

import numpy as np
import pandas as pd
from six.moves import zip

import pybel
from mofid.run_mofid import cif2mofid


def openbabel_count_bond_order(mol, bo=2):
    import openbabel as ob

    count = 0
    mole = mol.OBMol
    for bond in ob.OBMolBondIter(mole):
        # Note: the OB implementation is wrong. It assumes that if all
        # atoms in the ring are aromatic then the ring itself must be
        # aromatic. This is not necessarily true.
        if bond.GetBO() == bo:
            count += 1
    return count


def openbabel_count_aromatic_rings(mol):
    count = 0
    mole = mol.OBMol
    for ring in mole.GetSSSR():
        # Note: the OB implementation is wrong. It assumes that if all
        # atoms in the ring are aromatic then the ring itself must be
        # aromatic. This is not necessarily true.
        if ring.IsAromatic():
            count += 1
    return count


def openbabel_count_aromatic(mol):
    import openbabel as ob

    carboncounter = 1
    double_cc = 1
    mole = mol.OBMol
    for bond in ob.OBMolBondIter(mole):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        symbol1 = atom1.GetAtomicNum()
        symbol2 = atom2.GetAtomicNum()

        if symbol1 == symbol2 == 6:
            carboncounter += 1
            if bond.IsDouble():
                double_cc += 1

    return double_cc / carboncounter - 1


def get_group_counts(mol):
    import openbabel as ob

    mole = mol.OBMol
    group_dict = {
        'primary_amide': 0,
        'secondary_amide': 0,
        'tertiary_amide': 0,
        'ester': 0,
        'carbonyl': 0,
    }
    for bond in ob.OBMolBondIter(mole):
        if bond.IsPrimaryAmide():
            group_dict['primary_amide'] += 1
        elif bond.IsSecondaryAmide():
            group_dict['primary_amide'] += 1
        elif bond.IsTertiaryAmide():
            group_dict['tertiary_amide'] += 1
        elif bond.IsEster():
            group_dict['ester'] += 1
        elif bond.IsCarbonyl():
            group_dict['carbonyl'] += 1

    return group_dict


def get_molecular_descriptors(smiles):
    mymol = pybel.readstring('smi', smiles)

    descriptordict = {}

    group_counts = get_group_counts(mymol)
    desc = mymol.calcdesc()
    db_ratio = openbabel_count_aromatic(mymol)
    aromatic_rings = openbabel_count_aromatic_rings(mymol)

    descriptordict.update(group_counts)
    descriptordict['logP'] = desc['logP']
    descriptordict['MR'] = desc['MR']
    descriptordict['dbratio'] = db_ratio
    descriptordict['aromatic_rings'] = aromatic_rings
    descriptordict['dbonds'] = desc['dbonds']
    descriptordict['abonds'] = desc['abonds']

    return descriptordict


def get_smiles_features(cif):
    mofid = cif2mofid(cif)

    name = mofid['cifname']

    linker_descriptors = []

    try:
        for linker in mofid['smiles_linkers']:
            linker_descriptors.append(list(get_molecular_descriptors(linker).values()))
            # super inefficient to do this all the time. But i do not know if i'll change the descriptorlist ...
            keys = list(get_molecular_descriptors(linker).keys())
            mean_keys = [s + '_mean' for s in keys]
            sum_keys = [s + '_sum' for s in keys]

        linker_descriptors = np.array(linker_descriptors)
        means = np.mean(linker_descriptors, axis=0)
        sums = np.mean(linker_descriptors, axis=0)

        mean_dict = dict(list(zip(mean_keys, means)))
        sum_dict = dict(list(zip(sum_keys, sums)))

        result_dict = {}
        result_dict['name'] = name

        result_dict.update(sum_dict)
        result_dict.update(mean_dict)

        return result_dict
    except Exception as e:
        print(e)
        return None


def get_moldesc(cifs):
    result_list = []
    for cif in cifs:
        result_list.append(get_smiles_features(cif))
    try:
        df = pd.DataFrame(result_list)
    except Exception:
        df = result_list

    return df


def get_racs(cif):
    """For now rely on Mohamad on this, but in the long run all this code should simply use molsimplify"""
    raise NotImplementedError


def merge_racs_moldesc(df_moldesc, df_racs):
    df_merged = pd.merge(df_racs, df_moldesc, left_on='MOFname', right_on='name')
    return df_merged
