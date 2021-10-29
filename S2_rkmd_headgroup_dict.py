import numpy as np
import os, sys, pickle, ujson
import periodictable
from chempy.util.parsing import formula_to_composition
import IsoSpecPy as iso
import matplotlib.pyplot as plt


## SCRIPT 2: Run this script second to generate reference KMD values for each class & adduct combination.

# If adduct has ion addition and removal (e.g., M+K-2H), structure in tuple (e.g., ('K','-H2'))
def add_modification(formula, mod): 

    pt = {(element,i): 0 for i, element in enumerate(periodictable.elements)}
    comp = formula_to_composition(formula)
    for key in comp:
        for element in pt:
            if key in element:
                pt[element] += comp[key]

    if type(mod) == str:
        if '-' in mod:
            mod_comp = formula_to_composition(mod[1:])
        else:
            mod_comp = formula_to_composition(mod)

        for key in mod_comp:
            for element in pt:
                if key in element:
                    if '-' in mod:
                        pt[element] -= mod_comp[key]
                    else:
                        pt[element] += mod_comp[key]
    else:
        for part in mod:
            if '-' in part:
                part_forcalc = part[1:]
                part_comp = formula_to_composition(part_forcalc)
            else:
                part_comp = formula_to_composition(part)

            for key in part_comp:
                for element in pt:
                    if key in element:
                        if '-' in part:
                            pt[element] -= part_comp[key]
                        else:
                            pt[element] += part_comp[key]
    
    mod_formula = ''
    for element in pt:
        if pt[element] > 0 and 0 not in element:
            mod_formula += f'{element[0]}{pt[element]}'
    return mod_formula


if __name__ == '__main__':
    
    # Headgroup elemental compositions included all atoms in
    # the headgroup up to the last carbon with a bonded heteroatom.
    headgroups = {
        'PC': 'C10H20NO8P',
        'O-PC': 'C9H20NO7P',
        'PE': 'C7H14NO8P',
        'O-PE': 'C6H14NO7P',
        'PS': 'C8H14NO10P',
        'O-PS': 'C9H17O9NP',
        'PI': 'C11H19O13P',
        'O-PI': 'C12H22O12P',
        'PA': 'C5H9O8P',
        'O-PA': 'C4H9O7P',
        'PG': 'C8H15O10P',
        'O-PG': 'C7H15O9P',
        'TG': 'C6H8O6',
        'DG': 'C5H8O5',
        'MG': 'C4H8O4',
        'SM': 'C9H21N2O6P',
        'EPC': 'C6H15N2O6P',
        'CER' : 'C4H9NO3',
        'CERP': 'C4H10NO6P',
        'LCER': 'C20H26NO13',
        'HCER': 'C10H19NO8',
        'SHCER': 'C14H17NO11S',
        'GM3': 'C31H37N2O21',
        'GM1': 'C45H49N3O31',
        'CE': 'C28H46O2',
        'LPC': 'C9H20NO7P',
        'O-LPC': 'C8H20NO6P',
        'LPE': 'C6H14NO7P',
        'O-LPE': 'C5H14NO6P',
        'LPS': 'C7H14NO9P',
        'O-LPS': 'C6H14NO8P',
        'LPI': 'C10H19O12P',
        'O-LPI': 'C9H19O11P',
        'LPG': 'C7H15O9P',
        'O-LPG': 'C6H15O8P',
        'LPA': 'C4H9O7P',
        'O-LPA': 'C3H9O6P',
        'FA': 'C2H4O2'
    }

    # Include all adducts that might modifiy any class.
    adducts = ['H', 'Na', 'NH4', 'K', ('H','-H2O'), '-H']

    rkm_dict = {}
    for group in headgroups:
        for mod in adducts:
            if type(mod) == str:
                rkm_dict[f'{group}_{mod}'] = 0
            else:
                mod_str = f'{group}_'
                for part in mod:
                    mod_str += part
                rkm_dict[f'{mod_str}'] = 0          

    for group in headgroups:
        form = headgroups[group]
        for mod in adducts:
            adducted_form = add_modification(form, mod)
            sp = iso.IsoTotalProb(formula=f'{adducted_form}', prob_to_cover=0.9)
            mz_vals = [mz for mz, prob in sp]
            mz = sorted(mz_vals)[0] - 0.00054858
            km_mz = mz * (14/14.01565)
            kmd = km_mz - np.floor(km_mz)
            
            if type(mod) == str:
                rkm_dict[f'{group}_{mod}'] += km_mz
            else:
                mod_str = f'{group}_'
                for part in mod:
                    mod_str += part
                rkm_dict[f'{mod_str}'] += km_mz


    with open('rkmd_adduct_dict.json', 'w') as fp:
        ujson.dump(rkm_dict, fp)
    
        
    
