import numpy as np
import ujson


## SCRIPT 6: Run this script to determine the appropriate chain length addition for a new lipid class.

if __name__ == '__main__':

    molclass = 'SM'
    mod = 'H'
    radyl_carbons = 38
    unsaturations = 1
    mz_value = 759.6374

    with open('rkmd_adduct_dict.json', 'r') as fp:
        rkmd_dict = ujson.load(fp)

    km_mz = mz_value * (14 / 14.01565)
    kmd_mzs = km_mz - np.floor(km_mz)
    knms = np.floor(km_mz)

    if type(mod) != str:
        mod_str = ''
        for part in mod:
            mod_str += part
        mod = mod_str
  
    comp_kmd = rkmd_dict[f'{molclass}_{mod}'] - np.floor(rkmd_dict[f'{molclass}_{mod}'])
    rkmds = (kmd_mzs - comp_kmd) / 0.0134
    rkmd_dist = np.abs(rkmds-np.rint(rkmds))

    comp_mz = rkmd_dict[f'{molclass}_{mod}'] * (14.01565 / 14)
    pos_peaks = mz_value - comp_mz - (26.01565 * unsaturations)
    num_CH2 = pos_peaks / 14.01565

    chain_length = np.round(num_CH2 + (np.abs(np.round(rkmds)) * 2))
    cl_add = radyl_carbons - chain_length

    print(f'\nNumber of carbons to add for {molclass}:', cl_add, '\n')