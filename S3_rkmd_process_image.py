import numpy as np
import pandas as pd
import os, sys, pickle, ujson
from tqdm import tqdm, trange


## SCRIPT 3: Run this script third to generate RKMD-based annotations.

def mk_chunks(d, chunk_size):
    chunk = {}
    ctr = chunk_size
    for key, val in d.items():
        chunk[key] = val
        ctr -= 1
        if ctr == 0:
            yield chunk
            ctr = chunk_size
            chunk = {}
    if chunk:
        yield chunk


def dump_big_dict(d, fname, chunk_size):
    with open(f'{fname}.pkl', 'wb') as fout:
        for chunk in tqdm(mk_chunks(d, chunk_size), f'Serializing dictionary (Chunk size: {chunk_size})'):
            pickle.dump(chunk, fout, protocol=pickle.HIGHEST_PROTOCOL)


def get_closest(array, values):
    #make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return array[idxs]


def rkmd_process_image(image_data, adducts, comp_km_key,  max_db=9, max_rkmd_δ=0.35, max_rccl_ε=0.001, align_peaks=True):
    
    # Load dictionary with headgroup+adduct KMD values.
    with open('rkmd_adduct_dict.json', 'r') as fp:
        rkmd_dict = ujson.load(fp)

    # Dictionary with appropriate chain length additions per class. When calculating numbers
    # of radyl carbons, each class required a class-dependent addition of some integer. When
    # adding new classes, it is suggested that a representative lipid from the new class be
    # tested with SCRIPT S6. 
    cl_add_dict = {
        'SM': 4,
        'EPC': 4,
        'HCER': 4, 
        'CERP': 4,
        'CER': 4,
        'TG': 3,
        'PE': 2, 
        'PC': 2, 
        'PA': 2, 
        'PG': 2, 
        'PS': 2, 
        'PI': 2, 
        'DG': 2,
        'FA': 2,
        'O-PG': 1, 
        'O-PC': 1, 
        'O-PE': 1, 
        'O-PS': 1, 
        'O-PI': 1, 
        'O-PA': 1, 
        'MG': 1, 
        'LPC': 1, 
        'LPE': 1, 
        'LPS': 1, 
        'LPI': 1, 
        'LPG': 1, 
        'LPA': 1
    }

    # Load recalibrated m/z values from a CSV in working directory. File should have a single
    # column header with the label 'm/z' followed by a list of recalibrated m/z values, or, at
    # least, a list of m/z values that correspond to peak maxima from a summed MS spectrum.
    # Running the script without aligning peaks in not recommended.
    if align_peaks:
        recal_df = pd.read_csv('recalibrated mz values.csv')
        recal_mzs = np.array(recal_df['m/z'])

    # Find max x, y pixel coordinates in image data.
    xt, yt = 0, 0
    for x, y in image_data:
        if x > xt:
            xt = x
        if y > yt:
            yt = y
    
    # Instatiate annotation dictionary w/ empty dict at each coordinate.
    annotation_dict = {(x,y): {} for x in range(xt) for y in range(yt)}

    # Main processing loop; iterates through each x,y coordinate.
    for xi in trange(xt, desc=f'Performing RKMD analysis for {comp_km_key} filter'):
        for yi in range(yt):
            
            # Loads MS peak information from image data (note: image data indexed at 1).
            peak_mzs, peak_integrations = image_data[(xi+1,yi+1)]
            if np.sum(peak_integrations) != 0:
                
                # Align each spectrum with recalibrated spectrum.
                if align_peaks:
                    peak_mzs = get_closest(recal_mzs, peak_mzs)

                # Calculate KMD for each peak (CH2).
                km_mzs = peak_mzs * (14 / 14.01565)
                kmd_mzs = km_mzs - np.floor(km_mzs)

                # Iterate over each class-specific adduct form.
                for mod in adducts[comp_km_key]:
                    
                    if type(mod) != str:
                        mod_str = ''
                        for part in mod:
                            mod_str += part
                        mod = mod_str

                    # Perform RKMD calculations.
                    comp_kmd = rkmd_dict[f'{comp_km_key}_{mod}'] - np.floor(rkmd_dict[f'{comp_km_key}_{mod}'])
                    rkmds = (kmd_mzs - comp_kmd) / 0.0134
                    rkmd_dist = np.abs(rkmds-np.rint(rkmds))
                    comp_mz = rkmd_dict[f'{comp_km_key}_{mod}'] * (14.01565 / 14)

                    # Iterate through potential degrees of unsaturation...
                    for DoU in range(max_db+1):

                        # ...and caculate number of radyl carbons and ε.
                        pos_peaks = peak_mzs - comp_mz - (26.01565 * DoU)
                        num_CH2 = pos_peaks / 14.01565
                        chain_length = num_CH2 + (DoU * 2) + cl_add_dict[comp_km_key]
                        chain_defect = np.abs(chain_length - np.rint(chain_length))
                        
                        # Generate a mask array to determine features that satisfy
                        # inlcusion conditions.
                        m1 = (rkmd_dist <= max_rkmd_δ).astype(int)
                        m2 = (np.rint(rkmds) == DoU * -1).astype(int)
                        m3 = (chain_defect <= max_rccl_ε).astype(int)
                        mask = m1 + m2 + m3
                        
                        if 3 in set(mask):
                            unmasked_peaks = peak_mzs[mask == 3]
                            unmasked_integrations = peak_integrations[mask == 3]
                            chain_length = np.round(chain_length[mask == 3])
                            chain_defect = chain_defect[mask == 3]
                            rkm_defect = rkmd_dist[mask == 3]
                            even_chain = chain_length % 2 == 0

                            # Population annotation dictionary for class/adduct combination.
                            for j, peak in enumerate(unmasked_peaks):
                                peak_key = str(peak)
                                if peak_key not in annotation_dict[(xi,yi)]:
                                    annotation_dict[(xi,yi)].update({peak_key: {(comp_km_key,mod):
                                                                                                    {
                                                                            'molclass': comp_km_key,
                                                                            'adduct': mod, 
                                                                            'chainlength': chain_length[j], 
                                                                            'unsaturation': DoU, 
                                                                            'cl_defect': chain_defect[j], 
                                                                            'rkm_defect': rkm_defect[j], 
                                                                            'even_chain': even_chain[j], 
                                                                            'counts': unmasked_integrations[j]
                                                                            }}
                                    })
                                else:
                                    annotation_dict[(xi,yi)][peak_key][(comp_km_key,mod)] = {
                                                                            'molclass': comp_km_key,
                                                                            'adduct': mod, 
                                                                            'chainlength': chain_length[j], 
                                                                            'unsaturation': DoU, 
                                                                            'cl_defect': chain_defect[j], 
                                                                            'rkm_defect': rkm_defect[j], 
                                                                            'even_chain': even_chain[j], 
                                                                            'counts': unmasked_integrations[j]
                                                                            }
                        else:
                            continue

            else:
                continue

    return annotation_dict


if __name__ == '__main__':

    # Provide class strings for which you would like to perform RKMD annotation.
    class_list = ['TG', 'DG', 'MG', 'FA', 'SM', 'HCER', 'CERP', 'CER', 'PE', 'PC', 'PA', 'PG', 'O-PG', 'O-PC', 'O-PE', 'O-PA', 'LPE', 'LPC', 'LPA', 'LPG']

    # Provide adduct forms for each class included in RKMD annotation.
    adducts = {
        'TG': ['Na'],
        'DG': ['Na', ('H','-H2O')],
        'MG': ['Na', ('H','-H2O')],
        'FA': ['K', 'Na', 'H'],
        'SM': ['H', 'Na', 'K'],
        'CER': ['H', 'Na', 'K', ('H','-H2O')],
        'CERP': ['H', 'Na', 'K', ('H','-H2O')],
        'HCER': ['H', 'Na', 'K', ('H','-H2O')],
        'PE': ['H', 'Na', 'K'], 
        'PC': ['H', 'Na', 'K'],
        'PA': ['H', 'Na', 'K'],
        'PG': ['H', 'Na', 'K'],
        'O-PG': ['H', 'Na', 'K'],
        'O-PC': ['H', 'Na', 'K'],
        'O-PE': ['H', 'Na', 'K'],
        'O-PA': ['H', 'Na', 'K'],
        'LPE': ['H', 'Na', 'K'],
        'LPC': ['H', 'Na', 'K'],
        'LPA': ['H', 'Na', 'K'],
        'LPG': ['H', 'Na', 'K']
    }

    print('-- Loading MS image data --\n')
    image_data_fname = 'imaging_data_file.npy' # Provide name of .npy imaging file name (generated by SCRIPT 1A).
    image_data = np.load(image_data_fname, allow_pickle=True)[()] 
    print('-- Done --\n')

    print('-- Starting RKMD MALDI filtering --\n')
    master_annotation_dict = {}
    for k, key in enumerate(class_list):

        max_db = 9          # Provide global maximum number of double bonds to be considered during annotation.
        max_rkmd_δ = 0.35   # Provide maximum RKMD δ to be condsidered during annotation.
        max_rccl_ε = 0.001  # Provide maximum radyl carbon chain length ε to be considering during annotation.
        class_annotation_dict = rkmd_process_image(image_data, adducts, key, max_db, max_rkmd_δ, max_rccl_ε)

        # Population master annotation dictionary that will contain annotations for all class/adduct combos.
        for pos in tqdm(class_annotation_dict, desc='Updating annotation dictonary'):
            if k == 0:
                master_annotation_dict[pos] = class_annotation_dict[pos]
            else:
                for peak in class_annotation_dict[pos]:
                    if peak not in master_annotation_dict[pos]:
                        master_annotation_dict[pos][peak] = class_annotation_dict[pos][peak]
                    else:
                        for antn_key in class_annotation_dict[pos][peak]:
                            master_annotation_dict[pos][peak][antn_key] = class_annotation_dict[pos][peak][antn_key]
        
        print('\n-- Annotation dictionary updated --\n')

        print(f'\n-- {key} RKMD filtering complete --\n')

    output_dictionary_fname = 'annotation_dictionary' # Provide file name for the output annotation dictionary
                                                      # with no file extension.
    
    dump_big_dict(master_annotation_dict, output_dictionary_fname, 100)

    print('\n-- All jobs complete --\n')
