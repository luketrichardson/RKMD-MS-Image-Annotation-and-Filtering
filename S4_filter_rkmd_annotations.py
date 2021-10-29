import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, pickle
from tqdm import tqdm, trange


## SCRIPT 4: Run this script fourth to generate class-based image datasets.


def gen_image_params(classes=False, chain_lengths=False, unsaturations=False):
    run_list, run_type, evens = [], [], []
    if classes:
        for tag in classes:
            run_list.append(tag)
            run_type.append('molclass')
            evens.append(True)
    if chain_lengths:
        for tag in chain_lengths:
            run_list.append(tag)
            run_type.append('chainlength')
            evens.append(True)
    if unsaturations:
        for tag in unsaturations:
            run_list.append(tag)
            run_type.append('unsaturation')
            evens.append(True)

    params = list(zip(run_list,run_type,evens))

    return params


def load_annotation_pkldict(fname):
    annotation_dict = {}
    with open(f'{fname}.pkl', 'rb') as fin:
        while True:
            try:
                small_dict = pickle.load(fin)
            except EOFError:
                break
            annotation_dict.update(small_dict)
    
    return annotation_dict


def find_max_coordinates(d):
    xt, yt = 0, 0
    for x, y in d:
        if x > xt:
            xt = x
        if y > yt:
            yt = y

    return xt, yt


def batch_process(annotation_dict, image_params, max_coordinate, save_dir, max_ppm_error=2.5, save_svg=False):

    df = pd.DataFrame()
    tags, mzs, ids, score, ppm = [], [], [], [], []
    for params in image_params:
        image, annotations, levels, alias = annotations_to_image(master_annotation_dict, max_coordinate, params, max_ppm_error)
        
        for key in annotations:
            tags.append(f'{params[1]}: {params[0]}')
            mzs.append(key)
            ids.append(annotations[key][0])
            score.append(annotations[key][1])
            ppm.append(annotations[key][2])
        
        np.savez(os.path.join(save_dir, f'{params[1]}_{params[0]}_evens_{str(params[2])}_errorlimit_{max_ppm_error}ppm_image.npz'), image=image, annotations=annotations, levels=levels, alias=alias)
        
        if save_svg:
            plt.imsave(os.path.join(save_dir, f'{params[1]}_{params[0]}_evens_{str(params[2])}_errorlimit_{max_ppm_error}ppm_image.svg'), np.sum(image,-1), cmap='magma')

    df['Group'] = tags
    df['m/z'] = mzs
    df['Identity'] = ids
    df['RKMD δ'] = score
    df['PPM Error'] = ppm
    df.to_csv(os.path.join(save_dir, f'Comprehensive_evens{str(params[2])}_{max_ppm_error}ppm_annotations.csv'), index=False)

    return None


def annotations_to_image(annotation_dict, image_shape, params, max_ppm_error=2.5):

    x, y = image_shape
    preallocated_array_depth = 50
    img = np.zeros((x,y,preallocated_array_depth))

    # Dictonary containing lower and upper bounds for potential radyl carbon
    # chain lengths for each lipid class.
    max_cl_dict = {
        'PC': (26,47),
        'PCe': (26,47),
        'PE': (26,47),
        'O-PE': (26,47),
        'PS': (26,47),
        'O-PS': (26,47),
        'PI': (26,47),
        'O-PI': (26,47),
        'PA': (26,47),
        'O-PA': (26,47),
        'PG': (26,47),
        'O-PG': (26,47),
        'TG': (48,70),
        'DG': (26,47),
        'MG': (12,25),
        'SM': (26,47),
        'CERP': (26,47),
        'CER': (26,47),
        'HCER': (26,47),
        'LPC': (0,25),
        'LPE': (0,25),
        'LPS': (0,25),
        'LPI': (0,25),
        'LPG': (0,25),
        'LPA': (0,25),
        'FA': (0, 25)
    }
    
    # Dictonary containing upper bounds for potential degrees of 
    # unsaturation for each lipid class.
    max_unsat_dict = {
        'PC': 9,
        'O-PC': 9,
        'PE': 9,
        'O-PE': 9,
        'PS': 9,
        'O-PS': 9,
        'PI': 9,
        'O-PI': 9,
        'PA': 9,
        'O-PA': 9,
        'PG': 9,
        'O-PG': 9,
        'TG': 9,
        'DG': 9,
        'MG': 5,
        'SM': 3,
        'CERP': 2,
        'CER': 3,
        'HCER': 3,
        'LPC': 5,
        'LPE': 5,
        'LPS': 5,
        'LPI': 5,
        'LPG': 5,
        'LPA': 5,
        'FA': 6
    }

    level = 0
    image_annotations, image_levels, level_alias = {}, {}, {}
    criterion, dict_key, prefer_even = params

    # Iterate of each pixel entry in the annotation dictionary.
    for i in trange(image_shape[0], desc=f'Building {dict_key}: {criterion} image'):
        for j in range(image_shape[1]):

            # Annotations for each pixel position.
            pos_data = annotation_dict[(i,j)]

            for mz_key in pos_data:
                annotations = pos_data[mz_key].values()

                # Filter out unacceptable chain lengths and degrees of unsaturation.
                sort_antns = [antn for antn in sorted(annotations, key = lambda v: v['rkm_defect']) 
                                if max_cl_dict[antn['molclass']][0] <= antn['chainlength'] <= max_cl_dict[antn['molclass']][1]
                                and antn['unsaturation'] <= max_unsat_dict[antn['molclass']]]
                
                if sort_antns:
                    pos_class = sort_antns[0]

                    # Odd chain lengths are filtered out, if desired.
                    if prefer_even:
                        even_inds = [i for i, antn in enumerate(sort_antns) if antn['even_chain']]
                        if even_inds:
                            pos_class = sort_antns[even_inds[0]]
                        else:
                            continue
                    
                    # Check top ranked annotation against filter criterion and a ppm error limit.
                    if criterion == pos_class[dict_key] and pos_class['rkm_defect'] <= max_ppm_error / (13_415 * np.power(np.float64(mz_key), -1)):
                        
                        # Store annotation information for report.
                        if mz_key not in image_annotations:

                            tag = pos_class['molclass']
                            cl = pos_class['chainlength']
                            unsat = pos_class['unsaturation']
                            mod = pos_class['adduct']
                            rkmd_score = np.round(pos_class['rkm_defect'], 5)
                            ppm_error = np.round((13_415 * np.power(np.float64(mz_key), -1)) * rkmd_score, 5)
                            image_annotations[mz_key] = (f'{tag} {cl}:{unsat}+{mod}', rkmd_score, ppm_error)
                        
                        # Store image data and metadata on a per feature basis.
                        if mz_key not in image_levels:
                            image_levels[mz_key] = level
                            level_alias[(tag, cl, unsat, mod)] = mz_key
                            level += 1
                            
                        img[i,j, image_levels[mz_key]] += pos_class['counts']
                else:
                    continue
    
    img = img[:,:,:level]

    return img, image_annotations, image_levels, level_alias


if __name__ == '__main__':

    # In each of the below lists, denote the criteria (e.g., lipid molecular classes, radyl carbon 
    # chain lengths,and degrees of unsaturation) for which class-based image datasets should be 
    # generated. Only include criteria that were used to generate the annotation dictionary in 
    # use (generated in SCRIPT 3).

    classes = ['SM', 'CERP', 'PE', 'PC', 'PA', 'PG', 
               'O-PG', 'O-PC', 'O-PE', 'O-PA', 'LPE', 
               'LPC', 'LPA', 'LPG', 'TG', 'DG', 'MG', 
               'HCER', 'FA', 'CER']
    
    chain_lengths = [16.0, 18.0, 20.0, 22.0, 24.0, 
                     26.0, 28.0, 30.0, 32.0, 34.0, 
                     36.0, 38.0, 40.0, 42.0] 

    unsaturations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Include filter criteria lists that you would like to include in the batch process. If you,
    # would like to exclude a list from the batch, replace the variable in function gen_image_params
    # with False.
    batch_params = gen_image_params(classes, chain_lengths, unsaturations)

    annotation_dictionary_fname = 'annotation_dictionary' # Have dictionary located in working directionary. 
                                                          # Do not include file extension in name.

    print('\n-- Loading annotations dictionary --\n')
    master_annotation_dict = load_annotation_pkldict(annotation_dictionary_fname)
    print('\n-- Annotations dictionary loaded--\n')

    # save_dir = 'C:\\Save\\location\\for\\resulting\\image\\and\\annotation\\data'
    save_dir = 'C:\\Users\\luke_richardson\\Box\\Solouki\\Projects\\RKMD MALDI\\Paper #1\\Upload Scripts'

    xt, yt = find_max_coordinates(master_annotation_dict)
    max_ppm_error = 2.5 # Set the maximum allowable ppm error for any annotated feature.

    batch_process(master_annotation_dict, batch_params, (xt,yt), save_dir, max_ppm_error)
