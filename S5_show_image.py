import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy.signal as sgnl


def dir_dict(directory):
    dir_dict = {}
    for file in os.listdir(directory):
        
        if file.endswith('.npz'):
            key = file.split('_')[1]
            dir_dict[key] = os.path.join(directory, file)

    return dir_dict


def generate_subimage(selected_mainclass, img, levels_dict, alias_dict, selected_subclasses, colormap='magma', show=True, save=True):
    recon_img = np.zeros(img[:,:,0].shape)
    num_params = len(selected_subclasses)

    if num_params == 0:
        for alias in alias_dict:
            recon_img += img[:,:,levels_dict[alias_dict[alias]]]
        if save:
            plt.imsave(f'{selected_mainclass}_image.svg', recon_img, cmap=colormap)

    if num_params == 1:
        for alias in alias_dict:
            if selected_subclasses[0] in alias:
                recon_img += img[:,:,levels_dict[alias_dict[alias]]]
        if save:
            plt.imsave(f'{selected_mainclass}_{selected_subclasses[0]}_image.svg', recon_img, cmap=colormap)

    if num_params == 2:
        for alias in alias_dict:
            if selected_subclasses[0] in alias and selected_subclasses[1] in alias:
                recon_img += img[:,:,levels_dict[alias_dict[alias]]]
        if save:
            plt.imsave(f'{selected_mainclass}_{selected_subclasses[0]}_{selected_subclasses[1]}_image.svg', recon_img, cmap=colormap)

    
    if show:
        plt.imshow(recon_img, cmap='magma')
        plt.show()

    return None


def generate_img_params(sc1, sc2):
    params = []
    if sc1:
        params.append(sc1)
    if sc2:
        params.append(sc2)

    return params


if __name__ == '__main__':

    directory = 'C:\\Users\\luke_richardson\\Box\\Solouki\\Projects\\RKMD MALDI\\Paper #1\\Upload Scripts'
    # directory = 'C:\\Directory\\location\\of\\.npz\\image\\data'

    d = dir_dict(directory)

    # First, select a main class image dataset (var. select_mainclass). This dataset will contain image data for those species that are of that main class. Second, provide any desired
    # subclass information (vars. select_subclass1, select_subclass2); if none provided, set variables to False. For example, to visualize all sphingomyelin lipids, the main class is 
    # set to 'SM'; to specify sphingomyelins with 1 degree of unsaturation, either subclass variable is set to 1 (the other is set as False); to specifiy sphingomeylins with 1 DoU and 
    # 34 radyl carbons, the other subclass variable is set to 34.0.

    # Lipid molecular classes are selected with strings (e.g., 'SM'), radyl carbon chain lengths are selected with floats with one decimal place (e.g., 34.0),
    # and degrees of unsaturation are selected with integer values (e.g., 9).

    select_mainclass = 'SM'  # Corresponding to filter criteria used to generate class-specific image data (.npz) file
    select_subclass1 = 34.0  # If none provided, set to False
    select_subclass2 = 1     # If none provided, set to False

    data = np.load(d[select_mainclass], allow_pickle=True)
    image_data = data['image']
    levels = data['levels'][()]
    aliases = data['alias'][()]

    image_params = generate_img_params(select_subclass1, select_subclass2)

    generate_subimage(select_mainclass, image_data, levels, aliases, image_params)


