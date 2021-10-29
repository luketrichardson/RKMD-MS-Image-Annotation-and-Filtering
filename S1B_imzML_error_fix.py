from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import numpy as np
import os
from tqdm import tqdm


## SCRIPT 1B: Run this script if SCRIPT 1A encounters a ValueError.

def imzML_fix(data_path, imzml_file, polarity_str='positive'):

    imzml_path = os.path.join(data_path, imzml_file)

    print('-- Parsing imzML file --\n')
    p = ImzMLParser(imzml_path)
    print('-- Done --\n')
    
    with ImzMLWriter(os.path.join(data_path, f'{imzml_file[:-6]}_fixed.imzML'), polarity=polarity_str) as writer:
        for idx, coords in enumerate(tqdm(p.coordinates, desc='Loading MS data')):
            try:
                mzs, intensities = p.getspectrum(idx)
                writer.addSpectrum(mzs, intensities, coords)

            except ValueError:
                writer.addSpectrum(np.zeros(0), np.zeros(0))
        

if __name__ == '__main__':
    data_path = 'C:\\Path\\To\\IMS\\Data' # Provide path to data directory with imzML and ibd files
    imzml_file = 'image_data_file.imzML' # Provide imzML data file name

    imzML_fix(data_path, imzml_file)