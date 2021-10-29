from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm, trange
import scipy.signal as signal
import numpy as np
import os

## SCRIPT 1A: Run this script first.

def imzML_to_npydict(data_path, imzml_file, centroid=True, int_window=2):

    imzml_path = os.path.join(data_path, imzml_file)

    print('-- Parsing imzML file --\n')
    p = ImzMLParser(imzml_path)
    print('-- Done --\n')

    xt, yt = 0, 0
    for idx, (x,y,z) in enumerate(tqdm(p.coordinates, desc=f'Finding max coordinate')):
        if x > xt:
            xt = x
        if y > yt:
            yt = y

    print('\n')
    img_ms_dict = {}
    for idx, (x,y,z) in enumerate(tqdm(p.coordinates, desc=f'Loading MS data')):

        mzs, intensities = p.getspectrum(idx)
        if centroid:
            peaks = signal.find_peaks(intensities)[0]
            peak_mzs = np.take(mzs, peaks)
            peak_intensities = np.take(intensities, peaks)
        else:
            peaks = signal.find_peaks(intensities)[0]
            peak_mzs = np.take(mzs, peaks)
            peak_intensities = np.array([np.trapz(intensities[peak-int_window:peak+int_window]) for peak in peaks])
        
        img_ms_dict[(x,y)] = (peak_mzs, peak_intensities)

    for xi in trange(xt, desc='Filling in zero-coordinates'):
        for yi in range(yt):
            if (xi+1,yi+1) not in img_ms_dict:
                img_ms_dict[(xi+1,yi+1)] = (np.zeros(0), np.zeros(0))

    np.save(f'{imzml_file[:-6]}.npy', img_ms_dict)
    
    return None

if __name__ == '__main__':
    data_path = 'C:\\Path\\To\\IMS\\Data'      # Provide path to data directory with imzML and ibd files
    imzml_file = 'image_data_file.imzML'       # Provide imzML data file name

    # integration_window = 3                   # If using continuum m/z data (centroid = False), set integration window keybased on m/z peak width
    imzML_to_npydict(data_path, imzml_file)

    # NOTE: If a ValueError is encountered, process the imzML file with 'imzML_error_fix.py'.
