import numpy as np

import os
import glob

from matplotlib import rc

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':10})

# Set the font used for MathJax
rc('mathtext',**{'default':'regular'})
rc('figure',**{'figsize':(8,6)})

# rc('text',**{'usetex':True})

# plt.rcParams['text.usetex'] = True # TeX rendering

np.pow = np.power

def run_model(tRatio):
    os.system(f'python3 Ex_2DTopoRelax_Cartesian_FreeSurf_ALEIB_noswarm.py {tRatio}')

test_timeratio = [1, 2, 4, 8, 16, 32, 64]
for tRatio in test_timeratio:
    run_model(tRatio)
