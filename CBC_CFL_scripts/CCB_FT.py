"""

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
import sys,os
import h5py

def ccb_FT(plist_filename):
	plist_filename=os.path.abspath(plist_filename)
	plist=np.genfromtxt(plist_filename,skip_header=2)
	qx_arry=plist[:,4]
	qy_arry=plist[:,5]
	qz_arry=plist[:,6]
	
	H,xedges,yedges = np.histogram2d(qx_arry,qy_arry,bins=np.linspace(-1e10,1e10,100))
	H=H.T
	H_FT=np.fft.fft2(H)
	return H,H_FT,xedges,yedges


