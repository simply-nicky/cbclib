'''
CCB_K_viewer.py is to diplay the pupil funciton
'''
import sys,os
import numpy as np
import matplotlib.pyplot as plt



def read_kout(kout_file,bins):
	pix_arry=np.genfromtxt(kout_file)
	K_out=pix_arry[:,-6:-3] 
	K_in=pix_arry[:,-3:]
	K_out_len=np.linalg.norm(K_out,axis=-1)
	K_in_len=np.linalg.norm(K_in,axis=-1)
	#plt.figure();plt.hist2d(K_in[:,0],K_in[:,1],bins=500,cmin=0,cmax=50,cmap='jet');
	plt.figure();plt.hist2d(K_in[:,0],K_in[:,1],bins=bins,cmap='jet');
	plt.xlim(-8e8,8e8)
	plt.ylim(-8e8,8e8)
	plt.clim(0,50)
	plt.colorbar();
	plt.title('frame %d'%(int(os.path.basename(kout_file).split('.')[0].split('fr')[1])))
	#plt.show()
	plt.savefig(os.path.basename(kout_file).split('.')[0]+'.png')
	return None
if __name__=='__main__':
	kout_file=os.path.abspath(sys.argv[1])
	bins=int(sys.argv[2])
	read_kout(kout_file,bins)
	print(kout_file+'Done!')
