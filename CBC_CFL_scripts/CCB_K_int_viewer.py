'''
CCB_K_int_viewer.py is to diplay the pupil funciton
'''
import sys,os
import numpy as np
import matplotlib.pyplot as plt



def read_kout_int(kout_file,bins,save_kmap=False,clim_up=5e2):
	k_exp=np.genfromtxt(kout_file)
	bins_arry_x=np.linspace(-10e8,10e8,bins+1)
	bins_arry_y=np.linspace(-10e8,10e8,bins+1)
	bins_ind_x=np.digitize(k_exp[:,-6],bins_arry_x)
	bins_ind_y=np.digitize(k_exp[:,-5],bins_arry_y)
	Int_arry=np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
	for m in range(k_exp[:,-2].shape[0]):
		Int_arry[bins_ind_x[m]-1,bins_ind_y[m]-1]+=k_exp[m,3]
	KX,KY=np.meshgrid(bins_arry_x,bins_arry_y)
	plt.figure();plt.pcolor(KX,KY,Int_arry.T,cmap='jet');plt.clim(0,clim_up);
	plt.colorbar();
	plt.title('frame %d'%(int(os.path.basename(kout_file).split('.')[0].split('fr')[1])))
	#plt.show()
	plt.savefig('Int_'+os.path.basename(kout_file).split('.')[0]+'.png')
	if save_kmap==True:
		np.save('bins_arry_x_fr%d.npy'%(int(os.path.basename(kout_file).split('.')[0].split('fr')[1])),bins_arry_x)
		np.save('bins_arry_y_fr%d.npy'%(int(os.path.basename(kout_file).split('.')[0].split('fr')[1])),bins_arry_y)
		np.save('bins_ind_x_fr%d.npy'%(int(os.path.basename(kout_file).split('.')[0].split('fr')[1])),bins_ind_x)
		np.save('bins_ind_y_fr%d.npy'%(int(os.path.basename(kout_file).split('.')[0].split('fr')[1])),bins_ind_y)
		np.save('Int_arry_fr%d.npy'%(int(os.path.basename(kout_file).split('.')[0].split('fr')[1])),Int_arry)
	return None
if __name__=='__main__':
	kout_file=os.path.abspath(sys.argv[1])
	bins=int(sys.argv[2])
	save_kmap = bool(int(sys.argv[3]))
	clim_up = float(sys.argv[4])
	read_kout_int(kout_file,bins,save_kmap=save_kmap,clim_up=clim_up)
	print(kout_file+'Done!')
