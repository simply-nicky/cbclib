'''
'Bootstrapping.py' is the wrapper for Bootstrapping iterations to converge to 
a structure factor list and the projection images at the same time.
'''
import sys,os
sys.path.append('/gpfs/cfel/user/lichufen/CBDXT/P11_BT/scripts')
print(sys.path)
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle as pk
import glob
import time

import CCB_int_proc

save_list = [0,1,2,3,4,5,6,7,8,9,10,20,50,100,150,200]
show_list = [0,1,2,3,4,5,6,7,8,9,10,20,50]

def Bootstrap(K_map_file,ite_no):
	#save_list = [0,1,2,3,4,5,6,7,8,9,10,20,50,100,150,200]
	start_t = datetime.datetime.now()
	d_exp = CCB_int_proc.Dataset(K_map_file);
	t = datetime.datetime.now()
	delta_t = t-start_t
	start_t = t
	print('intial iteration took %f seconds'%(delta_t.total_seconds()))
	pkl_name = 'Bootstrap_0.pkl'
	with open(pkl_name,'bw') as p:
		pk.dump(d_exp,p)
	# add timing funciton
	for m in range(ite_no):
		if (m+1) in [1,2,3,4,5]:
			sf_arry = np.genfromtxt('/home/lichufen/CCB_ind/ethc_mk.pdb.hkl',skip_header=3,skip_footer=2,usecols=(0,1,2,3))
			sf_arry[:,3] = 1e4*np.random.rand(sf_arry.shape[0])
			
		else:
			sf_arry = d_exp.merge_all_HKL_crystfel(output=False)
 
		d_exp.update_reference_arry(sf_arry) 
		print('round: %d'%(m+1)) 
		d_exp.get_all_frames_norm() 
		d_exp.adj_frame()
		pkl_name = 'Bootstrap_%d.pkl'%(m+1)
		if (m+1) in save_list:
			with open(pkl_name,'bw') as p:
				pk.dump(d_exp,p)
		t = datetime.datetime.now()
		delta_t = t - start_t
		start_t = t
		print('took %f seconds'%(delta_t.total_seconds()))
	print('Done!')
	return

def get_ite_ind(file_name):
	file_name = file_name.strip()
	base_name = os.path.basename(file_name)
	ite_ind = int(base_name.split('.')[0].split('_')[-1])
	return ite_ind

def Invest_ite(file_name_pattern,hkl):
	# hkl in crytsfel convention.
	plt.ion()
	file_name_pattern = os.path.abspath(file_name_pattern)
	file_list = glob.glob(file_name_pattern)
	#print(file_list)
	file_list.sort(key=get_ite_ind)
	for file in file_list:
		file = file.strip()
		
		ite_ind = os.path.basename(file).strip().split('.')[0].split('_')[-1]
		ite_ind = int(ite_ind)
		if ite_ind not in show_list:
			continue
		with open(file,'rb') as f:
			d_exp = pk.load(f)
		vars()['hkl_arry_%d'%(ite_ind)] = d_exp.merge_all_HKL_crystfel(output=False)
		d_exp.show_frame_norm(72)
		plt.savefig('%d.pdf'%(ite_ind))
		time.sleep(0.1)
		plt.close('all')
		if file==file_list[-1]:
			_ = d_exp.merge_all_HKL_crystfel(output=True)
	plt.figure(figsize=(5,4))
	labels = []
	hist_list = []
	for file in file_list:
		file = file.strip()
		ite_ind = os.path.basename(file).strip().split('.')[0].split('_')[-1]
		ite_ind = int(ite_ind)
		if ite_ind not in show_list:
			continue
		labels.append('ite %d'%(ite_ind))
		hist_list.append(vars()['hkl_arry_%d'%(ite_ind)][:,5]/vars()['hkl_arry_%d'%(ite_ind)][:,3])
		plt.hist(vars()['hkl_arry_%d'%(ite_ind)][:,5]/vars()['hkl_arry_%d'%(ite_ind)][:,3],bins=np.linspace(1e-6,0.3,61),histtype='step')
	#plt.hist(hist_list,bins=np.linspace(1e-6,0.6,61),histtype='bar')
	plt.xlabel('sigma(I)/I')
	plt.title('histogram of sigma/I')
	plt.legend(labels)
	plt.savefig('std_hist.pdf')
	
	#hkl = [hkl[1],hkl[0],hkl[2]]
	HKL = np.abs(np.array(hkl))
	ite_no_list = []
	I_list = []
	sigma_list = []
	for file in file_list:
		file = file.strip()
		ite_ind = os.path.basename(file).strip().split('.')[0].split('_')[-1]
		ite_ind = int(ite_ind)
		if ite_ind not in show_list:
			continue
		ind1 = (vars()['hkl_arry_%d'%(ite_ind)][:,0:3]==HKL).all(axis=1).nonzero()[0]
		if len(ind1)==0:
			print('HKL (%d,%d,%d) not found in ite_no %d'%(HKL[0],HKL[1],HKL[2],ite_ind))
			continue
		else:
			ind1 = ind1[0]
			I_list.append(vars()['hkl_arry_%d'%(ite_ind)][ind1,3])
			sigma_list.append(vars()['hkl_arry_%d'%(ite_ind)][ind1,5])
			ite_no_list.append(ite_ind)
	I_list = np.array(I_list)
	sigma_list = np.array(sigma_list)
	ite_no_list = np.array(ite_no_list)
	fig,ax1 = plt.subplots(figsize=(6,3))
	ax1.plot(ite_no_list,I_list,'-bx',label='I')
	ax1.set_xlabel('iteration number')
	ax1.set_ylabel('I',color='b')
	ax1.set_ylim(0,1.2*I_list.max())
	ax1.set_title('I and sigma of (%d,%d,%d)'%(HKL[0],HKL[1],HKL[2]))
	#ax1.legend()
	
	ax2 = ax1.twinx()
	ax2.plot(ite_no_list,sigma_list,'-ro',label='sigma(I)')
	ax2.set_ylabel('sigma(I)',color='r')
	#ax2.legend()
	ax2.set_ylim(0,1.2*sigma_list.max())
	fig.tight_layout()
	plt.savefig('I_ite_no.pdf')

	plt.figure(figsize=(6,3))
	plt.plot(ite_no_list,sigma_list/I_list,'-gx')
	plt.xlabel('iteration number')
	plt.ylabel('sigma(I)/I')
	plt.title('I and sigma of (%d,%d,%d)'%(HKL[0],HKL[1],HKL[2]))
	plt.tight_layout()
	plt.savefig('std_ite_no.pdf')
	plt.close('all')
	
	dd=d_exp.search_HKL((HKL[0],HKL[1],HKL[2]))
	fig,ax1 = plt.subplots(figsize=(4,3));
	ax1.plot(dd['frame_ar'],dd['inte_ave_adj_ar'],'gx-');
	ax1.set_title('HKL: %d,%d,%d'%(HKL[0],HKL[1],HKL[2]))
	ax1.set_ylim(0,1.2*dd['inte_ave_adj_ar'].max())
	ax1.set_xlabel('frame number')
	ax1.set_ylabel('I_adj',color='g')
	
	ax2 = ax1.twinx()
	ax2.plot(dd['frame_ar'],dd['inte_ave_ar'],'bx-');
	ax2.set_title('HKL: %d,%d,%d'%(HKL[0],HKL[1],HKL[2]));
	ax2.set_ylim(0,1.2*dd['inte_ave_ar'].max())
	ax2.set_ylabel('I_raw',color='b')
	
	plt.tight_layout()
	plt.savefig('HKL_I_frames.pdf')
	plt.close()


	dd=d_exp.search_hkl((hkl[0],hkl[1],hkl[2]))
	fig,ax1 = plt.subplots(figsize=(4,3));
	ax1.plot(dd['frame_ar'],dd['inte_ave_adj_ar'],'gx-');
	ax1.set_title('hkl: %d,%d,%d'%(hkl[0],hkl[1],hkl[2]))
	ax1.set_ylim(0,1.2*dd['inte_ave_adj_ar'].max())
	ax1.set_xlabel('frame number')
	ax1.set_ylabel('I_adj',color='g')

	ax2 = ax1.twinx()
	ax2.plot(dd['frame_ar'],dd['inte_ave_ar'],'bx-')
	ax2.set_title('hkl: %d,%d,%d'%(hkl[0],hkl[1],hkl[2]))
	ax2.set_ylim(0,1.2*dd['inte_ave_ar'].max())
	ax2.set_xlabel('frame number')
	ax2.set_ylabel('I_raw',color='b')


	plt.tight_layout()
	plt.savefig('hkl_I_frames.pdf')
	plt.close()
    

	return

if __name__=='__main__':
	K_map_file  = os.path.abspath(sys.argv[1])
	ite_no = int(sys.argv[2])
	Bootstrap(K_map_file,ite_no)
	hkl = [int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
	Invest_ite('Boot*.pkl',hkl)
	with open('./Bootstrap_20.pkl','rb') as f:
		d_exp = pk.load(f)
	#import CCB_int_proc
	plt.ion()
	for frame in range(720): 
		d_exp.show_frame_norm(frame)
		plt.xlim(-10e8,10e8)
		plt.ylim(-10e8,10e8)
		plt.gca().set_aspect('equal',adjustable='box')
		plt.draw()
		plt.savefig('./frame_norm_fr%03d.png'%(frame))  
		plt.close() 
	norm_all=np.zeros_like(d_exp.frame_obj_list[0].INT_n_mean); 
	for f in d_exp.frame_obj_list: 
		norm_all=norm_all+f.INT_n_mean 
	KX,KY=np.meshgrid(f.bins_arry_x,f.bins_arry_y);
	plt.figure(figsize=(3,3));
	plt.pcolor(KX,KY,norm_all);
	plt.colorbar();
	plt.title('frame norm 800');
	plt.xlim(-10e8,10e8)
	plt.ylim(-10e8,10e8)
	plt.gca().set_aspect('equal',adjustable='box')
	plt.draw()
	#plt.show()
	plt.savefig('./frame_norm_fr800.png')   

	
