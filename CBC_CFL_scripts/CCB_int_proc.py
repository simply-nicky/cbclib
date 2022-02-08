'''
CCB_int_proc.py 

processes the intensities of steaks and pixels for frames and data set.

'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
from skimage import measure, morphology, feature
import scipy
import glob
import h5py
import re
import CCB_ref
import CCB_pred
import CCB_pat_sim
import CCB_read
#import gen_match_figs as gm
#import CCB_streak_det
import matplotlib
#matplotlib.use('TkAgg') # To be adjusted for the batch job mode.
import matplotlib.pyplot as plt
import h5py
import scipy.ndimage as ndi
import pickle as pk


E_ph = 17.4
wave_len = 1e-10*12.4/E_ph
k0 = 1/wave_len


class Dataset:
	
	def __init__(self,K_map_file):
		self.K_map_file = os.path.abspath(K_map_file)
		self.arry = np.genfromtxt(K_map_file)
		self.get_frames()
		self.clean()
		
		reference_arry = np.genfromtxt('/home/lichufen/CCB_ind/ethc_mk.pdb.hkl',skip_header=3,skip_footer=2,usecols=(0,1,2,3))
		self.reference_arry = np.hstack((reference_arry[:,0:1],reference_arry[:,1:2],reference_arry[:,2:]))

		#self.reference_arry[:,3] = 1

		self.get_hkl_arry()
		self.get_HKL_arry()
		#self.get_batch_norm(300)
		
		self.get_all_frames_norm(bins=200)
			
		self.adj_frame()
		
		
	def get_frames(self):
		frame_arry, frame_ind = np.unique(self.arry[:,0], return_inverse=True)
		self.frame_total_no = frame_arry.shape[0]
		self.frame_arry = frame_arry.astype(np.int)
		self.frame_obj_list=[]
		for i in range(self.frame_total_no):
			frame = Frame(self, self.frame_arry[i])               # create the Frame obj.
			self.frame_obj_list.append(frame)
		
	
	def get_HKL_arry(self):
		HKL_arry=np.array([],dtype=np.int).reshape(-1,3)
		for f in self.frame_obj_list:
			hkl_arry = np.abs(f.hkl_arry)
			HKL_arry = np.vstack((HKL_arry, hkl_arry))
		HKL_arry, ind = np.unique(HKL_arry,axis=0,return_inverse=True)
		Redundancy=np.array([np.count_nonzero(ind==m) for m in range(HKL_arry.shape[0])])
		self.HKL_arry = HKL_arry
		self.Redundancy = Redundancy
	
	def get_hkl_arry(self):
		hkl_arry = np.array([],dtype=np.int).reshape(-1,3)
		for f in self.frame_obj_list:
			hkl_arry = np.vstack((hkl_arry, f.hkl_arry))
		hkl_arry, ind = np.unique(hkl_arry, axis=0, return_inverse=True)
		redundancy = np.array([np.count_nonzero(ind==m) for m in range(hkl_arry.shape[0])])
		self.hkl_arry = hkl_arry
		self.redundancy = redundancy
		
	def search_hkl(self, hkl):
		frame_ar = np.array([],dtype=np.int)
		inte_ave_ar = np.array([])
		inte_sum_ar = np.array([])
		inte_ave_adj_ar = np.array([])
		inte_sum_adj_ar = np.array([])
		pixel_total_no_ar = np.array([])
		r_obj_list=[]
		for f in self.frame_obj_list:
			frame = f.frame_no
			r = f.find_hkl(hkl)
			if r is not None:
				frame_ar = np.append(frame_ar, frame)
				inte_ave_ar = np.append(inte_ave_ar, r.ref_inte_ave)
				inte_sum_ar = np.append(inte_sum_ar, r.ref_inte_sum)
				inte_ave_adj_ar = np.append(inte_ave_adj_ar, r.ref_inte_ave_adj)
				inte_sum_adj_ar = np.append(inte_sum_adj_ar, r.ref_inte_sum_adj)
				pixel_total_no_ar = np.append(pixel_total_no_ar, r.pixel_total_no)
				r_obj_list.append(r)
		return {'frame_ar':frame_ar, 'inte_ave_ar':inte_ave_ar, 'inte_sum_ar':inte_sum_ar, 'pixel_total_no_ar':pixel_total_no_ar,\
					'r_obj_list':r_obj_list, 'inte_ave_adj_ar':inte_ave_adj_ar, 'inte_sum_adj_ar':inte_sum_adj_ar}

	def search_HKL(self, HKL):
		frame_ar = np.array([],dtype=np.int)
		inte_ave_ar = np.array([])
		inte_sum_ar = np.array([])
		inte_ave_cor_ar = np.array([])
		inte_ave_adj_ar = np.array([])
		inte_sum_adj_ar = np.array([])
		pixel_total_no_ar = np.array([])
		r_obj_list=[]
		for f in self.frame_obj_list:
			frame = f.frame_no
			r_l = f.find_HKL(HKL)
			#print(len(r_l))
			if len(r_l)!=0:
				for r in r_l:
					frame_ar = np.append(frame_ar, frame)
					inte_ave_ar = np.append(inte_ave_ar, r.ref_inte_ave)
					inte_sum_ar = np.append(inte_sum_ar, r.ref_inte_sum)
					inte_ave_cor_ar = np.append(inte_ave_cor_ar, r.ref_inte_ave/r.Lorf)
					inte_ave_adj_ar = np.append(inte_ave_adj_ar, r.ref_inte_ave_adj)
					inte_sum_adj_ar = np.append(inte_sum_adj_ar, r.ref_inte_sum_adj)
					pixel_total_no_ar = np.append(pixel_total_no_ar, r.pixel_total_no)
					r_obj_list.append(r)
		return {'frame_ar':frame_ar, 'inte_ave_ar':inte_ave_ar, 'inte_sum_ar':inte_sum_ar, 'inte_ave_cor_ar':inte_ave_cor_ar,'pixel_total_no_ar':pixel_total_no_ar,\
				'r_obj_list':r_obj_list, 'inte_ave_adj_ar':inte_ave_adj_ar, 'inte_sum_adj_ar':inte_sum_adj_ar}

	def adj_k_in(self,frame_no):
		# add the k_in_adj attribute to pixels
		ind = (self.frame_arry==frame_no).nonzero()[0]
		if len(ind)==0:
			sys.exit('no frame obj found,check adj_k_in method')
		ind = ind[0]
		f = self.frame_obj_list[ind]
        
		k_c0 = np.array([(1556-1594)*75e-6,(1748-1764)*75e-6,0.1291])
		k_c0 = k_c0/np.linalg.norm(k_c0)*k0

		############################################
		# read the pu_arry from the pupil image
		#with h5py.File('/home/lichufen/CCB_ind/scan_corrected_00135.h5','r') as fi:
		#	pu_arry = np.array(fi['/data/data'][f.frame_no,:,:])
		###########################################
		# read the pu_arry from the INT_norm map
		#with open('INT_norm.pkl','rb') as fi:
		#	norm_dict = pk.load(fi)
		#bins_arry_x = norm_dict['bins_arry_x']
		#bins_arry_y = norm_dict['bins_arry_y']
		#INT_mean = norm_dict['INT_mean']

		for r in f.reflection_obj_list:
			#ind = (self.reference_arry[:,0:3]==np.abs(r.hkl)).all(axis=1).nonzero()[0]
			#if len(ind)!=0:
			#	ind = ind[0]
			#	r.reference_I = self.reference_arry[ind,3]
			#else:
			#	r.reference_I = np.nan

			for p in r.pixel_obj_list:
				p.k_in_adj = p.k_in
				#############
				## adjust the k_in accoding to the "big xtal effect"
				#p.k_in_adj = np.array([0.795*p.k_in[0],0.795*p.k_in[1],np.sqrt(k0**2-(0.795*p.k_in[0])**2-(0.795*p.k_in[1])**2)])
            	#######
            	###### compute the image pu value according to the pu_arry
				#x_ind, y_ind = (p.k_in_adj/p.k_in_adj[2]*0.1291)[0:2]
				#x_ind = np.rint((x_ind/75e-6)+1594).astype(np.int)
				#y_ind = np.rint((y_ind/75e-6)+1764).astype(np.int)
				#pu = pu_arry[y_ind-2:y_ind+2,x_ind-2:x_ind+2].mean()/1e6


				#if (pu>2):
				#	p.pu = pu
				#else:
				#	p.pu = np.nan
            	######################
            	## compute the pu value according to the reflection INT_norm map
				#bins_ind_x = np.digitize(p.k_in_adj[0],bins_arry_x)
				#bins_ind_y = np.digitize(p.k_in_adj[1],bins_arry_y)    
				#pu = INT_mean[bins_ind_y-1,bins_ind_x-1]

				#if (pu>0.1):
				#	p.pu = pu
				#else:
				#	p.pu = np.nan
				#####################
				## compute the pu value accordint to the frame norm map
				bins_ind_x = np.digitize(p.k_in_adj[0],f.bins_arry_x)
				bins_ind_y = np.digitize(p.k_in_adj[1],f.bins_arry_y)
				pu = f.INT_n_mean[bins_ind_y-1,bins_ind_x-1]
				
				if (pu>=0)*(pu<=2e8):
					p.pu = pu
				else:
					p.pu = np.nan
				###########################
				
				p.inte_adj = (p.inte-r.bkg)/p.pu
				

			r.pu_value = np.nanmean(np.array([p.pu for p in r.pixel_obj_list]))
			r.get_inte_adj()
		return


	def adj_frame(self):
		for frame_no in self.frame_arry:
			self.adj_k_in(frame_no)
		return
	
	def merge_all_hkl(self):
		out_put_arry=np.zeros((self.hkl_arry.shape[0],16))
		for m in range(self.hkl_arry.shape[0]):
			hkl = self.hkl_arry[m,:]
			dd = self.search_hkl(hkl)
			out_put_arry[m,0:3] = hkl
			out_put_arry[m,3] = dd['inte_ave_ar'].shape[0]
			out_put_arry[m,4] = dd['inte_ave_ar'].mean()
			#out_put_arry[m,4] = dd['inte_ave_ar'].mean()
			out_put_arry[m,5] = dd['inte_ave_ar'].std()
			out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			out_put_arry[m,7] = dd['inte_sum_ar'].mean()
			out_put_arry[m,8] = dd['inte_sum_ar'].std()
			out_put_arry[m,9] = out_put_arry[m,7]/out_put_arry[m,8]
			out_put_arry[m,10] = dd['inte_ave_adj_ar'].mean()
			out_put_arry[m,11] = dd['inte_ave_adj_ar'].std()
			out_put_arry[m,12] = out_put_arry[m,10]/out_put_arry[m,11]
			out_put_arry[m,13] = dd['inte_sum_adj_ar'].mean()
			out_put_arry[m,14] = dd['inte_sum_adj_ar'].std()
			out_put_arry[m,15] = out_put_arry[m,13]/out_put_arry[m,14]
			print('%d out of %d hkl done!'%(m+1,self.hkl_arry.shape[0]))
		np.savetxt('all_hkl.txt',out_put_arry,fmt=['%4d','%4d','%4d','%03d','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f'])
		return
	
	def merge_all_HKL(self):
		out_put_arry=np.zeros((self.hkl_arry.shape[0],16))
		for m in range(self.HKL_arry.shape[0]):
			HKL = self.HKL_arry[m,:]
			dd = self.search_HKL(HKL)
			out_put_arry[m,0:3] = HKL
			out_put_arry[m,3] = dd['inte_ave_ar'].shape[0]
			out_put_arry[m,4] = dd['inte_ave_ar'].mean()
			out_put_arry[m,5] = dd['inte_ave_ar'].std()
			out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			out_put_arry[m,7] = dd['inte_sum_ar'].mean()
			out_put_arry[m,8] = dd['inte_sum_ar'].std()
			out_put_arry[m,9] = out_put_arry[m,7]/out_put_arry[m,8]
			out_put_arry[m,10] = dd['inte_ave_adj_ar'].mean()
			out_put_arry[m,11] = dd['inte_ave_adj_ar'].std()
			out_put_arry[m,12] = out_put_arry[m,10]/out_put_arry[m,11]
			out_put_arry[m,13] = dd['inte_sum_adj_ar'].mean()
			out_put_arry[m,14] = dd['inte_sum_adj_ar'].std()
			out_put_arry[m,15] = out_put_arry[m,13]/out_put_arry[m,14]
			print('%d out of %d HKL done!'%(m+1,self.HKL_arry.shape[0]))
		np.savetxt('all_HKL.txt',out_put_arry,fmt=['%4d','%4d','%4d','%03d','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f','%10.2f'])
		return

	def merge_all_HKL_crystfel(self,output=True):
		out_put_arry=np.zeros((self.HKL_arry.shape[0],16))
		for m in range(self.HKL_arry.shape[0]):
			HKL = self.HKL_arry[m,:]
			dd = self.search_HKL(HKL)
			out_put_arry[m,0:3] = HKL
			out_put_arry[m,3] = dd['inte_ave_ar'].shape[0]
			out_put_arry[m,4] = np.nanmean(dd['inte_ave_ar'])
			out_put_arry[m,5] = np.nanstd(dd['inte_ave_ar'])
			out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			out_put_arry[m,7] = np.nanmean(dd['inte_sum_ar'])
			out_put_arry[m,8] = np.nanstd(dd['inte_sum_ar'])
			out_put_arry[m,9] = out_put_arry[m,7]/out_put_arry[m,8]
			out_put_arry[m,10] = np.nanmean(dd['inte_ave_adj_ar'])
			out_put_arry[m,11] = np.nanstd(dd['inte_ave_adj_ar'])
			out_put_arry[m,12] = out_put_arry[m,10]/out_put_arry[m,11]
			out_put_arry[m,13] = np.nanmean(dd['inte_sum_adj_ar'])
			out_put_arry[m,14] = np.nanstd(dd['inte_sum_adj_ar'])
			out_put_arry[m,15] = out_put_arry[m,13]/out_put_arry[m,14]
			#print('%d out of %d HKL done!'%(m+1,self.HKL_arry.shape[0]))
		print('Merging Done!')
		######################
		## output the array in CrystFEL reflection list format.
		header = '''CrystFEL reflection list version 2.0
Symmetry: mmm
   h    k    l          I    phase   sigma(I)   nmeas'''
		footer = '''End of reflections
Generated by CrystFEL'''
		Crystfel_out_put_arry = np.hstack((out_put_arry[:,0:1],out_put_arry[:,1:2],out_put_arry[:,2:3],out_put_arry[:,10:11],np.zeros((out_put_arry.shape[0],1)),out_put_arry[:,11:12]/np.sqrt(out_put_arry[:,3:4]),out_put_arry[:,3:4]))
		ind = np.isnan(Crystfel_out_put_arry).any(axis=1)
		Crystfel_out_put_arry = Crystfel_out_put_arry[~ind,:]
		if output:
			np.savetxt('all_HKL_crystfel.hkl',Crystfel_out_put_arry,header=header,footer=footer,fmt=['%4d','%4d','%4d','%10.2f','%8s','%10.2f','%7d'],comments='')
			print('.hkl file output done!')
		return Crystfel_out_put_arry


	def merge_all_HKL_raw_ave_crystfel(self,correction=False,output=True):
		out_put_arry=np.zeros((self.HKL_arry.shape[0],16))
		for m in range(self.HKL_arry.shape[0]):
			HKL = self.HKL_arry[m,:]
			dd = self.search_HKL(HKL)
			out_put_arry[m,0:3] = HKL
			out_put_arry[m,3] = dd['inte_ave_ar'].shape[0]
			if correction==False:
				out_put_arry[m,4] = np.nanmean(dd['inte_ave_ar'])
				out_put_arry[m,5] = np.nanstd(dd['inte_ave_ar'])
				out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			if correction==True:
				out_put_arry[m,4] = np.nanmean(dd['inte_ave_cor_ar'])
				out_put_arry[m,5] = np.nanstd(dd['inte_ave_cor_ar'])
				out_put_arry[m,6] = out_put_arry[m,4]/out_put_arry[m,5]
			out_put_arry[m,7] = np.nanmean(dd['inte_sum_ar'])
			out_put_arry[m,8] = np.nanstd(dd['inte_sum_ar'])
			out_put_arry[m,9] = out_put_arry[m,7]/out_put_arry[m,8]
			out_put_arry[m,10] = np.nanmean(dd['inte_ave_adj_ar'])
			out_put_arry[m,11] = np.nanstd(dd['inte_ave_adj_ar'])
			out_put_arry[m,12] = out_put_arry[m,10]/out_put_arry[m,11]
			out_put_arry[m,13] = np.nanmean(dd['inte_sum_adj_ar'])
			out_put_arry[m,14] = np.nanstd(dd['inte_sum_adj_ar'])
			out_put_arry[m,15] = out_put_arry[m,13]/out_put_arry[m,14]
			#print('%d out of %d HKL done!'%(m+1,self.HKL_arry.shape[0]))
		print('Merging Done!')
		######################
		## output the array in CrystFEL reflection list format.
		header = '''CrystFEL reflection list version 2.0
Symmetry: mmm
   h    k    l          I    phase   sigma(I)   nmeas'''
		footer = '''End of reflections
Generated by CrystFEL'''
		Crystfel_out_put_arry = np.hstack((out_put_arry[:,0:1],out_put_arry[:,1:2],out_put_arry[:,2:3],out_put_arry[:,4:5],np.zeros((out_put_arry.shape[0],1)),out_put_arry[:,5:6]/np.sqrt(out_put_arry[:,3:4]),out_put_arry[:,3:4]))
		ind = np.isnan(Crystfel_out_put_arry).any(axis=1)
		Crystfel_out_put_arry = Crystfel_out_put_arry[~ind,:]
		if output:
			np.savetxt('all_HKL_raw_ave_crystfel.hkl',Crystfel_out_put_arry,header=header,footer=footer,fmt=['%4d','%4d','%4d','%10.2f','%8s','%10.2f','%7d'],comments='')
			print('.hkl file output done!')
		return Crystfel_out_put_arry


	def update_reference_arry(self,reference_arry):
		reference_arry = reference_arry[:,0:4]
		self.reference_arry = np.hstack((reference_arry[:,0:1],reference_arry[:,1:2],reference_arry[:,2:]))
		return

	def clean(self):
		del self.arry
		return

	def show_frame(self,frame_no,image_file='/asap3/petra3/gpfs/p11/2021/data/11010570/raw/scan_frames/Scan_210/Scan_210_data_000001.h5',h5path='/entry/data/data',show_hkl=True):
		with h5py.File(image_file,'r') as im:
			img_arry = np.array(im[h5path][frame_no,:,:])
		with h5py.File('/gpfs/cfel/user/lichufen/CBDXT/P11_BT/CFL_mask_scan_210.h5','r') as m:
			mask=np.array(m['/data/data']).astype(bool)
			bkg = np.array(m['/data/bkg'])
		img_arry = img_arry - bkg
		img_arry = img_arry*mask
		plt.figure(figsize=(15,15))
		#plt.imshow(img_arry*(mask.astype(np.int16)),cmap='viridis',origin='lower')
		plt.imshow(img_arry*(mask.astype(np.int16)),cmap='gray_r',origin='lower')
		plt.colorbar()
		plt.clim(0,30)
		ind = (self.frame_arry==frame_no).nonzero()[0][0]
		f_obj = self.frame_obj_list[ind]
		for r in f_obj.reflection_obj_list:
			hkl = r.hkl
			r_x_arry = np.array([p.xy[0] for p in r.pixel_obj_list])
			r_y_arry = np.array([p.xy[1] for p in r.pixel_obj_list])
			r_cen_x = r_x_arry.mean()
			r_cen_y = r_y_arry.mean()
			plt.scatter(r_cen_x,r_cen_y,edgecolors='r',facecolors='none')
			plt.scatter(r_x_arry,r_y_arry,s=1,marker='x')
			if show_hkl:
				plt.annotate('(%d,%d,%d)'%(hkl[0],hkl[1],hkl[2]),color='r',fontsize=15,xy=(r_cen_x,r_cen_y),xycoords='data')
		title_Str=image_file+'\nframe: %d '%(frame_no)
		plt.title(title_Str)
		plt.show()
		return 

	def get_reflection_norm(self,hkl,show=False):
		
		k_in_x = []
		k_in_y = []
		r_obj_list=[]
		Int = []
		for f in self.frame_obj_list:
			frame = f.frame_no
			r = f.find_hkl(hkl)
			if r is not None:
				r_obj_list.append(r)

		for r in r_obj_list:
			k_in_x = k_in_x + [p.k_in[0] for p in r.pixel_obj_list]
			k_in_y = k_in_y + [p.k_in[1] for p in r.pixel_obj_list]
			Int = Int + [p.inte for p in r.pixel_obj_list]
		Int_norm = np.array(Int)/(np.array(Int).max()+sys.float_info.epsilon)
		Int_norm = Int_norm.tolist()
		if show:
			plt.figure()
			plt.scatter(k_in_x,k_in_y,c=Int_norm,cmap='jet',s=3)
			plt.colorbar()
			plt.show()
		return k_in_x, k_in_y, Int, Int_norm

	def get_batch_norm(self,rank,bins=200,show=False,save=True):
		ind = np.argsort(self.redundancy)
		ind = ind[::-1]
		K_in_x = [] 
		K_in_y = []
		INT_norm = []
		for m in range(rank):
			hkl = self.hkl_arry[ind[m]]
			k_in_x,k_in_y,Int,Int_norm = self.get_reflection_norm(hkl)
			K_in_x = K_in_x + k_in_x
			K_in_y = K_in_y + k_in_y
			INT_norm = INT_norm + Int_norm
		K_in_x = np.array(K_in_x)
		K_in_y = np.array(K_in_y)
		INT_norm = np.array(INT_norm)
		bins_arry_x = np.linspace(-10e8,10e8,bins+1)
		bins_arry_y = np.linspace(-10e8,10e8,bins+1)
		bins_ind_x = np.digitize(K_in_x,bins_arry_x)
		bins_ind_y = np.digitize(K_in_y,bins_arry_y)
		INT_arry = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
		counter = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
		for m in range(K_in_x.shape[0]):
			INT_arry[bins_ind_y[m]-1,bins_ind_x[m]-1]+=INT_norm[m]
			counter[bins_ind_y[m]-1,bins_ind_x[m]-1]+=1
		KX,KY = np.meshgrid(bins_arry_x,bins_arry_y)
		INT_mean = INT_arry/(counter+sys.float_info.epsilon)
		if show:
			plt.figure(figsize=(3,3))
			plt.scatter(K_in_x,K_in_y,c=INT_norm,cmap='jet',s=3)
			plt.colorbar()
			plt.figure()
			plt.pcolor(KX,KY,INT_mean,edgecolors='face')
			#plt.clim(0,1)
			plt.colorbar()
			plt.show()
		norm_dict = {'bins_arry_x':bins_arry_x,'bins_arry_y':bins_arry_y,'KX':KX,'KY':KY,'INT_mean':INT_mean,'counter':counter}
		if save:
			with open('INT_norm.pkl','wb') as f:
				pk.dump(norm_dict,f)
		
		return norm_dict

	def get_frame_norm_raw(self,frame_no,bins=200):
		k_in_x = []
		k_in_y = []
		Int_n = []
		ind = (self.frame_arry==frame_no).nonzero()[0]
		if len(ind)==0:
			sys.exit('no frame obj found')
		ind = ind[0]
		f = self.frame_obj_list[ind]
		
		for r in f.reflection_obj_list:
			ind = (self.reference_arry[:,0:3]==np.abs(r.hkl)).all(axis=1).nonzero()[0]
			if len(ind)!=0:
				ind = ind[0]
				r.reference_I = self.reference_arry[ind,3]
			else:
				r.reference_I = np.nan
			if not np.isnan(r.reference_I):
				k_in_x = k_in_x + [p.k_in[0] for p in r.pixel_obj_list]
				k_in_y = k_in_y + [p.k_in[1] for p in r.pixel_obj_list]
				k_out_m = np.nanmean(np.array([p.k_out for p in r.pixel_obj_list]),axis=0)
				q_vec = r.hkl*(1e10/np.array([15.79,22.50,25.70]))
				Lorf = CCB_pat_sim.get_Lorf(q_vec,k0,k_out_m)
				#Lorf = 1

				Int_n = Int_n + [(p.inte-r.bkg)/(r.reference_I*Lorf) for p in r.pixel_obj_list]
		f.k_in_x = k_in_x
		f.k_in_y = k_in_y
		f.Int_n = Int_n

		k_in_x_arry = np.array(k_in_x)
		k_in_y_arry = np.array(k_in_y)
		Int_n = np.array(Int_n)
		bins_arry_x = np.linspace(-10e8,10e8,bins+1)
		bins_arry_y = np.linspace(-10e8,10e8,bins+1)
		bins_ind_x = np.digitize(k_in_x_arry,bins_arry_x)
		bins_ind_y = np.digitize(k_in_y_arry,bins_arry_y)
		INT_arry = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
		counter = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
		for m in range(k_in_x_arry.shape[0]):
			if (Int_n[m]>0)*(Int_n[m]<2e8):
				INT_arry[bins_ind_y[m]-1,bins_ind_x[m]-1]+=Int_n[m]
				counter[bins_ind_y[m]-1,bins_ind_x[m]-1]+=1
		KX,KY = np.meshgrid(bins_arry_x,bins_arry_y)
		INT_n_mean = INT_arry/(counter+sys.float_info.epsilon)
		
		f.bins_arry_x = bins_arry_x
		f.bins_arry_y = bins_arry_y
		#f.bins_ind_x = bins_ind_x
		#f.bins_ind_y = bins_ind_y
		f.INT_n_mean = INT_n_mean
		
		return KX,KY,INT_n_mean

	def get_all_frames_norm(self,bins=200):
		norm_all = []
		for frame_no in self.frame_arry:
			KX,KY,INT_n_mean = self.get_frame_norm_raw(frame_no,bins=bins)
			norm_all = norm_all + INT_n_mean.reshape(-1,).tolist()
		norm_all = np.array(norm_all)
		
		fix_factor = 2e-3/norm_all.mean()

		for f in self.frame_obj_list:
			f.INT_n_mean = f.INT_n_mean*fix_factor
		return
	
	def show_frame_norm(self,frame_no):
		ind = (self.frame_arry==frame_no).nonzero()[0]
		if len(ind)==0:
			sys.exit('no frame obj found')
		ind = ind[0]
		f = self.frame_obj_list[ind]
		
		KX,KY = np.meshgrid(f.bins_arry_x,f.bins_arry_y)	
		
		with h5py.File('/asap3/petra3/gpfs/p11/2021/data/11010570/raw/scan_frames/Scan_210/Scan_210_data_000001.h5','r') as im:
			ref_image = np.array(im['/entry/data/data'][frame_no,:,:])
		with h5py.File('/gpfs/cfel/user/lichufen/CBDXT/P11_BT/CFL_mask_scan_210.h5','r') as m:
			mask=np.array(m['/data/data']).astype(bool)
			bkg = np.array(m['/data/bkg'])
		ref_image = np.where(ref_image<1e8,ref_image,0)
		#img_arry = img_arry*mask
 

		f.D = np.zeros_like(f.INT_n_mean)
		for ind,value in np.ndenumerate(f.D):
			k_inx = f.bins_arry_x[ind[1]]
			k_iny = f.bins_arry_y[ind[0]]
			k_inz = np.sqrt(k0**2-k_inx**2-k_iny**2)
			P_value = CCB_pat_sim.get_P(ref_image,[k_inx,k_iny,k_inz])
			if P_value>=1:
				f.D[ind] = f.INT_n_mean[ind]/(P_value+sys.float_info.epsilon)
			else:
				f.D[ind] = 0 

		plt.figure(figsize=(3,3))
		#plt.pcolor(KX,KY,f.INT_n_mean)
		#plt.clim(0,0.2e-1)
		print(f.D.mean())
		plt.pcolor(KX,KY,f.D)
		#plt.clim(0,0.7e-1)
		plt.clim(0,1e2*f.D.mean())
		plt.colorbar()
		plt.xlim(-10e8,10e8)
		plt.ylim(-10e8,10e8)
		plt.xlabel(r'$K_{in,x}(m^{-1})$')
		plt.ylabel(r'$K_{in,y}(m^{-1})$')
		plt.gca().set_aspect('equal')
		plt.draw()
		plt.title('frame: %d'%(f.frame_no))
		plt.tight_layout()
		plt.show()

		return

class Frame:
		
	def __init__(self, dataset_obj, frame_no):
		self.arry = dataset_obj.arry[dataset_obj.arry[:,0]==int(frame_no),:]
		self.frame_no = frame_no
		reference_arry = np.genfromtxt('/home/lichufen/CCB_ind/ethc_mk.pdb.hkl',skip_header=3,skip_footer=2,usecols=(0,1,2,3))
		self.reference_arry = np.hstack((reference_arry[:,0:1],reference_arry[:,1:2],reference_arry[:,2:]))
		
		#self.reference_arry[:,3] = 1

		self.get_reflections()
		
		self.total_inte = self.arry[:,3].sum()
		self.clean()
		
	def get_reflections(self):
		hkl_arry = np.unique(self.arry[:,4:7], axis=0)
		self.hkl_arry = hkl_arry.astype(np.int)
		self.hkl_total_no = self.hkl_arry.shape[0]
		self.reflection_obj_list = []
		for i in range(self.hkl_total_no):
			hkl = self.hkl_arry[i,:]
			reflection = Reflection(self,hkl)
			self.reflection_obj_list.append(reflection)
	
	def find_hkl(self, hkl):
		reflection_obj = None
		ind = (self.hkl_arry==np.array(hkl).astype(np.int)).all(axis=1).nonzero()[0]
		if len(ind)!=0:
			reflection_obj = self.reflection_obj_list[ind[0]]
		return reflection_obj

	@staticmethod
	def extend_HKL(HKL):
		HKL = np.array(HKL,dtype=np.int)
		H = HKL[0]
		K = HKL[1]
		L = HKL[2]
		ext_arry = np.array([[H,K,L],[-H,K,L],[H,-K,L],[H,K,-L],[H,-K,-L],[-H,K,-L],[-H,-K,L],[-H,-K,-L]])
		ext_arry = np.unique(ext_arry,axis=0)
		return ext_arry


	def find_HKL(self, HKL):
		ext_arry = Frame.extend_HKL(HKL)
		#print(ext_arry)
		reflection_obj = []
		
		for m in range(ext_arry.shape[0]):
			hkl = ext_arry[m,:]
			ind = (self.hkl_arry==np.array(hkl).astype(np.int)).all(axis=1).nonzero()[0]
			if len(ind)!=0:
				reflection_obj.append(self.reflection_obj_list[ind[0]])
		return reflection_obj


	def clean(self):
		del self.arry
	
				

	def get_diff_arry(self,bins=200,thld=1e0,plot=False,save=False):
		diff_eff_arry = []
		kx_arry = []
		ky_arry = []
		for r in self.reflection_obj_list:
			for p in r.pixel_obj_list:
				p.diffraction_eff = p.inte/r.reference_I
				if (not np.isnan(p.diffraction_eff)) and (p.diffraction_eff<thld):
					diff_eff_arry.append(p.diffraction_eff)
					kx_arry.append(p.k_in[0])
					ky_arry.append(p.k_in[1])
		
		kx_arry = np.array(kx_arry)
		ky_arry = np.array(ky_arry)
		diff_eff_arry = np.array(diff_eff_arry)			
		bins_arry_x=np.linspace(-10e8,10e8,bins+1)
		bins_arry_y=np.linspace(-10e8,10e8,bins+1)
		bins_ind_x=np.digitize(kx_arry,bins_arry_x)	
		bins_ind_y=np.digitize(ky_arry,bins_arry_y)
		Int_arry = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))
		counter = np.zeros((bins_arry_x.shape[0],bins_arry_y.shape[0]))	
		for m in range(kx_arry.shape[0]):
			Int_arry[bins_ind_x[m]-1,bins_ind_y[m]-1]+=diff_eff_arry[m]
			counter[bins_ind_x[m]-1,bins_ind_y[m]-1]+=1
		Int_arry = Int_arry/(counter+sys.float_info.epsilon)
		KX,KY=np.meshgrid(bins_arry_x,bins_arry_y)
		
		
		diff_dict = {'kx_arry':kx_arry,'ky_arry':ky_arry,'diff_eff_arry':diff_eff_arry,'bins_arry_x':bins_arry_x,'bins_arry_y':bins_arry_y,'KX':KX,'KY':KY,'Int_arry':Int_arry}
		
		if save:
			file_name_base = 'Dif_EFF_fr%d.pkl'%(self.frame_no)
			with open(file_name_base,'wb') as f:
				pk.dump(diff_dict,f)
			
		if plot:
			fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,4))
			plt.colorbar(im,ax=ax[1])
			ax[1].axis('equal')
			ax[1].set_xlabel('k_in_x(A^-1)')
			ax[1].set_ylabel('k_in_y(A^-1)')
			ax[1].set_title('diffraction efficiency,linear\nframe: %d'%(self.frame_no))
			ax[2].hist(np.log10(diff_eff_arry),bins=200)
			ax[2].set_label('log10(diffraction efficiency)')
			ax[2].set_title('histogram')
			plt.show()


		return diff_dict
	


class Reflection:
	def __init__(self, frame_obj, hkl):
		hkl = np.array(hkl)
		self.hkl = hkl
		self.arry = frame_obj.arry[(frame_obj.arry[:,4:7]==hkl).all(axis=1),:]
		self.pixel_total_no = self.arry.shape[0]
		self.ref_inte_sum = self.arry[:,3].sum()
		self.ref_inte_ave = self.ref_inte_sum/self.pixel_total_no
		self.major_axis_length = self.arry[0,13]
		self.bkg = self.arry[0,15]
		self.get_pixels()
		self.clean()

	def get_pixels(self):
		self.pixel_obj_list = []
		for i in range(self.pixel_total_no):
			pixel = Pixel(self,i)
			self.pixel_obj_list.append(pixel)

	def clean(self):
		del self.arry

	def get_inte_adj(self):
		if hasattr(self.pixel_obj_list[0], 'inte_adj'):
			###########################
			# pixel based pupil correction
			ref_inte_sum_adj = np.nansum(np.array([p.inte_adj for p in self.pixel_obj_list]))/self.major_axis_length
			ref_inte_ave_adj = np.nanmean(np.array([p.inte_adj for p in self.pixel_obj_list]))
			###########################
			# reflection based pupil corretion
			#ref_inte_sum_adj = self.ref_inte_sum/self.pu_value
			#ref_inte_ave_adj = self.ref_inte_ave/self.pu_value
			####################################################
			# Add Lorentz correction for intensity integration.
			q_vec = self.hkl*(1e10/np.array([15.79,22.50,25.70]))
			k_out_m = np.nanmean(np.array([p.k_out for p in self.pixel_obj_list]),axis=0)
			Lorf = CCB_pat_sim.get_Lorf(q_vec,k0,k_out_m)
			self.Lorf = Lorf
			ref_inte_sum_adj = ref_inte_sum_adj/Lorf
			ref_inte_ave_adj = ref_inte_ave_adj/Lorf
			####################################################
			self.ref_inte_sum_adj = ref_inte_sum_adj
			self.ref_inte_ave_adj = ref_inte_ave_adj
		else:
			sys.exit('no attribute inte_adj for Pixel object')
		

class Pixel:
	pixel_size = 75e-6 # in m
	
	def __init__(self, reflection_obj, pixel_id):
		self.pixel_id = pixel_id
		self.arry = reflection_obj.arry[pixel_id,:]
		self.xy = self.arry[1:3]
		self.inte = self.arry[3]
		self.k_out = self.arry[7:10]
		self.k_in = self.arry[10:13]
	


if __name__=='__main__':
	K_map_file_name = os.path.abspath(sys.argv[1])
	#rank = int(sys.argv[2]) # the rank of the hkl accoridng to redundancy in ascending order
	h = int(sys.argv[2])
	k = int(sys.argv[3])
	l = int(sys.argv[4])
	dset = Dataset(K_map_file_name)
	ind = np.argsort(dset.redundancy)
	#hkl = dset.hkl_arry[ind[rank]]
	hkl = (h,k,l)
	dd = dset.search_hkl(hkl)


	
##################################################
	
##################################################
	#dd = dset.search_HKL(hkl)

	#for frame in dset.frame_obj_list:
		#print('frame %d has %d reflctions'%(frame.frame_no,frame.hkl_total_no))
	print('This dataset has {0:d} unique HKL measured'.format(dset.HKL_arry.shape[0]))
	#[print('HKL: ',*dset.HKL_arry[m,:],'red: ',dset.redundancy[m]) for m in range(dset.HKL_arry.shape[0])]
	#plt.figure(figsize=(5,5))
	#total_inte_arry = np.array([f.total_inte/f.hkl_total_no for f in dset.frame_obj_list])
	#plt.plot(dset.frame_arry, total_inte_arry)
	#plt.xlabel('frame')
	#plt.ylabel('total signal intensity per reflection')
	#plt.show()
	fig, ax = plt.subplots(nrows=1, ncols=2 ,figsize=(12,5))
	plt.title('hkl: (%2d,%2d,%2d) '%(hkl[0],hkl[1],hkl[2]))
	#ax[0].plot(dd['frame_ar'],dd['inte_ave_ar'],'bx-')
	ax[0].plot(dd['frame_ar'],dd['inte_ave_adj_ar'],'bx')
	#ax[1].plot(dd['frame_ar'],dd['inte_sum_ar'],'rx-')
	ax[1].plot(dd['frame_ar'],dd['inte_sum_adj_ar'],'rx')
	
	#ax[2].plot(dd['frame_ar'],dd['pixel_total_no_ar'],'kx-')
	ax[0].set_xlabel('frame')
	ax[1].set_xlabel('frame')
	#ax[2].set_xlabel('frame')
	ax[0].set_ylabel('average intensity_adj')
	ax[1].set_ylabel('sum intensity_adj')
	#ax[2].set_ylabel('# of pixels')
	plt.axis('tight')
	
	fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
	labels=['frame'+str(frame_no) for frame_no in dd['frame_ar']]
	plt.title('hkl: (%2d,%2d,%2d) '%(hkl[0],hkl[1],hkl[2]))
	ax[0].plot(dd['frame_ar'],dd['inte_ave_ar'],'bx-')
	ax[1].plot(dd['frame_ar'],dd['inte_sum_ar'],'rx-')
	ax[0].set_xlabel('frame')
	ax[1].set_xlabel('frame')
	ax[0].set_ylabel('average intensity')
	ax[1].set_ylabel('sum intensity')
	plt.axis('tight')

	fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
	labels=['frame'+str(frame_no) for frame_no in dd['frame_ar']]
	for r in dd['r_obj_list']:
		k_out_arry = np.array([]).reshape(-1,3)
		k_in_arry = np.array([]).reshape(-1,3)
		labels.append
		for p in r.pixel_obj_list:
			k_out_arry = np.vstack((k_out_arry ,p.k_out.reshape(1,3)))
			k_in_arry = np.vstack((k_in_arry, p.k_in_adj.reshape(1,3)))
		ax[0].scatter(k_out_arry[:,0],k_out_arry[:,1],s=1,marker='x')
		ax[1].scatter(k_in_arry[:,0],k_in_arry[:,1],s=5,marker='x')
	ax[0].set_title('k_out scatter for hkl: (%2d, %2d, %2d)'%(hkl[0],hkl[1],hkl[2]))
	ax[1].set_title('k_in scatter for hkl: (%2d, %2d, %2d)'%(hkl[0],hkl[1],hkl[2]))
	ax[0].axis('equal')
	ax[1].axis('equal')
	ax[0].legend(labels)
	ax[1].legend(labels)
	plt.show()



