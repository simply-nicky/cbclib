'''
CCB_kmap.py consists of the functions to evaluate the 
K_in and K_out wave-vectors for each pixel of the
each diffraction streak detected.

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
import gen_match_figs as gm
import CCB_streak_det
import matplotlib
matplotlib.use('TkAgg') # To be adjusted for teh batch job mode.
import matplotlib.pyplot as plt

OR_mat = np.genfromtxt('../../OR.txt')
OR_mat=OR_mat/1.0

###################
# for expanding lattice constants
expanding_const = 1
OR_mat = OR_mat/expanding_const
##################


E_ph=17.4 #in keV
#wave_len=12.40/E_ph #in Angstrom
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
k0 = 1/wave_len
k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),np.sqrt(k0**2-(3e8)**2-(2.2e8)**2)*np.ones((1500,1))))
k_cen = k_cen/(np.linalg.norm(k_cen,axis=1).reshape(-1,1))*1/wave_len
pix_size = 75e-6

def get_K_frame(exp_img_file,frame,res_file='/home/lichufen/CCB_ind/Best_GA_res.txt',thld=10,min_pix=10):
	'''
	get_k_frame,for each frame,
	 returns the k_in, k_out, HkL_in along with other info from the streak detection.
	'''
	if 'sim' not in exp_img_file:
		label_filtered_sorted,weighted_centroid_filtered,props,exp_img,all_labels=CCB_streak_det.single_peak_finder(exp_img_file,frame,thld=thld,min_pix=min_pix,mask_file='/gpfs/cfel/user/lichufen/CBDXT/P11_BT/CFL_mask_scan_210.h5',interact=False)
	else:
		label_filtered_sorted,weighted_centroid_filtered,props,exp_img,all_labels=CCB_streak_det.single_peak_finder(exp_img_file,frame,thld=thld,min_pix=min_pix,mask_file='None',interact=False)
	
	#####################################
	print('hit finding done')

	streak_ind=label_filtered_sorted-1
	res_arry=gm.read_res(res_file)
	ind=np.where(res_arry[:,0]==frame)[0][0]
	frame=int(res_arry[ind,0])
	theta=res_arry[ind,1]
	phi=res_arry[ind,2]
	alpha=res_arry[ind,3]
	cam_len=res_arry[ind,4]
	k_out_osx=res_arry[ind,5]
	k_out_osy=res_arry[ind,6]

	num_s=weighted_centroid_filtered.shape[0]
	end_point1 = np.array([[props[label-1].coords.min(axis=0)[0], props[label-1].coords.min(axis=0)[1]] if props[label-1].orientation>=0 else [props[label-1].coords.min(axis=0)[0], props[label-1].coords.max(axis=0)[1]]  for label in label_filtered_sorted])
	end_point2 = np.array([[props[label-1].coords.max(axis=0)[0], props[label-1].coords.max(axis=0)[1]] if props[label-1].orientation>=0 else [props[label-1].coords.max(axis=0)[0], props[label-1].coords.min(axis=0)[1]]  for label in label_filtered_sorted])
	end_vector1 = np.hstack(((end_point1[:,-1::-1]-np.array([(1908-k_out_osx*0.2/cam_len/(75e-6)),(2207+k_out_osy*0.2/cam_len/(75e-6))]).reshape(-1,2))*pix_size/cam_len,np.ones((num_s,1))))
	end_vector2 = np.hstack(((end_point2[:,-1::-1]-np.array([(1908-k_out_osx*0.2/cam_len/(75e-6)),(2207+k_out_osy*0.2/cam_len/(75e-6))]).reshape(-1,2))*pix_size/cam_len,np.ones((num_s,1))))
	end_vector1 = end_vector1/(np.linalg.norm(end_vector1,axis=-1).reshape(-1,1))
	end_vector2 = end_vector2/(np.linalg.norm(end_vector2,axis=-1).reshape(-1,1))

	end_vector1[:,0] *= -1
	end_vector2[:,0] *= -1

	diff_vector = end_vector2 - end_vector1
	diff_vector = diff_vector/(np.linalg.norm(diff_vector,axis=-1).reshape(-1,1))
	Diff_vector = diff_vector*(1/wave_len)




	num_streak=streak_ind.shape[0]
	K_out_arry=np.zeros((num_streak,3))
	Q_arry=np.zeros((num_streak,3))
	Pxy_cen_arry=np.zeros((num_streak,2))
	
	bkg_val = np.zeros((num_streak,))

	for ind,s_ind in np.ndenumerate(streak_ind):
		ind=ind[0]
		Py_cen,Px_cen=props[s_ind].centroid
		Pxy_cen_arry[ind,:]=np.array([Px_cen,Py_cen])
		x_cen=(Px_cen-(1908-k_out_osx*0.2/cam_len/(75e-6)))*75e-6
		
		x_cen = -x_cen	
	
		y_cen=(Py_cen-(2207+k_out_osy*0.2/cam_len/(75e-6)))*75e-6
		#z_cen=0.1025*cam_len
		z_cen=0.20/cam_len
		k_cen_dir=np.array([x_cen,y_cen,z_cen])/np.linalg.norm(np.array([x_cen,y_cen,z_cen]))
		k_out_cen=(1/wave_len)*k_cen_dir
		Q_cen=k_out_cen-k_cen[frame,:].reshape(-1,) # the first rough estimate of Q vector.
		
		K_out_arry[ind,:]=k_out_cen
		Q_arry[ind,:]=Q_cen
		binary_im = (all_labels==label_filtered_sorted[ind])
		dilated_bin_im = morphology.binary_dilation(binary_im,morphology.disk(3))
		bkg_bin_im = dilated_bin_im.astype(np.int) - binary_im.astype(np.int)
		bkg_props = measure.regionprops(bkg_bin_im.astype(np.int),exp_img)
		bkg = bkg_props[0].mean_intensity
		bkg_val[ind] = bkg
		
	
	########################
	print('vectors and bkg done')
	
	frac_offset=np.array([0,0,0])
	OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(0.5*frame)@OR_mat
	HKL_frac,HKL_int,Q_int,Q_resid=CCB_ref.get_HKL8(OR,Q_arry,frac_offset) #the shape of HKL_int and Q_int,(num,3,8) 
	Delta_k, Dist, Dist_1=CCB_ref.exctn_error8_nr(k_cen[frame,:],OR,Q_arry,Q_int,frac_offset,E_ph)
	
	K_in_arry = K_out_arry.reshape(-1,3,1)- Q_int
	K_in_mag = np.linalg.norm(K_in_arry-k_cen[frame,:].reshape(1,3,1),axis=1)
	ind=np.argsort(K_in_mag,axis=1)
	#ind=np.argsort(Dist_1,axis=1)
	ind=np.array([ind[m,0] for m in range(ind.shape[0])])


	
	#HKL_table, K_in_table, K_out_table = CCB_pat_sim.pat_sim_q(k_cen[frame,:],OR,0.8)
	############################
	#print('no pat_sim_q done')
	#K_in_pred_s, K_out_pred_s = CCB_pat_sim.kout_pred(OR,k_cen[frame,:],HKL_table[:,0:3])
	#Min_delta_arry = np.zeros((K_out_arry.shape[0],))
	#HKL_exp = np.zeros((K_out_arry.shape[0],3))
	#for id, k_out_exp in enumerate(K_out_arry):
	#	delta_k_out = k_out_exp.reshape(-1,3) - K_out_pred_s
	#	np.linalg.norm(delta_k_out,axis=1)
	#	ind_delta = np.argsort(np.linalg.norm(delta_k_out,axis=1))
	#	min_delta = np.linalg.norm(delta_k_out[ind_delta[0]])
	#	Min_delta_arry[id] = min_delta
	#	print(min_delta)
	#	HKL_exp[id] = HKL_table[ind_delta[0],0:3]
		
	############################
	#print('no delta_k_out done')
	HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
	K_in_arry = np.array([K_in_arry[m,:,ind[m]] for m in range(K_in_arry.shape[0])])
	pupil_valid = np.array([CCB_pat_sim.pupil_func(k_in) for k_in in K_in_arry])
	#pupil_valid=True
	#match_valid = Min_delta_arry<=1e8
	#pupil_valid = pupil_valid*match_valid

	K_pix_arry_all=np.array([]).reshape(-1,16)
	for ind, s_ind in np.ndenumerate(streak_ind):
		#ind=ind[0]
		#if not pupil_valid[ind]:
		#	print('outlier')
		#	continue
		num_pix=props[s_ind].coords.shape[0]
		maj_len = props[s_ind].major_axis_length
		mino_len = props[s_ind].minor_axis_length
		K_pix_arry=np.zeros((num_pix,16))	
		K_pix_arry[:,0]=int(frame)
		K_pix_arry[:,1]=props[s_ind].coords[:,1] # x coordinate of the pixel
		K_pix_arry[:,2]=props[s_ind].coords[:,0] # y coordinate of the pixel
		K_pix_arry[:,3]=exp_img[K_pix_arry[:,2].astype(np.int),K_pix_arry[:,1].astype(np.int)] # intensity of the pixel
		K_pix_arry[:,4:7]=HKL_int[ind,:]
		x_pix=(K_pix_arry[:,1]-(1908-k_out_osx*0.2/cam_len/(75e-6)))*75e-6

		x_pix = -x_pix

		y_pix=(K_pix_arry[:,2]-(2207+k_out_osy*0.2/cam_len/(75e-6)))*75e-6
		z_pix=np.ones((num_pix,))*0.20/cam_len
		k_pix_cen_dir=np.hstack((x_pix.reshape(-1,1),y_pix.reshape(-1,1),z_pix.reshape(-1,1)))
		k_pix_cen_dir=k_pix_cen_dir/np.linalg.norm(k_pix_cen_dir,axis=-1).reshape(-1,1)
		K_pix_arry[:,7:10]=(1/wave_len)*k_pix_cen_dir
		OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(0.5*frame)@OR_mat
		Q=OR@(HKL_int[ind,:].reshape(3,1))
		#print(Q)
		K_pix_arry[:,10:13]=K_pix_arry[:,7:10]-Q.reshape(1,3)
		K_pix_arry[:,13] = maj_len
		K_pix_arry[:,14] = mino_len
		K_pix_arry[:,15] = bkg_val[ind]
	######	col0:frame, col1~3: x,y,I, col4~6:HKL, col7~9:kout, col10~12:k_in, col13:major_axis_length
		kin_val = np.array([CCB_pat_sim.pupil_func(k_in) for k_in in K_pix_arry[:,10:13]])
		if (kin_val.sum()/kin_val.shape[0])<0.8:
			print('outlier',K_pix_arry[0,4:7])
			#print(kin_val)
			continue
		K_pix_arry_all=np.vstack((K_pix_arry_all,K_pix_arry))
		
		#kin_valid = [CCB_pat_sim.pupil_func(k_in) for k_in in K_pix_arry_all[:,10:13]]
		#K_pix_arry_all = K_pix_arry_all[kin_valid]
	###########
	print('k_pix_arry_all done')
	return K_pix_arry_all, HKL_int, Pxy_cen_arry, OR


def HKL_patch():
	pass
	return K_in, HKL



def K_output_frame(K_pix_arry_all):

	#frame=int(K_pix_arry_all[0,0])
	
	#f.open('K_map_fr%d.txt'%(frame),'w')
	np.savetxt('K_map_fr%d.txt'%(frame),K_pix_arry_all,fmt=['%3d','%7.1f','%7.1f','%7.1f','%3d','%3d','%3d','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%7.1f','%7.1f','%7.1f'])
	return None

if __name__=='__main__':

	exp_img_file=os.path.abspath(sys.argv[1])
	res_file=os.path.abspath(sys.argv[2])
	start_frame=int(sys.argv[3])
	end_frame=int(sys.argv[4])
	thld=int(sys.argv[5])
	min_pix=int(sys.argv[6])
	for frame in range(start_frame,end_frame+1):
		K_pix_arry_all, HKL_int, Pxy_cen_arry, OR=get_K_frame(exp_img_file,frame,res_file=res_file,thld=thld,min_pix=min_pix)
		K_output_frame(K_pix_arry_all)
		print('frame %d done'%(frame))

