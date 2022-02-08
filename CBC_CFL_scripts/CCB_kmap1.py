'''
CCB_kmap1.py consists of the functions to evaluate the 
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

#OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
#[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
#[0.00000000e+00,0.00000000e+00,4.00000000e+08]])
#OR_mat=OR_mat/1.03

OR_mat = np.genfromtxt('../../OR.txt')
OR_mat=OR_mat/1.0


E_ph=17.4 #in keV
#wave_len=12.40/E_ph #in Angstrom
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m

k0 = 1/wave_len
k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),np.sqrt(k0**2-(3e8)**2-(2.2e8)**2)*np.ones((1500,1))))
k_cen = k_cen/(np.linalg.norm(k_cen,axis=1).reshape(-1,1))*1/wave_len


def get_K_frame(exp_img_file,frame,res_file='/home/lichufen/CCB_ind/Best_GA_res.txt',thld=10,min_pix=10):
	'''
	get_k_frame,for each frame,
	 returns the k_in, k_out, HkL_in along with other info from the streak detection.
	'''
	label_filtered_sorted,weighted_centroid_filtered,props,exp_img=CCB_streak_det.single_peak_finder(exp_img_file,frame,thld=thld,min_pix=min_pix,mask_file='/home/lichufen/CCB_ind/mask.h5',interact=False)
	streak_ind=label_filtered_sorted-1
	res_arry=gm.read_res(res_file)
	ind=np.where(res_arry[:,0]==frame)[0][0]
	frame=res_arry[ind,0]
	theta=res_arry[ind,1]
	phi=res_arry[ind,2]
	alpha=res_arry[ind,3]
	cam_len=res_arry[ind,4]
	k_out_osx=res_arry[ind,5]
	k_out_osy=res_arry[ind,6]
	num_streak=streak_ind.shape[0]
	Q_arry=np.zeros((num_streak,3))
	Pxy_cen_arry=np.zeros((num_streak,2))
	for ind,s_ind in np.ndenumerate(streak_ind):
		ind=ind[0]
		Py_cen,Px_cen=props[s_ind].centroid
		Pxy_cen_arry[ind,:]=np.array([Px_cen,Py_cen])
		x_cen=(Px_cen-(1908+k_out_osx*0.2/cam_len/(75e-6)))*75e-6

		x_cen = -x_cen

		y_cen=(Py_cen-(2207+k_out_osy*0.2/cam_len/(75e-6)))*75e-6
		#z_cen=0.1025*cam_len
		z_cen=0.20/cam_len
		k_cen_dir=np.array([x_cen,y_cen,z_cen])/np.linalg.norm(np.array([x_cen,y_cen,z_cen]))
		k_out_cen=(1/wave_len)*k_cen_dir
		Q_cen=k_out_cen-k_cen[frame,:].reshape(-1,)
		Q_arry[ind,:]=Q_cen
	frac_offset=np.array([0,0,0])
	OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(0.5*frame)@OR_mat
	HKL_frac,HKL_int,Q_int,Q_resid=CCB_ref.get_HKL8(OR,Q_arry,frac_offset)
	Delta_k, Dist, Dist_1=CCB_ref.exctn_error8_nr(OR,Q_arry,Q_int,frac_offset,E_ph)
	ind=np.argsort(Dist,axis=1)

	ind=np.array([ind[m,0] for m in range(ind.shape[0])])
	Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
	HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
	K_pix_arry_all=np.array([]).reshape(-1,13)
	HKL_counter=0
	for ind, s_ind in np.ndenumerate(streak_ind):
		ind=ind[0]
		HKL=HKL_int[ind,:]
		K_in_SL, K_out_SL=CCB_pat_sim.source_line_scan(k_cen,OR,HKL,rot_ang_step=0.05,rot_ang_range=3.0)
		
		if K_in_SL.shape[0]!=0:
			K_in_cen_pred=0.5*(K_in_SL[0,:]+K_in_SL[-1,:])
			K_out_cen_pred=0.5*(K_out_SL[0,:]+K_out_SL[-1,:])
			print('Reflection [%d,%d,%d]'%(HKL[0],HKL[1],HKL[2]))
			XY_cen_pred_arry=CCB_pat_sim.in_plane_cor(0,1e8,0.10/cam_len,90,K_in_cen_pred.reshape(-1,3),K_out_cen_pred.reshape(-1,3))
			PXY_cen_pred_arry=CCB_pat_sim.XY2P(XY_cen_pred_arry,75.0e-6,1594+k_out_osx*0.1/cam_len/(75e-6),1764+k_out_osy*0.1/cam_len/(75e-6))
			kerror_arry=Pxy_cen_arry[ind,:]-PXY_cen_pred_arry
			if np.linalg.norm(kerror_arry.reshape(-1,))<=50:	
				num_pix=props[s_ind].coords.shape[0]
				K_pix_arry=np.zeros((num_pix,13))	
				K_pix_arry[:,0]=int(frame)
				K_pix_arry[:,1]=props[s_ind].coords[:,1] # x coordinate of the pixel
				K_pix_arry[:,2]=props[s_ind].coords[:,0] # y coordinate of the pixel
				K_pix_arry[:,3]=exp_img[K_pix_arry[:,2].astype(np.int),K_pix_arry[:,1].astype(np.int)] # intensity of the pixel
			
				K_pix_arry_cor=K_pix_arry[:,1:3]-kerror_arry
				K_pix_arry[:,4:7]=HKL_int[ind,:]
				x_pix=(K_pix_arry_cor[:,0]-(1594+k_out_osx*0.1/cam_len/(75e-6)))*75e-6
				y_pix=(K_pix_arry_cor[:,1]-(1764+k_out_osy*0.1/cam_len/(75e-6)))*75e-6
				z_pix=np.ones((num_pix,))*0.10/cam_len#instead of cam_len
				k_pix_cen_dir=np.hstack((x_pix.reshape(-1,1),y_pix.reshape(-1,1),z_pix.reshape(-1,1)))
				k_pix_cen_dir=k_pix_cen_dir/np.linalg.norm(k_pix_cen_dir,axis=-1).reshape(-1,1)
				K_pix_arry[:,7:10]=(1/wave_len)*k_pix_cen_dir
				OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat
				Q=OR@(HKL_int[ind,:].reshape(3,1))
				#print(Q)
				K_pix_arry[:,10:13]=K_pix_arry[:,7:10]-Q.reshape(1,3)
	######	col0:frame, col1~3: x,y,I, col4~6:HKL, col7~9:kout, col10~12:k_in
				K_pix_arry_all=np.vstack((K_pix_arry_all,K_pix_arry))
			else:
				HKL_counter=HKL_counter+1
				print('The [%d,%d,%d] reflection is not matching closely'%(HKL[0],HKL[1],HKL[2]))
		else:
			HKL_counter=HKL_counter+1
			print('The [%d,%d,%d] reflection is not excited by the model'%(HKL[0],HKL[1],HKL[2]))
			K_in_cen_pred=np.array([0,0,0])
			K_out_cen_pred=np.array([0,0,0])
	print('%d out  of %d reflections not included.'%(HKL_counter,HKL_int.shape[0]))
	return K_pix_arry_all, HKL_int, Pxy_cen_arry, OR, props

def K_output_frame(K_pix_arry_all):

	#frame=int(K_pix_arry_all[0,0])
	
	#f.open('K_map_fr%d.txt'%(frame),'w')
	np.savetxt('K_map_fr%d.txt'%(frame),K_pix_arry_all,fmt=['%3d','%7.1f','%7.1f','%7.1f','%3d','%3d','%3d','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e'])
	return None

if __name__=='__main__':

	exp_img_file=os.path.abspath(sys.argv[1])
	res_file=os.path.abspath(sys.argv[2])
	start_frame=int(sys.argv[3])
	end_frame=int(sys.argv[4])
	thld=int(sys.argv[5])
	min_pix=int(sys.argv[6])
	for frame in range(start_frame,end_frame+1):
		K_pix_arry_all, HKL_int, Pxy_cen_arry, OR, props=get_K_frame(exp_img_file,frame,res_file='/home/lichufen/CCB_ind/Best_GA_res.txt',thld=thld,min_pix=min_pix)
		K_output_frame(K_pix_arry_all)
		print('frame %d done'%(frame))

