'''
CCB_kerror1.py computes the error (displacement) vector
for each streaks based on the observed and predicted positions of
the streak centroids.
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
import CCB_kmap1
import matplotlib
matplotlib.use('TkAgg') # To be adjusted for teh batch job mode.
import matplotlib.pyplot as plt

OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
[0.00000000e+00,0.00000000e+00,4.00000000e+08]])
OR_mat=OR_mat/1.03

E_ph=17 #in keV
#wave_len=12.40/E_ph #in Angstrom
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
k_cen=np.array([0,0,1/wave_len]).reshape(3,1)

def get_kerror_frame(exp_img_file,frame,res_file='/home/lichufen/CCB_ind/Best_GA_res.txt',thld=10,min_pix=20):
	
	K_pix_arry_all, HKL_int, Pxy_cen_arry, OR, props=CCB_kmap1.get_K_frame(exp_img_file,frame,res_file=res_file,thld=thld,min_pix=min_pix)


	res_arry=gm.read_res(res_file)
	ind=np.where(res_arry[:,0]==frame)[0][0]
	frame=res_arry[ind,0]
	theta=res_arry[ind,1]
	phi=res_arry[ind,2]
	alpha=res_arry[ind,3]
	cam_len=res_arry[ind,4]
	k_out_osx=res_arry[ind,5]
	k_out_osy=res_arry[ind,6]


	
	num_s=HKL_int.shape[0]
	K_in_cen_pred_arry=np.zeros((num_s,3))
	K_out_cen_pred_arry=np.zeros((num_s,3))
	for m in range(num_s):
		HKL=HKL_int[m,:]
		#print(HKL)
		K_in_SL, K_out_SL=CCB_pat_sim.source_line_scan(k_cen,OR,HKL,rot_ang_step=0.05,rot_ang_range=2.0)
		if K_in_SL.shape[0]!=0:
			K_in_cen_pred=0.5*(K_in_SL[0,:]+K_in_SL[-1,:])
			K_out_cen_pred=0.5*(K_out_SL[0,:]+K_out_SL[-1,:])
			K_in_cen_pred_arry[m,:]=K_in_cen_pred
			K_out_cen_pred_arry[m,:]=K_out_cen_pred
			print('Reflection [%d,%d,%d]'%(HKL[0],HKL[1],HKL[2]))
		else:
			print('The [%d,%d,%d] reflection is not excited by the model'%(HKL[0],HKL[1],HKL[2]))
			K_in_cen_pred=np.array([0,0,0])
			K_out_cen_pred=np.array([0,0,0])
			K_in_cen_pred_arry[m,:]=K_in_cen_pred
			K_out_cen_pred_arry[m,:]=K_out_cen_pred	
	XY_cen_pred_arry=CCB_pat_sim.in_plane_cor(0,1e8,0.10/cam_len,90,K_in_cen_pred_arry,K_out_cen_pred_arry)
	PXY_cen_pred_arry=CCB_pat_sim.XY2P(XY_cen_pred_arry,75.0e-6,(1540+k_out_osx*0.1/cam_len/(75e-6)),(1724.4+k_out_osy*0.1/cam_len/(75e-6)))
	kerror_arry=Pxy_cen_arry-PXY_cen_pred_arry
	ind_filter=(K_out_cen_pred_arry[:,0]==0)*(K_out_cen_pred_arry[:,1]==0)*(K_out_cen_pred_arry[:,2]==0)
	ind_filter=np.logical_not(ind_filter)
	#K_pix_arry_all=K_pix_arry_all[ind_filter,:]
	HKL_int=HKL_int[ind_filter,:]
	Pxy_cen_arry=Pxy_cen_arry[ind_filter,:]
	K_in_cen_pred_arry=K_in_cen_pred_arry[ind_filter,:]
	K_out_cen_pred_arry=K_out_cen_pred_arry[ind_filter,:]
	PXY_cen_pred_arry=PXY_cen_pred_arry[ind_filter,:]
	kerror_arry=kerror_arry[ind_filter,:]
	num_s2=HKL_int.shape[0]
	print('%d out of %d are predicted not excited by the model\n'%(num_s-num_s2,num_s))
	return HKL_int, Pxy_cen_arry, K_in_cen_pred_arry,PXY_cen_pred_arry, kerror_arry

def kerror_output_frame(frame,HKL_int, Pxy_cen_arry, PXY_cen_pred_arry,kerror_arry):
	ALL_arry=np.hstack((np.ones((HKL_int.shape[0],1))*frame,HKL_int,Pxy_cen_arry,K_in_cen_pred_arry,PXY_cen_pred_arry,kerror_arry))
	np.savetxt('K_error_fr%d.txt'%(frame),ALL_arry,fmt=['%3d','%3d','%3d','%3d','%7.1f','%7.1f','%13.3e','%13.3e','%13.3e','%7.1f','%7.1f','%7.3f','%7.3f'])

	return None

if __name__=='__main__':
	
	exp_img_file=os.path.abspath(sys.argv[1])
	res_file=os.path.abspath(sys.argv[2])
	start_frame=int(sys.argv[3])
	end_frame=int(sys.argv[4])
	thld=int(sys.argv[5])
	min_pix=int(sys.argv[6])
	for frame in range(start_frame,end_frame+1):
		HKL_int, Pxy_cen_arry, K_in_cen_pred_arry,PXY_cen_pred_arry, kerror_arry=get_kerror_frame(exp_img_file,frame,res_file=res_file,thld=thld,min_pix=min_pix)
		kerror_output_frame(frame, HKL_int, Pxy_cen_arry, PXY_cen_pred_arry,kerror_arry)
		print('frame %d done'%(frame))
