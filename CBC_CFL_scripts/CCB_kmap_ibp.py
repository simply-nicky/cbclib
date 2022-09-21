'''
CCB_kmap_ibp.py consists of the functions to evaluate the
K_in and K_out wave-vectors for each pixel of the
each diffraction streak predicted.

the 'ibp' version is for the method of "integration by prediction", in contrast
to other methods such as "integration by detection"
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
# matplotlib.use('TkAgg') # To be adjusted for teh batch job mode.
import matplotlib.pyplot as plt

#OR_mat=np.array([[-2.05112078e+08,-3.89499652e+08,-1.22739594e+08],
#[-2.21095123e+08,1.95462074e+08,-3.25490173e+08],
#[ 5.57086534e+08,-8.72801443e+07,-1.74338424e+08]])
OR_mat = np.genfromtxt('/gpfs/cfel/user/lichufen/CBDXT/P11_BT_2022/OR.txt')
OR_mat=OR_mat/1.0

###################
# for expanding lattice constants
expanding_const = 1
OR_mat = OR_mat/expanding_const
##################


E_ph=17.5 #in keV
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m

k0 = 1/wave_len
k_cen = np.hstack((-3.04e8*np.ones((1500,1)),2.43e8*np.ones((1500,1)),np.sqrt(k0**2-(3.04e8)**2-(2.43e8)**2)*np.ones((1500,1))))
k_cen = k_cen/(np.linalg.norm(k_cen,axis=1).reshape(-1,1))*1/wave_len
pix_size = 75e-6

def integration_mask_maker(end_point1,end_point2,v_vec,n_vec,img_dim=(4362,4148),r_n=10,R_n=15,r_v_add=5,R_v_add=10):
    Integration_mask = np.zeros(img_dim,dtype=np.int8)
    streak_rad = int(np.ceil(np.linalg.norm(end_point2-end_point1)/2))
    centroid = np.round((end_point1+end_point2)/2)
    centroid = centroid.astype(np.int32)
    r_v = r_v_add + streak_rad
    R_v = R_v_add + streak_rad
    for k in range(centroid[0]-streak_rad-50,centroid[0]+streak_rad+50):
        for l in range(centroid[1]-streak_rad-50,centroid[1]+streak_rad+50):
            p_v = np.abs(np.dot((np.array([k,l,0]) - np.array([centroid[0],centroid[1],0])),v_vec))
            p_n = np.abs(np.dot((np.array([k,l,0]) - np.array([centroid[0],centroid[1],0])),n_vec))
            Cond1 = (p_v<=r_v)*(p_n<=r_n)
            Cond2 = (p_v<=R_v)*(p_n<=R_n)
            if Cond1==True:
                Integration_mask[l,k] = 1
            elif Cond1==False:
                if Cond2==True:
                    Integration_mask[l,k] = 2
                elif Cond2==False:
                    Integration_mask[l,k] = 0
    return Integration_mask,centroid


def get_K_frame(exp_img_file,mask_file,frame,OR,geom_dict,r_n=10,R_n=15,r_v_add=5,R_v_add=10):
	'''
	get_k_frame,for each frame,
	 returns the k_in, k_out, HkL_in along with other info from the streak detection.
	'''

	#####################################

	# print('Read streak detection info from the file:\n %s'%(frame))
	print('Extracting signals:\n %s frame %s '%(exp_img_file,frame))

	with h5py.File(mask_file,'r') as m:
	    mask_arry = np.array(m['/data/data'])
	    bkg = np.array(m['/data/bkg'])  # the background roughly estimated through frames.

	with h5py.File(exp_img_file,'r') as dsum:
	    img_arry=np.array(dsum['entry/data/data'][frame:frame+1,:,:])
	    print(dsum['entry/data/data'].shape)
	    img = img_arry.mean(axis=0)
	img1 = img - bkg

	# beam_cx = 2077
	# beam_cy = 2208
	beam_cx = geom_dict['beam_cx']
	beam_cy = geom_dict['beam_cy']

	# cam_len = 1.022
	# k_out_osx = 1.93972178e-03
	# k_out_osy = 9.16594929e-04
	det_dist = geom_dict['det_dist']
	cam_len = geom_dict['cam_len']
	k_out_osx = geom_dict['k_out_osx']
	k_out_osy = geom_dict['k_out_osy']
	### geometry not yet optimized and updated.
	print('prediction from forward model')
	HKL_table, K_in_table, K_out_table = CCB_pat_sim.pat_sim_q(k_cen[frame,:],OR,3)
	print('pat_sim done')
	#####
	XY0=CCB_pat_sim.in_plane_cor(0,1e8,det_dist/cam_len,0,K_in_table,K_out_table)
	PXY0=CCB_pat_sim.XY2P(XY0,75.0e-6,beam_cx-k_out_osx*det_dist/cam_len/(75e-6),beam_cy+k_out_osy*det_dist/cam_len/(75e-6))
	PXY0_int = np.round(PXY0,decimals=0).astype(np.int32)
	PXY0_int = np.unique(PXY0_int,axis=0)
	num_ref = HKL_table.shape[0]
	num_st = num_ref
	end_point1 = np.zeros((num_st,2))
	end_point2= np.zeros((num_st,2))
	centroid_arry = np.zeros((num_st,2))
	orientation_arry = np.zeros((num_st,))
	v_vector = np.zeros((num_st,3))
	n_vector = np.zeros((num_st,3))
	for n in range(num_st):
	    if n==0:
	        ind1 = 0
	        ind2 = int(HKL_table[0,-1])
	    else:
	        ind1 = int(HKL_table[0:n,-1].sum())
	        ind2 = int(HKL_table[0:n+1,-1].sum())
	    hkl = HKL_table[n,0:3]
	    cen_x = PXY0[:,0][ind1:ind2]
	    cen_y = PXY0[:,1][ind1:ind2]
	    num_pix = int(ind2-ind1)
	    centroid_arry[n,:] = np.array([cen_y.mean(),cen_x.mean()])
	    #### computing the orientation for each streak
	    x_min = PXY0[:,0][ind1:ind2].min()
	    x_max = PXY0[:,0][ind1:ind2].max()
	    y_min = PXY0[:,1][ind1:ind2].min()
	    y_max = PXY0[:,1][ind1:ind2].max()
	    delta_x = np.abs(x_max - x_min)
	    delta_y = np.abs(y_max - y_min)
	    if delta_x>=delta_y:
	        ind11 = np.where(PXY0[:,0][ind1:ind2]==x_min)[0]
	        ind22 = np.where(PXY0[:,0][ind1:ind2]==x_max)[0]
	        y1 = PXY0[:,1][ind1:ind2][ind11].mean()
	        y2 = PXY0[:,1][ind1:ind2][ind22].mean()
	        orientation_arry[n] = (x_min-x_max)/(y1-y2+sys.float_info.epsilon)
	    else:
	        ind11 = np.where(PXY0[:,1][ind1:ind2]==y_min)[0]
	        ind22 = np.where(PXY0[:,1][ind1:ind2]==y_max)[0]
	        x1 = PXY0[:,0][ind1:ind2][ind11].mean()
	        x2 = PXY0[:,0][ind1:ind2][ind22].mean()
	        orientation_arry[n] = (x1-x2)/(y_min-y_max+sys.float_info.epsilon)

	    if orientation_arry[n]>=0:
	        end_point1[n,:] = np.array([x_min,y_min])
	        end_point2[n,:] = np.array([x_max,y_max])
	    else:
	        end_point1[n,:] = np.array([x_min,y_max])
	        end_point2[n,:] = np.array([x_max,y_min])

	    v_vector[n,:] = np.array([end_point2[n,0]-end_point1[n,0],end_point2[n,1]-end_point1[n,1],0])
	#     print(v_vector[n,:])
	    v_vector[n,:] = v_vector[n,:]/np.linalg.norm(v_vector[n,:])
	    n_vector[n,:] = np.cross(np.array([0,0,1]),v_vector[n,:])
	ind_notnan = np.logical_not(np.isnan(v_vector)).all(axis=1).nonzero()[0]
	end_point1=end_point1[ind_notnan]
	end_point2=end_point2[ind_notnan]
	v_vector = v_vector[ind_notnan]
	n_vector = n_vector[ind_notnan]
	HKL_table=HKL_table[ind_notnan]
	num_st = ind_notnan.shape[0]
	print(f'{num_st:d} reflections to be integrated.')
	############################

	num_streak = num_st
	K_pix_arry_all=np.array([]).reshape(-1,16)
	K_streak_arry_all = np.array([]).reshape(-1,16)
	for ind in range(num_streak):
		Int_mask,centroid = integration_mask_maker(end_point1[ind],end_point2[ind],v_vector[ind],n_vector[ind],\
                                               img_dim=(4362,4148),r_n=r_n,R_n=R_n,r_v_add=r_v_add,R_v_add=R_v_add)

		bkg_pix_no = ((Int_mask==2)*mask_arry).sum()
		bkg_sum = ((Int_mask==2)*mask_arry*(img1)).sum()
		bkg_streak_pix = bkg_sum/bkg_pix_no  #local background per pixel of a streak.

		sig_pix_no = ((Int_mask==1)*mask_arry).sum()
		print(f'sig_pix_no:{sig_pix_no}, bkg_pix_no:{bkg_pix_no}')
		if (sig_pix_no)*(bkg_pix_no)==0:
			print('skip empty region (e.g.detector gaps)')
			continue
		sig_sum = ((Int_mask==1)*mask_arry*(img1)).sum()
		sig_sum_clean = sig_sum - bkg_streak_pix*sig_pix_no

		num_pix=sig_pix_no
		print(f'{ind}th reflection, {num_pix} pixels')
		maj_len = np.linalg.norm(end_point2[ind]-end_point1[ind])
		mino_len = r_n*2
		K_pix_arry=np.zeros((num_pix,16))
		K_pix_arry[:,0]=int(frame)
		ind_pix = ((Int_mask==1)*mask_arry).nonzero()
		K_pix_arry[:,1]=ind_pix[1] # x coordinate of the pixel
		K_pix_arry[:,2]=ind_pix[0] # y coordinate of the pixel
		K_pix_arry[:,3] = ((Int_mask==1)*mask_arry*(img1))[ind_pix]
		K_pix_arry[:,4:7]=HKL_table[ind,0:3]
		x_pix=(K_pix_arry[:,1]-(beam_cx-k_out_osx*det_dist/cam_len/(75e-6)))*75e-6
		y_pix=(K_pix_arry[:,2]-(beam_cy+k_out_osy*det_dist/cam_len/(75e-6)))*75e-6
		z_pix=np.ones((num_pix,))*det_dist/cam_len
		k_pix_cen_dir=np.hstack((-x_pix.reshape(-1,1),y_pix.reshape(-1,1),z_pix.reshape(-1,1)))
		k_pix_cen_dir=k_pix_cen_dir/np.linalg.norm(k_pix_cen_dir,axis=-1).reshape(-1,1)
		K_pix_arry[:,7:10]=(1/wave_len)*k_pix_cen_dir

		Q=OR@(HKL_table[ind,0:3].reshape(3,1))
		#print(Q)
		K_pix_arry[:,10:13]=K_pix_arry[:,7:10]-Q.reshape(1,3)
		K_pix_arry[:,13] = maj_len
		K_pix_arry[:,14] = mino_len
		K_pix_arry[:,15] = bkg_streak_pix

		K_streak_arry = np.zeros((1,16))
		K_streak_arry[:,0] = int(frame)
		K_streak_arry[:,1] = centroid[0]
		K_streak_arry[:,2] = centroid[1]
		K_streak_arry[:,3] = sig_sum_clean
		K_streak_arry[:,4:7] = HKL_table[ind,0:3]
		x_pix_s=(K_streak_arry[:,1]-(beam_cx-k_out_osx*det_dist/cam_len/(75e-6)))*75e-6
		y_pix_s=(K_streak_arry[:,2]-(beam_cy+k_out_osy*det_dist/cam_len/(75e-6)))*75e-6
		z_pix_s=det_dist/cam_len
		k_pix_cen_dir_s=np.hstack((np.array([-x_pix_s]).reshape(-1,1),np.array([y_pix_s]).reshape(-1,1),np.array([z_pix_s]).reshape(-1,1)))
		k_pix_cen_dir_s=k_pix_cen_dir_s/np.linalg.norm(k_pix_cen_dir_s,axis=-1).reshape(-1,1)
		K_streak_arry[:,7:10]=(1/wave_len)*k_pix_cen_dir_s
		Q=OR@(HKL_table[ind,0:3].reshape(3,1))
		K_streak_arry[:,10:13]=K_streak_arry[:,7:10]-Q.reshape(1,3)
		K_streak_arry[:,13] = maj_len
		K_streak_arry[:,14] = mino_len
		K_streak_arry[:,15] = bkg_streak_pix

	######	col0:frame, col1~3: x,y,I, col4~6:HKL, col7~9:kout, col10~12:k_in, col13:major_axis_length, col14:minor_axis_length, col15:bkg
		kin_val = np.array([CCB_pat_sim.pupil_func(k_in) for k_in in K_pix_arry[:,10:13]])
		##### rejection module for the streak according to K_in and the pupil range.
		# if (kin_val.sum()/kin_val.shape[0])<0.3:
		# 	print(kin_val.sum()/kin_val.shape[0])
		# 	print('outlier_2',K_pix_arry[0,4:7])
		# 	#print(kin_val)
		# 	continue
		#########
		K_pix_arry_all=np.vstack((K_pix_arry_all,K_pix_arry))
		K_streak_arry_all = np.vstack((K_streak_arry_all,K_streak_arry))
	print('K_pix_arry_all,K_streak_arry_all done')
	return K_pix_arry_all, K_streak_arry_all


def HKL_patch():
	pass
	return K_in, HKL



def K_output_frame(exp_img_file,K_pix_arry_all,K_streak_arry_all):
    short = os.path.basename(exp_img_file)
    scan_id = int(short.split('.')[0].split('_')[1])
    file_id = int(short.split('.')[0].split('_')[3])
    frame = int(K_pix_arry_all[0,0])
    np.savetxt(f'K_map_scan{scan_id:d}_file{file_id:d}_fr{frame:d}.txt',K_pix_arry_all,fmt=['%3d','%7.1f','%7.1f','%7.3f','%3d','%3d','%3d','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%7.1f','%7.1f','%7.3f'])
    np.savetxt(f'K_streak_scan{scan_id:d}_file{file_id:d}_fr{frame:d}.txt',K_streak_arry_all,fmt=['%3d','%7.1f','%7.1f','%7.3f','%3d','%3d','%3d','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%7.1f','%7.1f','%7.3f'])
    return None

if __name__=='__main__':
    exp_img_file=os.path.abspath(sys.argv[1])
    mask_file=os.path.abspath(sys.argv[2])
    start_frame=int(sys.argv[3])
    end_frame=int(sys.argv[4])
    r_n = int(sys.argv[5])
    R_n = int(sys.argv[6])
    r_v_add = int(sys.argv[7])
    R_v_add = int(sys.argv[8])
    angle_position = np.array([-170,-135,-90,-45,0,10])
    k_out_osx0 = 1.93972178e-03
    k_out_osy0 = 9.16594929e-04
    k_out_osx_arry_s = np.array([1,2,-0.1,-1,0,1])*1e-3+k_out_osx0
    k_out_osy_arry_s = np.array([-6,-6.5,-5,-3,0,0])*1e-3+k_out_osy0

    def test_func1(x,a):
        phi=-45
        b = k_out_osx0 - a*np.cos(-2*np.pi*phi/180)
        y = a*np.cos(2*np.pi*(x-phi)/180)+b
        return y

    def test_func2(x,a):
        phi=-45
        b = k_out_osy0 - a*np.sin(-2*np.pi*phi/360)
        y = a*np.sin(2*np.pi*(x-phi)/360)+b
        return y

    params1, params_covariance1 = scipy.optimize.curve_fit(test_func1, angle_position, k_out_osx_arry_s,p0=[1])
    params2, params_covariance2 = scipy.optimize.curve_fit(test_func2, angle_position, k_out_osy_arry_s,p0=[1])
    for frame in range(start_frame,end_frame+1,10):
        short = os.path.basename(exp_img_file)
        scan_id = int(short.split('.')[0].split('_')[1])
        file_id = int(short.split('.')[0].split('_')[3])
        if scan_id==207:
            angle_val = ((file_id-10)*1000+(frame-0))/100 # currently, the OR.txt if for scan 207, file 10, frame 0.
        elif scan_id==206:
            angle_val = (-9500 + (scan_id-8)*1000+(frame-0))/100
        else:
            sys.exit('Check the exp_img_file name')
        print(f'angle_val:{angle_val:f}')
        k_out_osx = test_func1(angle_val-5,params1[0])
        k_out_osy = test_func2(angle_val-5,params2[0])

        geom_dict=dict()
        geom_dict['beam_cx'] = 2077
        geom_dict['beam_cy'] = 2208
        geom_dict['det_dist']= 0.4668
        geom_dict['cam_len'] = 1.022
        # geom_dict['k_out_osx'] = 1.93972178e-03
        geom_dict['k_out_osx'] = k_out_osx
        # geom_dict['k_out_osy'] = 9.16594929e-04
        geom_dict['k_out_osy'] = k_out_osy
        OR=CCB_ref.rot_mat_yaxis(angle_val)@OR_mat
        K_pix_arry_all,K_streak_arry_all = get_K_frame(exp_img_file,mask_file,frame,OR,geom_dict,r_n=r_n,R_n=R_n,r_v_add=r_v_add,R_v_add=R_v_add)
        K_output_frame(exp_img_file,K_pix_arry_all,K_streak_arry_all)
        print('frame %d done'%(frame))
