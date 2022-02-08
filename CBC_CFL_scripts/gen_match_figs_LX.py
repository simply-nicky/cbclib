'''
'gen_match_figs_LX.py' is to generate the "matching" figures for individual diffraction frames
for the CBC data sets.

(intially took from the 'CCB_FFT-Copy1.ipynb' file)
'''

import sys,os
import numpy as np
import CCB_ref
import CCB_pred
import CCB_pat_sim
import CCB_read
import h5py
import re
import matplotlib
matplotlib.use('pdf')
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


def read_frame(exp_img_file,frame):

    #exp_img_file='/Users/lichufen/Nextcloud/CCB_B12/scan_corrected_00135.h5'
    #exp_img_file='/Users/chufeng/Downloads/scan_corrected_00135.h5'
    f=h5py.File(exp_img_file,'r')
    f['/corrected_data/corrected_data'].shape
   #frame=1
    exp_img=np.array(f['/corrected_data/corrected_data'][frame,:,:])
    f.close()
    # plt.figure(figsize=(10,10))
    # plt.imshow(exp_img)
    # #plt.axis('equal')
    # plt.xlim(250,2100)
    # plt.ylim(500,2300)
    # plt.clim(0,50)
    return exp_img

def get_Ks(frame,OR_angs):
    theta,phi,alpha=OR_angs
    OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat
    res_cut=1.2
    HKL_table, K_in_table, K_out_table=CCB_pat_sim.pat_sim_q(OR,res_cut)
    K_in_pred_s,K_out_pred_s=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_table[:,0:3])
    # plt.figure(figsize=(10,10))
    # plt.scatter(K_out_table[:,0],K_out_table[:,1],s=1,marker='x',c='g')
    # plt.scatter(K_out[:,0],K_out[:,1],s=20,marker='x',color='b')
    # plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=20,marker='x',color='r')
    # plt.scatter(K_out_pred_s[:,0],K_out_pred_s[:,1],s=40,marker='o',edgecolor='black',facecolor='None')
    # plt.axis('equal')
    # plt.figure(figsize=(10,10))
    # plt.scatter(K_in_table[:,0],K_in_table[:,1],s=4,marker='o',c='g')
    # plt.scatter(K_in_pred_s[:,0],K_in_pred_s[:,1],s=4,marker='o',c='black')
    # plt.scatter(K_in_pred[:,0],K_in_pred[:,1],s=4,marker='o',c='r')
    # #print(HKL_table[:30,:],HKL_int)
    # plt.figure(figsize=(10,10))
    # plt.scatter(Delta_k_in_new[:,0],Delta_k_in_new[:,1],s=10,marker='o',c=np.linalg.norm(Delta_k_out_new,axis=1),cmap='jet')
    # plt.colorbar()
    return HKL_table, K_in_table, K_out_table, K_in_pred_s,K_out_pred_s

def read_res(res_file):
    res_file=os.path.abspath(res_file)
    f=open(res_file,'r')
    lines=f.readlines()
    f.close()
    counter=0
    frame_ind_list=[]
    for ind,l in enumerate(lines):
        if l.startswith('frame'):
            frame_ind_list.append(ind)
            counter=counter+1
    #print('%d frames found from %s'%(counter,res_file))
    res_arry=np.zeros((counter,9))
    for  m,ind in enumerate(frame_ind_list):
        frame=int(re.split(' ',lines[ind])[1])
        initial_TG=float(re.split(' ',lines[ind+1])[2])
        final_TG=float(re.split(' ',lines[ind+2])[2])
        res_par=[float(m) for m in re.split('[ \n]',lines[ind+4])[:-1]]
        res_arry[m,0]=frame
        res_arry[m,1:-2]=res_par
        res_arry[m,-2]=initial_TG
        res_arry[m,-1]=final_TG
    return res_arry


def gen_single_match(exp_img_file,res_file,ind1):
    res_arry=read_res(res_file)
    frame=int(res_arry[ind1,0])
    OR_angs=tuple(res_arry[ind1,1:4])
    cam_len=res_arry[ind1,4]
    k_out_osx=res_arry[ind1,5]
    k_out_osy=res_arry[ind1,6]
    exp_img=read_frame(exp_img_file,frame)

    HKL_table, K_in_table, K_out_table, K_in_pred_s,K_out_pred_s = get_Ks(frame,OR_angs)
    XY0=CCB_pat_sim.in_plane_cor(0,1e8,0.10/cam_len,11,K_in_table,K_out_table)
    XY1=CCB_pat_sim.in_plane_cor(1.0e-3,3e8,0.10/cam_len,68,K_in_table,K_out_table)
    XY2=CCB_pat_sim.off_plane_cor(1e-3,2e8,0.1,11,K_in_table,K_out_table)


    PXY0=CCB_pat_sim.XY2P(XY0,75.0e-6,1540,1724.4)
    PXY1=CCB_pat_sim.XY2P(XY1,75e-6,1540+k_out_osx*0.1/cam_len/(75e-6),1724.4+k_out_osy*0.1/cam_len/(75e-6))
    PXY2=CCB_pat_sim.XY2P(XY2,75e-6,1540,1724.4)
    # ################################
    # The above does not use the k_out_osx,k_out_osy,cam_len.
    #
    #
    # ###############################

	#################################
	# save the K_in and K_out arrys in .txt file
    out_txt_file='K_map_sim_LX_fr%d.txt'%(frame)
    np.savetxt(out_txt_file,np.hstack((K_in_table,K_out_table,PXY1)),fmt='%13.3e %13.3e %13.3e %13.3e %13.3e %13.3e %7.2f %7.2f')
	#################################
    plt.figure(figsize=(10,10))
    plt.imshow(exp_img)
    #plt.axis('equal')
    plt.xlim(250,2100)
    plt.ylim(500,2300)
    plt.clim(0,50)
    #plt.scatter(PXY0[:,0],PXY0[:,1],s=1,marker='x',c='g')
    plt.scatter(PXY1[:,0],PXY1[:,1],s=0.2,marker='x',c='b')
    #plt.scatter(PXY2[:,0],PXY2[:,1],s=1,marker='x',c='r')
    plt.savefig('match_LX'+'fr'+str(int(frame))+'.png')
    #print('res_file: %s'%(res_file))
    print('frame %d done!\n'%(frame))
    return None

if __name__=='__main__':
    exp_img_file=os.path.abspath(sys.argv[1])
    res_file=os.path.abspath(sys.argv[2])
    res_arry=read_res(res_file)
    print('res_file: %s'%(res_file))
    print('%d frames loaded in the res_file'%(res_arry.shape[0]))
    for ind1 in range(res_arry.shape[0]):
        gen_single_match(exp_img_file,res_file,ind1)
    print('ALL DONE!!!')
