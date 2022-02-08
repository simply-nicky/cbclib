'''
'gen_match_figs.py' is to generate the "matching" figures for individual diffraction frames
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
#matplotlib.use('pdf')
import matplotlib.pyplot as plt

#OR_mat=np.array([[-2.05112078e+08,-3.89499652e+08,-1.22739594e+08],
#[-2.21095123e+08,1.95462074e+08,-3.25490173e+08],
#[ 5.57086534e+08,-8.72801443e+07,-1.74338424e+08]])
OR_mat = np.genfromtxt('../../OR.txt')
OR_mat=OR_mat/1.0

#rot_mat0 = np.array([[0.97871449,-0.20522657,0],\
#[0.20522657,0.97871449,0],\
#[0,0,1]])
rot_mat0=np.array([[1,0,0],[0,1,0],[0,0,1]])

###################
# for expanding lattice constants
expanding_const = 1
OR_mat = OR_mat/expanding_const
##################


#########################
# for lysozyme 
#OR_mat0 = 1e10*np.array([[1/79.14,0,0],\
#[0,1/79.14,0],\
#[0,0,1/38.02]])

#OR_mat = rot_mat0@OR_mat0

#########################

E_ph=17.4 #in keV
#wave_len=12.40/E_ph #in Angstrom
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
k0 = 1/wave_len
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=1/wave_len*np.array([-0.03115,-0.02308,0.999248]).reshape(3,1)
#k_cen = np.genfromtxt('/home/lichufen/CCB_ind/k_cen.txt')
k0 = 1/wave_len
k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),np.sqrt(k0**2-(3e8)**2-(2.2e8)**2)*np.ones((1500,1))))
k_cen = k_cen/(np.linalg.norm(k_cen,axis=1).reshape(-1,1))*1/wave_len


def read_frame(exp_img_file,frame,h5path='/corrected_data/corrected_data'):

    #exp_img_file='/Users/lichufen/Nextcloud/CCB_B12/scan_corrected_00135.h5'
    #exp_img_file='/Users/chufeng/Downloads/scan_corrected_00135.h5'
    if 'sim_data' in exp_img_file:
        h5path = 'data/simulated_data'
    else:
        #h5path = 'corrected_data/corrected_data'
        h5path = 'entry/data/data'
    print('before')
    with h5py.File(exp_img_file,'r') as f:
    #f[h5path].shape
    #frame=1
        print('after')
        exp_img=np.array(f[h5path][frame,:,:])
    # plt.figure(figsize=(10,10))
    # plt.imshow(exp_img)
    # #plt.axis('equal')
    # plt.xlim(250,2100)
    # plt.ylim(500,2300)
    # plt.clim(0,50)
    return exp_img

def get_Ks(frame,OR_angs):
    theta,phi,alpha=OR_angs
    OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(0.5*frame)@OR_mat
    res_cut=1.2*expanding_const #adapted for artifical lattice testing.
    HKL_table, K_in_table, K_out_table=CCB_pat_sim.pat_sim_q(k_cen[frame,:],OR,res_cut)
    K_in_pred_s,K_out_pred_s=CCB_pat_sim.kout_pred(OR,k_cen[frame,:],HKL_table[:,0:3])
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


def gen_single_match(exp_img_file,res_file,ind1,save_fig=True,save_K_sim_txt=True):
    res_arry=read_res(res_file)
    frame=int(res_arry[ind1,0])
    OR_angs=tuple(res_arry[ind1,1:4])
    cam_len=res_arry[ind1,4]
    k_out_osx=res_arry[ind1,5]
    k_out_osy=res_arry[ind1,6]

    if 'sim_data' in exp_img_file:
        exp_img=read_frame(exp_img_file,frame,h5path='data/simulated_data')
    else:
        exp_img=read_frame(exp_img_file,frame)
        with h5py.File('/gpfs/cfel/user/lichufen/CBDXT/P11_BT/CFL_mask_scan_210.h5','r') as m:
            mask = np.array(m['/data/data']).astype(bool)
            bkg = np.array(m['/data/bkg'])
        exp_img = exp_img - bkg
        exp_img = exp_img*mask

    HKL_table, K_in_table, K_out_table, K_in_pred_s,K_out_pred_s = get_Ks(frame,OR_angs)
    XY0=CCB_pat_sim.in_plane_cor(0,1e8,0.2/cam_len,11,K_in_table,K_out_table)
    XY1=CCB_pat_sim.in_plane_cor(1e-3,2e8,0.1,11,K_in_table,K_out_table)
    XY2=CCB_pat_sim.off_plane_cor(1e-3,2e8,0.1,11,K_in_table,K_out_table)


    PXY0=CCB_pat_sim.XY2P(XY0,75.0e-6,1908-k_out_osx*0.2/cam_len/(75e-6),2207+k_out_osy*0.2/cam_len/(75e-6))
    PXY1=CCB_pat_sim.XY2P(XY1,73.5e-6,1535,1723)
    PXY2=CCB_pat_sim.XY2P(XY2,73.5e-6,1535,1723)
    # ################################
    # The above does not use the k_out_osx,k_out_osy,cam_len.
    #
    #
    # ###############################

	#################################
	# save the K_in and K_out arrys in .txt file
    HKL_table1 = np.repeat(HKL_table[:,0:3],HKL_table[:,3].astype(np.int),axis=0)
	##################################
	# Compute the simulated intensity Int_sim
#    theta,phi,alpha=OR_angs
#    rot_mat = CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(0.5*frame)@rot_mat0
#    with h5py.File('/home/lichufen/CCB_ind/scan_corrected_00135.h5','r') as f:
#        ref_image = np.array(f['/data/data'][frame,:,:]) 
#    xyz_range = [-1.1e-3,-0.4e-3,-1.1e-3,-0.4e-3,-0.10275,-0.10225]
#    pivot_coor = [-0.75e-3,-0.75e-3,-0.1025]
#    xtal_model0_dict = CCB_pat_sim.xtal_model_init(xyz_range,voxel_size=10e-6)
#    xtal_model_dict = CCB_pat_sim.k_in_render(xtal_model0_dict,rot_mat,pivot_coor,focus_coor=[0,0,-0.129])

#    Int_ref_arry = CCB_pat_sim.get_Int_ref('/home/lichufen/CCB_ind/ethc_mk.pdb.hkl')   #for B12
#    #Int_ref_arry = CCB_pat_sim.get_Int_ref('/home/lichufen/CCB_ind/1azf.pdb.hkl')   # for lysozyme 1azf.
#    Int_sim = np.zeros((HKL_table1.shape[0],1))
#    for m in range(HKL_table1.shape[0]):
#		
#        HKL = HKL_table1[m,:]
#        k_in = K_in_table[m,:]
#        #print('k_in',k_in)
#        D_value = CCB_pat_sim.get_D(xtal_model_dict,k_in,delta_k_in=1e7)
#        #D_value = 1
#        P_value = CCB_pat_sim.get_P(ref_image,k_in)
#        #P_value = 1
#        k_in_cen = k_cen[frame,:]
#        q_vec = CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat@(HKL.reshape(3,1))
#        Lorf = CCB_pat_sim.get_Lorf(q_vec,k0)
#        Int = CCB_pat_sim.compt_Int_sim(Int_ref_arry,HKL,P_value,D_value,Lorf)
#        Int_sim[m] = Int
	##################################
#    Int_sim = Int_sim/1e2  #normalisation
#    output_arry = np.hstack((frame*np.ones((K_in_table.shape[0],1)),PXY0,Int_sim,HKL_table1,K_out_table,K_in_table))
#    ind_nan = np.isnan(Int_sim)+(Int_sim==0)
#    output_arry = output_arry[~ind_nan.reshape(-1,),:]

#    if save_K_sim_txt == True:
#        out_txt_file='K_map_sim_fr%d.txt'%(frame)
#        np.savetxt(out_txt_file,output_arry,fmt=['%3d','%7.1f','%7.1f','%7.2e','%3d','%3d','%3d','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e','%13.3e'])#need to insert the Int_sim entry
	#################################
    if save_fig == True:
        plt.figure(figsize=(10,10))
        plt.title('frame %d'%(frame))
        plt.imshow(exp_img)
        plt.xlim(0,4000)
        plt.ylim(0,4000)
        plt.clim(0,10)
        plt.scatter(PXY0[:,0],PXY0[:,1],s=0.02,marker='x',c='g')
        plt.savefig('match_'+'fr'+str(int(frame))+'.png')
    print('frame %d done!\n'%(frame))
    return None

if __name__=='__main__':
    exp_img_file=os.path.abspath(sys.argv[1])
    res_file=os.path.abspath(sys.argv[2])
    save_fig = bool(int(sys.argv[3]))
    save_K_sim_txt = bool(int(sys.argv[4]))
    res_arry=read_res(res_file)
    print('res_file: %s'%(res_file))
    print('%d frames loaded in the res_file'%(res_arry.shape[0]))
    for ind1 in range(res_arry.shape[0]):
        gen_single_match(exp_img_file,res_file,ind1,save_fig=save_fig,save_K_sim_txt=save_K_sim_txt)
    print('ALL DONE!!!')
