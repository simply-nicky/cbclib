import CCB_pred
import CCB_ref
import CCB_read
import importlib
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
importlib.reload(CCB_pred)
importlib.reload(CCB_ref)
importlib.reload(CCB_read)
import matplotlib.animation as animation
OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
[0.00000000e+00,0.00000000e+00,4.00000000e+08]])

def para_plot(frame,OR,amp_fact):
    #frame=45
    #amp_fact=1.0
    #OR=CCB_ref.rot_mat_zaxis(0)@CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat
    print(CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat)
    print(OR)
    E_ph=17
    wave_len=1e-10*12.04/E_ph
    frac_offset=np.array([0,0,0])

    kout_dir_dict=CCB_read.kout_read('./k_out.txt')
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact)
    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    HKL_frac, HKL_int, Q_int, Q_resid = CCB_ref.get_HKL(OR,Q_arry,np.array([0,0,0]))



    K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)

    Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out

    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<10e9
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<10e9
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]


    fig1=plt.figure(1)
    plt.scatter(Delta_k_in_new[:,0],Delta_k_in_new[:,1],s=1,marker='x',color='b')
    plt.axis('equal')
    plt.xticks(np.linspace(-5e8,5e8,5));
    plt.yticks(np.linspace(-5e8,5e8,5));
    plt.xlim(-5e8,5e8)
    plt.ylim(-5e8,5e8)

    fig2=plt.figure(2)
    plt.scatter(Delta_k_out_new[:,0],Delta_k_out_new[:,1],s=1,marker='x',color='b')
    plt.axis('equal')
    plt.xticks(np.linspace(-5e8,5e8,5));
    plt.yticks(np.linspace(-5e8,5e8,5));
    plt.xlim(-5e8,5e8)
    plt.ylim(-5e8,5e8)

    fig3=plt.figure(3,figsize=(10,10))
    plt.scatter(K_out[:,0],K_out[:,1],s=1,marker='x',color='b')
    plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=1,marker='x',color='r')
    plt.xlim(-10e9,10e9)
    plt.ylim(-10e9,10e9)
    #plt.axis('equal')

    return fig1,fig2,fig3

def x_rot_scan(frame,amp_fact=1,range=1,step=0.1):
    OR_start=CCB_ref.rot_mat_zaxis(0)@CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat
    fig=plt.figure()
    ims=[]
    for ang in np.arange(-range,range,step):
        OR=CCB_ref.rot_mat_xaxis(ang)@OR_start
        fig1,fig2,fig3=para_plot(frame,OR,amp_fact=1)
        plt.figure(3)
        plt.title('frame:%d xAng=%5.3f L=%5.3f'%(frame,xAng,amp_fact))
        plt.savefig('frame%d_xang%s.png'%(frame,ang))
        plt.close(3)

def y_rot_scan(frame,amp_fact=1,range=1,step=0.1):
    OR_start=CCB_ref.rot_mat_zaxis(0)@CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat
    fig=plt.figure()
    ims=[]
    for ang in np.arange(-range,range,step):
        OR=CCB_ref.rot_mat_yaxis(ang)@OR_start
        fig1,fig2,fig3=para_plot(frame,OR,amp_fact=1)
        plt.figure(3)
        plt.title('frame:%d yAng=%5.3f L=%5.3f'%(frame,yAng,amp_fact))
        plt.savefig('frame%d_yang%5.3f.png'%(frame,ang))
        plt.close(3)

def L_scan(frame,range=0.1,step=0.01):
    OR_start=CCB_ref.rot_mat_zaxis(0)@CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat
    fig=plt.figure()
    ims=[]
    for amp_fact in np.arange(1-range,1+range,step):
        OR=OR_start
        fig1,fig2,fig3=para_plot(frame,OR,amp_fact=amp_fact)
        plt.figure(3)
        plt.title('frame:%d L=%5.3f'%(frame,amp_fact))
        plt.savefig('frame:%d_L%5.3f.png'%(frame,amp_fact))
        plt.close(3)


    return None
