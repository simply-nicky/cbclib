'''
DE_output.py writes the OR,K_in, K_out, HKL_in for each frame in the order given
by 'kout.txt' (created by Nikokay), so that he can match up
with each 'feature' detected for later integration and K_in mapping.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import matplotlib
import h5py
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import Xtal_calc_util as xu
import CCB_ref
import CCB_pred
import CCB_read
import CCB_pat_sim
import matplotlib.pyplot as plt
import scipy.optimize
import gen_match_figs as gm

OR_mat=np.array([[ 2.1146e+08,-3.7666e+08,1.1832e+08],\
[-2.1310e+08,-1.8828e+08,-3.1376e+08],\
[5.3262e+08,8.4683e+07,-1.6805e+08]])
OR_mat=OR_mat/1.0


E_ph=17.4 #in keV
#wave_len=12.40/E_ph #in Angstrom
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
def point_match(frame,OR,amp_fact,kosx,kosy,E_ph):
    frac_offset=np.array([0,0,0])

    kout_dir_dict=CCB_read.kout_read('/home/lichufen/CCB_ind/k_out.txt')
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)
    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    #HKL_frac, HKL_int, Q_int, Q_resid = CCB_ref.get_HKL(OR,Q_arry,np.array([0,0,0]))
    frac_offset=np.array([0,0,0])
    HKL_frac, HKL_int, Q_int, Q_resid = CCB_ref.get_HKL8(OR,Q_arry,frac_offset)
    Delta_k, Dist, Dist_1=CCB_ref.exctn_error8_nr(OR,Q_arry,Q_int,frac_offset,E_ph)

    ind=np.argsort(Dist,axis=1)

    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])

    K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out
    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<100e8
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<100e8
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]
    K_in_pred=K_in_pred[ind_filter,:]

    return K_out, K_in_pred, K_out_pred, HKL_int, Q_int


if __name__=='__main__':
    os.system('export PYTHONUNBUFFERED=1');
    res_file=os.path.abspath(sys.argv[1])
    res_arry=gm.read_res(res_file)
    print('res_file: %s'%(res_file))
    print('%d frames loaded in the res_file'%(res_arry.shape[0]))
    f=open('DE_output.txt','w')
    o=open('DE_orientation_matrix.txt','w')
    f.write('# DE refine res_file:\n'+res_file+'\n')
    o.write('# DE refine res_file:\n'+res_file+'\n')
    f.write('#   H   K   L   Q(3 cols)   K_out(3 cols)    K_in_pred(3 cols)   k_out_pred(3 cols)\n')
    o.write('# asx asy asz bsx bsy bsz csx csy csz\n')
    for ind1 in range(res_arry.shape[0]):
        frame=int(res_arry[ind1,0])
        OR_angs=tuple(res_arry[ind1,1:4])
        theta,phi,alpha=OR_angs
        OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat
        cam_len=res_arry[ind1,4]
        k_out_osx=res_arry[ind1,5]
        k_out_osy=res_arry[ind1,6]
        K_out, K_in_pred, K_out_pred, HKL_int, Q_int=point_match(frame,OR,cam_len,k_out_osx,k_out_osy,E_ph)
        Q=OR@(HKL_int.T)
        Q=Q.T
        f.write('# frame %d\n'%(frame))
        o.write('# frame %d\n'%(frame))
        OR_flat=OR.T.reshape(-1,)
        o.write(('%7.3e '*9+'\n')%(OR_flat[0],OR_flat[1],OR_flat[2],OR_flat[3],OR_flat[4],OR_flat[5],OR_flat[6],OR_flat[7],OR_flat[8]))
        #print(HKL_int.shape,K_out.shape,K_out_pred.shape,K_in_pred.shape,Q_int.shape)
        for m in range(K_out.shape[0]):
            f.write('%3d %3d %3d %13.3e %13.3e %13.3e %13.3e %13.3e %13.3e %13.3e %13.3e %13.3e %13.3e %13.3e %13.3e\n'%(HKL_int[m,0],HKL_int[m,1],HKL_int[m,2],Q[m,0],Q[m,1],Q[m,2],K_out[m,0],K_out[m,1],K_out[m,2],K_in_pred[m,0],K_in_pred[m,1],K_in_pred[m,2],K_out_pred[m,0],K_out_pred[m,1],K_out_pred[m,2]))
        print('frame %d output done!'%(frame))
    f.close()
    o.close()
