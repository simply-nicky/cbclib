'''
latt_refine_batch.py does the lattice refinement in a batch,
after the DE refinement.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import h5py
import Xtal_calc_util as xu
import CCB_ref
import CCB_pred
import CCB_read
import CCB_pat_sim
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import scipy.optimize
import batch_refine
import gen_match_figs as gm

#a=15.4029218
#b=21.86892773
#c=25
#Alpha=90
#Beta=90
#Gamma=90
OR_mat=np.array([[ 2.1146e+08,-3.7666e+08,1.1832e+08],\
[-2.1310e+08,-1.8828e+08,-3.1376e+08],\
[5.3262e+08,8.4683e+07,-1.6805e+08]])
OR_mat=OR_mat/1.0

E_ph=17.4 #in keV
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
k_cen=np.array([0,0,1/wave_len]).reshape(3,1)

def latt_frame_refine(ind1,res_file):
    res_arry=gm.read_res(res_file)
    frame=int(res_arry[ind1,0])
    OR_angs=tuple(res_arry[ind1,1:4])
    cam_len=res_arry[ind1,4]
    k_out_osx=res_arry[ind1,5]
    k_out_osy=res_arry[ind1,6]
    theta,phi,alpha=OR_angs
    OR=CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat
    OR=OR*1e-8
    k_out_osx=k_out_osx*1e2
    K_out_osy=k_out_osy*1e2
    x0=tuple(np.concatenate(((OR.T).reshape(-1,),np.array([cam_len,k_out_osx,k_out_osy])),axis=-1))
    x0_GA=tuple(res_arry[ind1,1:7])
    args=(frame,x0_GA)
    #print('Refining OR for frame %d'%(frame))
    bounds=((x0[0]-0.05,x0[0]+0.05),(x0[1]-0.05,x0[1]+0.05),(x0[2]-0.05,x0[2]+0.05),(x0[3]-0.05,x0[3]+0.05),(x0[4]-0.05,x0[4]+0.05),(x0[5]-0.05,x0[5]+0.05),(x0[6]-0.05,x0[6]+0.05),(x0[7]-0.05,x0[7]+0.05),(x0[8]-0.05,x0[8]+0.05),(x0[9]-0.2,x0[9]+0.2),(x0[10]-0,x0[10]+0),(x0[11]-0,x0[11]+0))
    #res = scipy.optimize.minimize(CCB_ref._TG_func6, x0, args=args, bounds=bounds,method='L-BFGS-B', options={'disp': True})
    #res = scipy.optimize.dual_annealing(CCB_ref._TG_func6, bounds, args, x0=x0, maxiter=1000)
    res = scipy.optimize.differential_evolution(CCB_ref._TG_func6,bounds,args=args,strategy='best1bin',disp=True,polish=True)
	
    print(res.x)
    amp_fact=res.x[9]
    kosx,kosy=res.x[10]*1e-2,res.x[11]*1e-2
    #lp=np.array([1e-10*res.x[6],1e-10*res.x[7],1e-10*res.x[8],res.x[9],res.x[10],res.x[11]])
    #_,OR_mat=xu.A_gen(lp)
    #OR_start=CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame)@CCB_ref.rot_mat_zaxis(11.84)@OR_mat
    #OR=CCB_ref.Rot_mat_gen(res.x[0],res.x[1],res.x[2])@OR_start
    OR=(np.array(res.x[0:9]).reshape(3,3)).T*1e8
    K_out, K_in_pred, K_out_pred=batch_refine.point_match(frame,OR,amp_fact,kosx,kosy,E_ph)
    res_cut=1
    HKL_table, K_in_table, K_out_table=CCB_pat_sim.pat_sim_q(OR,res_cut)
    K_in_pred_s,K_out_pred_s=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_table[:,0:3])



    plt.figure(figsize=(10,10))
    plt.scatter(K_out_table[:,0],K_out_table[:,1],s=1,marker='x',c='g')
    plt.scatter(K_out[:,0],K_out[:,1],s=20,marker='x',color='b')
    plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=20,marker='x',color='r')
    plt.scatter(K_out_pred_s[:,0],K_out_pred_s[:,1],s=40,marker='o',edgecolor='black',facecolor='None')
    plt.axis('equal')
    plt.savefig('line_match_latt_ref_frame%03d.png'%(frame))
    plt.close('all')


    return res


def latt_batch_refine(res_file,out_file):
    f=open(out_file,'a',1)
    f.write('The GA_refine res_file is:\n%s\n'%(os.path.abspath(res_file)))
    f.write('====================================\n')
    res_arry=gm.read_res(res_file)
    print('res_file: %s'%(os.path.abspath(res_file)))
    print('%d frames loaded in the res_file'%(res_arry.shape[0]))
    for ind1 in range(res_arry.shape[0]):
        frame=int(res_arry[ind1,0])
        print('Lattice Refining frame %03d'%(frame))
        res=latt_frame_refine(ind1,res_file)
        f.write('frame %03d \n'%(frame))
        f.write('TG before Lattice refinement: %7.3e \n'%(res_arry[ind1,-1]))
        f.write('TG after Lattice refinement: %7.3e \n'%(res.fun))
        f.write('res: \n')
        #f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5]))
        f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0]*1e8,res.x[1]*1e8,res.x[2]*1e8,res.x[3]*1e8,res.x[4]*1e8,res.x[5]*1e8,res.x[6]*1e8,res.x[7]*1e8,res.x[8]*1e8,res.x[9],res.x[10]*1e-2,res.x[11]*1e-2))
        f.write('------------------------------------\n')
        print('Done!')
    f.close()
    return

if __name__=='__main__':
    res_file=os.path.abspath(sys.argv[1])
    out_file=os.path.abspath(sys.argv[2])
    latt_batch_refine(res_file,out_file)
