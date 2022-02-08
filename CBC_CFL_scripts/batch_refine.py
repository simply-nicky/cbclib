'''
batch_refine.py does the refinement in a batch.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import matplotlib
import h5py
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import Xtal_calc_util as xu
import CCB_ref
#import CCB_pred
import CCB_read
import CCB_pat_sim
import matplotlib.pyplot as plt
import scipy.optimize


#a=15.4029218
#b=21.86892773
#c=25
#Alpha=90
#Beta=90
#Gamma=90

#OR_mat=np.array([[-2.05112078e+08,-3.89499652e+08,-1.22739594e+08],
#[-2.21095123e+08,1.95462074e+08,-3.25490173e+08],
#[ 5.57086534e+08,-8.72801443e+07,-1.74338424e+08]])
OR_mat = np.genfromtxt('../../OR.txt')
OR_mat=OR_mat/1.0



###################
# for expanding lattice constants
expanding_const = 1
OR_mat = OR_mat/expanding_const
##################


E_ph=17.4 #in keV
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=1/wave_len*np.array([-0.03115,-0.02308,0.999248]).reshape(3,1)
#k_cen = np.genfromtxt('/home/lichufen/CCB_ind/k_cen.txt')
#k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),1/wave_len*np.ones((1500,1))))
k0 = 1/wave_len
k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),np.sqrt(k0**2-(3e8)**2-(2.2e8)**2)*np.ones((1500,1))))
k_cen = k_cen/(np.linalg.norm(k_cen,axis=1).reshape(-1,1))*1/wave_len

def point_match(frame,OR,amp_fact,kosx,kosy,E_ph):
    #frame=0
    #amp_fact=1
    #kosx,kosy=0,0
    #OR=CCB_ref.rot_mat_zaxis(0)@CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat


    #amp_fact=res.x[-3]
    #kosx,kosy=res.x[-2],res.x[-1]
    #OR=CCB_ref.Rot_mat_gen(res.x[0],res.x[1],res.x[2])@CCB_ref.rot_mat_yaxis(-frame)@OR_mat

    #amp_fact=1.008
    #kosx,kosy=3.614829e-2,5.833e-3
    #OR=CCB_ref.Rot_mat_gen(6.1267e1,-9.697e1,2.424e0)@CCB_ref.rot_mat_yaxis(-frame)@OR_mat

    #OR=Rot_mat@OR
    #OR=x_arry[-1,0:9].reshape(3,3)
    #OR=OR_V
    #OR=OR_refd.reshape(3,3)
    #print(CCB_ref.rot_mat_yaxis(-frame+0)@OR_mat)
    #print(OR)
    E_ph=17.4
    wave_len=1e-10*12.40/E_ph
    frac_offset=np.array([0,0,0])

    #kout_dir_dict=CCB_read.kout_read('/home/lichufen/CCB_ind/k_out.txt')
    kout_dir_dict=CCB_read.kout_read('../../k_out.txt')
    print('reading the file %s'%(os.path.abspath('../../k_out.txt')))
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)
    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    Diff_vector = kout_dict['diff_vector_'+str(frame)] # This is for q,streak constraint.
    #HKL_frac, HKL_int, Q_int, Q_resid = CCB_ref.get_HKL(OR,Q_arry,np.array([0,0,0]))
    frac_offset=np.array([0,0,0])
    HKL_frac, HKL_int, Q_int, Q_resid = CCB_ref.get_HKL8(OR,Q_arry,frac_offset)
    Delta_k, Dist, Dist_1=CCB_ref.exctn_error8_nr(k_cen[frame,:],OR,Q_arry,Q_int,frac_offset,E_ph)

    K_in_arry = K_out.reshape(-1,3,1) - Q_int #the shape of Q_int and HKL_int is (num,3,8)



    ind=np.argsort(np.linalg.norm(K_in_arry-k_cen[frame,:].reshape(-1,3,1),axis=1),axis=1)

    #ind=np.argsort(Dist,axis=1)


    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])



    #K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    K_in_pred,K_out_pred=CCB_pat_sim.kout_pred(OR,k_cen[frame,:],HKL_int)
    #Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_in_new=K_in_pred-k_cen[frame,:].reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out

    #K_in_pred,K_out_pred=CCB_pred.kout_pred8(OR,[0,0,1/wave_len],HKL_int)
    #Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3,1)
    #Delta_k_out_new=K_out_pred-K_out.reshape(-1,3,1)
    #Dist2=np.linalg.norm(Delta_k_out_new,axis=1)
    #ind2=np.argsort(Dist2,axis=1)
    #ind2=np.array([ind2[m,0] for m in range(ind2.shape[0])])
    #Dist2=np.array([Dist2[m,ind2[m]] for m in range(Dist2.shape[0])])
    #HKL_int=np.array([HKL_int[m,:,ind2[m]] for m in range(HKL_int.shape[0])])
    #Delta_k=np.array([Delta_k[m,:,ind2[m]] for m in range(Delta_k.shape[0])])
    #K_in_pred=np.array([K_in_pred[m,:,ind2[m]] for m in range(K_in_pred.shape[0])])
    #K_out_pred=np.array([K_out_pred[m,:,ind2[m]] for m in range(K_out_pred.shape[0])])
    #Delta_k_in_new=np.array([Delta_k_in_new[m,:,ind2[m]] for m in range(Delta_k_in_new.shape[0])])
    #Delta_k_out_new=np.array([Delta_k_out_new[m,:,ind2[m]] for m in range(Delta_k_out_new.shape[0])])


    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<10e8
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<10e8
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]
    K_in_pred=K_in_pred[ind_filter,:]

    #plt.figure()
    #plt.scatter(Delta_k_in_new[:,0],Delta_k_in_new[:,1],s=1,marker='x',color='b')
    #plt.axis('equal')
    #plt.xticks(np.linspace(-5e8,5e8,5));
    #plt.yticks(np.linspace(-5e8,5e8,5));
    #plt.xlim(-5e8,5e8)
    #plt.ylim(-5e8,5e8)

    #plt.figure()
    #plt.scatter(Delta_k_out_new[:,0],Delta_k_out_new[:,1],s=1,marker='x',color='b')
    #plt.axis('equal')
    #plt.xticks(np.linspace(-5e8,5e8,5));
    #plt.yticks(np.linspace(-5e8,5e8,5));
    #plt.xlim(-5e8,5e8)
    #plt.ylim(-5e8,5e8)


    #plt.figure(figsize=(10,10))
    #plt.scatter(K_out[:,0],K_out[:,1],s=1,marker='x',color='b')
    #plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=1,marker='x',color='r')
    #plt.axis('equal')
    #plt.savefig('point_match_frame%03d'%(frame)+'.png')
    #plt.close('all')

    return K_out, K_in_pred, K_out_pred

def GA_refine(frame,bounds):
    args=(frame,)
    #bounds=((0,90),(-180,180),(-6,6),(0.95,1.05),(-5e-2,5e-2),(-5e-2,5e-2),(-0.1,0.1),(-0.1,0.1),(-0.1,0.1),(-3,3),(-3,3),(-3,3))
    res = scipy.optimize.differential_evolution(CCB_ref._TG_func3,bounds,args=args,strategy='best1bin',disp=True,polish=True)
    print('intial','TG: %7.3e'%CCB_ref._TG_func3(np.array([0,0,0,1,0,0]),frame))
    print('final',res.x,'TG: %7.3e'%CCB_ref._TG_func3(res.x,frame))
    #print('intial','TG: %7.3e'%CCB_ref._TG_func5(np.array([0,0,0,1,0,0,a,b,c,Alpha,Beta,Gamma]),frame))
    return res

def Latt_refine(frame_list,x0,bounds,res_file):
    args = (frame_list,res_file,)
    #res = scipy.optimize.fmin_l_bfgs_b(CCB_ref._TG_func8,x0,bounds=bounds,approx_grad=1,args=args,disp=1)
    #res = scipy.optimize.minimize(CCB_ref._TG_func8,x0,bounds=bounds,args=args,options={'disp':1})
    res = scipy.optimize.differential_evolution(CCB_ref._TG_func8,bounds,args=args,strategy='best1bin',disp=True,polish=True,workers=4)
    
    print('initial','TG: %7.3e'%CCB_ref._TG_func8(x0,frame_list,res_file))
    print('final',res.x,'\nTG: %7.3e'%CCB_ref._TG_func8(res.x,frame_list,res_file))
    return res

def frame_refine(frame,res_cut=1,E_ph=17.4):
    wave_len= 1e-10*12.40/E_ph
    k0=1/wave_len
    #k_in_cen=np.array([0,0,k0]).reshape(3,1)
    #k_in_cen=np.array([0,0,1/wave_len]).reshape(3,1)
    k_in_cen=k_cen[frame,:].reshape(3,1)
    a=15.4029218
    b=21.86892773
    c=25
    Alpha=90
    Beta=90
    Gamma=90
    #OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
    #[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
    #[0.00000000e+00,0.00000000e+00,4.00000000e+08]])
    #OR_mat=OR_mat/1.03
    
    ###################
    # for expanding lattice constants
    #expanding_const = 2
    #OR_mat = OR_mat/expanding_const
    ##################


    amp_fact=1
    kosx,kosy=0,0
    OR=CCB_ref.rot_mat_yaxis(frame*0.5)@OR_mat
    #print(OR)
    K_out, K_in_pred, K_out_pred=point_match(frame,OR,amp_fact,kosx,kosy,E_ph)
    HKL_table, K_in_table, K_out_table=CCB_pat_sim.pat_sim_q(k_in_cen,OR,res_cut)
    K_in_pred_s,K_out_pred_s=CCB_pat_sim.kout_pred(OR,k_in_cen,HKL_table[:,0:3])
    plt.figure(figsize=(10,10))
    plt.scatter(K_out_table[:,0],K_out_table[:,1],s=1,marker='x',c='g')
    plt.scatter(K_out[:,0],K_out[:,1],s=20,marker='x',color='b')
    plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=20,marker='x',color='r')
    plt.scatter(K_out_pred_s[:,0],K_out_pred_s[:,1],s=40,marker='o',edgecolor='black',facecolor='None')
    plt.axis('equal')
    plt.savefig('line_match_before_frame%03d.png'%(frame))

    bounds=((0,90),(-180,180),(-5,5),(0.90,1.10),(-5e-2,5e-2),(-5e-2,5e-2))
    res=GA_refine(frame,bounds)
    #f.write('frame %03d \n'%(frame))
    #f.write('intial','TG: %7.3e'%CCB_ref._TG_func3(np.array([0,0,0,1,0,0]),frame))
    #f.write('final',res.x,'TG: %7.3e'%CCB_ref._TG_func3(res.x,frame))
    #f.write('------------------------------------\n')
    amp_fact=res.x[3]
    kosx,kosy=res.x[4],res.x[5]
    #lp=np.array([1e-10*res.x[6],1e-10*res.x[7],1e-10*res.x[8],res.x[9],res.x[10],res.x[11]])
    #_,OR_mat=xu.A_gen(lp)
    #OR_start=CCB_ref.rot_mat_xaxis(0)@CCB_ref.rot_mat_yaxis(-frame)@CCB_ref.rot_mat_zaxis(11.84)@OR_mat
    #OR=CCB_ref.Rot_mat_gen(res.x[0],res.x[1],res.x[2])@OR_start
    OR=CCB_ref.Rot_mat_gen(res.x[0],res.x[1],res.x[2])@CCB_ref.rot_mat_yaxis(0.5*frame)@OR_mat
    K_out, K_in_pred, K_out_pred=point_match(frame,OR,amp_fact,kosx,kosy,E_ph)
    HKL_table, K_in_table, K_out_table=CCB_pat_sim.pat_sim_q(k_in_cen,OR,res_cut)
    #K_in_pred_s,K_out_pred_s=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_table[:,0:3])
    K_in_pred_s,K_out_pred_s=CCB_pat_sim.kout_pred(OR,k_in_cen,HKL_table[:,0:3])


    plt.figure(figsize=(10,10))
    plt.scatter(K_out_table[:,0],K_out_table[:,1],s=1,marker='x',c='g')
    plt.scatter(K_out[:,0],K_out[:,1],s=20,marker='x',color='b')
    plt.scatter(K_out_pred[:,0],K_out_pred[:,1],s=20,marker='x',color='r')
    plt.scatter(K_out_pred_s[:,0],K_out_pred_s[:,1],s=40,marker='o',edgecolor='black',facecolor='None')
    plt.axis('equal')
    plt.savefig('line_match_after_frame%03d.png'%(frame))
    plt.close('all')
    #plt.figure(figsize=(10,10))
    #plt.scatter(K_in_table[:,0],K_in_table[:,1],s=4,marker='o',c='g')
    #plt.scatter(K_in_pred_s[:,0],K_in_pred_s[:,1],s=4,marker='o',c='black')
    #plt.scatter(K_in_pred[:,0],K_in_pred[:,1],s=4,marker='o',c='r')

    #plt.figure(figsize=(10,10))
    #plt.scatter(Delta_k_in_new[:,0],Delta_k_in_new[:,1],s=10,marker='o',c=np.linalg.norm(Delta_k_out_new,axis=1),cmap='jet')
    #plt.colorbar()

    return res

def batch_refine(start_frame,end_frame):
    f=open('GA_refine.txt','a',1)

    for frame in range(start_frame,end_frame+1):
        print('Refining frame %03d'%(frame))
        res=frame_refine(frame,res_cut=2*expanding_const,E_ph=17.4)
        f.write('frame %03d \n'%(frame))
        f.write('intial TG: %7.3e \n'%CCB_ref._TG_func3(np.array([0,0,0,1,0,0]),frame))
        f.write('final TG: %7.3e \n'%CCB_ref._TG_func3(res.x,frame))
        f.write('res: \n')
        f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5]))
        #f.write('%7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n'%(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],res.x[6],res.x[7],res.x[8],res.x[9],res.x[10],res.x[11]))
        f.write('------------------------------------\n')
    f.close()
    return

if __name__=='__main__':
    start_frame=int(sys.argv[1])
    end_frame=int(sys.argv[2])
    batch_refine(start_frame,end_frame)
