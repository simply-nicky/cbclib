import sys,os
sys.path.append('/home/lichufen/CCB_ind/scripts/')
import numpy as np 
import batch_refine
R_mat=np.array([[ 2.1146e+08,-3.7666e+08,1.1832e+08],\
[-2.1310e+08,-1.8828e+08,-3.1376e+08],\
[5.3262e+08,8.4683e+07,-1.6805e+08]])
OR_mat=OR_mat/1.0


x0 = OR_mat.T.reshape(-1,)                                              
x_l = x0-4e7                                                            
x_h = x0+4e7                                                            
bounds = tuple([(x_l[m],x_h[m]) for m in range(9)])                     
res = batch_refine.Latt_refine(np.arange(0,100),x0,bounds,'../K_ref_test2/Best_GA_res.txt')      

