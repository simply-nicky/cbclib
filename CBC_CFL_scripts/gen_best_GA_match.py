'''
gen_best_GA_match.py is to generate the matching figures
for the best DE refinement solutions.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import CCB_ref
import CCB_pred
import CCB_pat_sim
import CCB_read
import h5py
import re
import gen_match_figs as gm
import glob2

def read_best_res_dir(res_file):
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
    best_res_round_dir_list=[]
    for  m,ind in enumerate(frame_ind_list):
        frame=int(re.split(' ',lines[ind])[1])
        best_res_round_dir=os.path.dirname(lines[ind-1][:-1])
        best_res_round_dir_list.append(best_res_round_dir)
    return best_res_round_dir_list

if __name__=='__main__':
    par_dir=os.path.abspath(sys.argv[1])
    exp_img_file=os.path.abspath(sys.argv[2])
    save_fig = sys.argv[3]
    save_K_sim_txt = sys.argv[4]
    best_GA_res_file_list=glob2.glob(par_dir+'/Best_GA_res_*_*.txt')
    for best_GA_res_file in best_GA_res_file_list:
        best_res_round_dir_list=read_best_res_dir(best_GA_res_file)
        #for best_res_round_dir in best_res_round_dir_list:
            #os.system('cp -fr '+best_res_round_dir+'/line*.png'+' ./.')
        os.system('sbatch /gpfs/cfel/user/lichufen/CBDXT/P11_BT/scripts/best_GA_match.sh '+exp_img_file+' '+best_GA_res_file+' '+save_fig+' '+save_K_sim_txt)
        print('batch job submitted')

