'''
'DE_gather.py' finds the best DE solution for each frame and 
dumps the solution to one file, the 'grand' file under /CBC_ind
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import CCB_ref
import CCB_pred
import CCB_pat_sim
import CCB_read
import h5py
import re
import gen_match_figs as gm
import glob2

def best_res(par_dir,frame):
	par_dir=os.path.abspath(par_dir)
	res_file_list=glob2.glob(par_dir+'/fr%d_%d'%(frame,frame)+'/round13/GA*.txt')
	if len(res_file_list) ==0:
		best_res_file=''
	else:
		TG_arry=np.zeros((len(res_file_list),))
		for m,res_file_name in enumerate(res_file_list):
			res_arry=gm.read_res(res_file_name)
			TG_arry[m]=res_arry[0,-1]
		ind=np.argsort(TG_arry)
		best_res_file=res_file_list[ind[0]]
	return best_res_file

def batch_best_res(par_dir,frame_start,frame_end):
	file_lines=[]
	for frame in range(frame_start,frame_end+1):
		best_res_file=best_res(par_dir,frame)
		if best_res_file=='':
			continue
		f=open(best_res_file,'r')
		new_file_lines=f.readlines()
		f.close()
		new_file_lines=['original res file:\n',best_res_file+'\n']+new_file_lines
		file_lines=file_lines+new_file_lines
	w=open('Best_GA_res_%d_%d.txt'%(frame_start,frame_end),'w')
	#print(file_lines)
	file_lines=''.join(file_lines)
	w.write(file_lines)
	w.close()
	return None

if __name__=='__main__':
	par_dir=sys.argv[1]
	#par_dir='/home/lichufen/CCB_ind/'
	frame_start=int(sys.argv[2])
	frame_end=int(sys.argv[3])
	batch_best_res(par_dir,frame_start,frame_end)
	print('Done!')



