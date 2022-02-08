'''
 a wrapper to run the CCB data analysis pipeline.

for gathering the best res files from the K_DE_refinement, and merging
'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import datetime
import CCB_int_proc
import CCB_pat_sim
import CCB_streak_det
import DE_gather

#print('gathering best res files from refinement')
#for i in range(1,10):
#	DE_gather.batch_best_res('.',i*10+1,i*10+10)
#	print('Done!')
#DE_gather.batch_best_res('.',0,10)
#print('Done!')

#os.system('cat Best_GA_res_*_*.txt > Best_GA_res.txt')

#os.system('python /home/lichufen/CCB_ind/scripts/CCB_kmap.py ./sim_data.h5 ./Best_GA_res.txt 0 100 15 20')
os.system('python /home/lichufen/CCB_ind/scripts/CCB_kmap.py ~/CCB_ind/scan_corrected_00135.h5 ./Best_GA_res.txt 0 100 15 20')
print('K_map txt files Done!')

os.system('rm -f K_map_sim_fr101.txt')
os.system('rm -f K_map_fr101.txt')
#os.system('cat K_map_sim_fr*.txt > K_map_sim_fr101.txt')
os.system('cat K_map_fr*.txt > K_map_fr101.txt')

#print('Generating Dataset object....\n counting time...')
st = datetime.datetime.now()
#d_sim = CCB_int_proc.Dataset('K_map_sim_fr101.txt')
d_exp = CCB_int_proc.Dataset('K_map_fr101.txt')
et = datetime.datetime.now()
dt = et - st

print('Generated, took %f seconds'%(dt.total_seconds()))
st = datetime.datetime.now()
#print('Merging sim data set...')
#d_sim.merge_all_HKL_crystfel()
#os.system('mv all_HKL_crystfel.hkl   sim_HKL_nomask_crystfel.hkl')
#print('Done!')

print('Merging exp data set...')
#d_exp.merge_all_HKL_crystfel()
#os.system('mv all_HKL_crystfel.hkl   ref_HKL_crystfel_ave0.hkl')
print('Done!')
et = datetime.datetime.now()
dt = et - st
print('Merging done, took %f seconds'%(dt.total_seconds()))

print('Start Bootstrapping!')
st = datetime.datetime.now()
os.system('python ../Bootstrapping.py ./K_map_fr101.txt 20 -6 -4 1')
et = datetime.datetime.now()
dt = et - st
print('Bootstrapping done, took %f minutes'%(dt.total_seconds()/60))

