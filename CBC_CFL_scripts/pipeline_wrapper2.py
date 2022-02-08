'''
This is a wrapper to run the CCB data analysis pipeline.

for converting the K_map_sim txt files to h5 image and running  the K_refinement.
'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import datetime
import CCB_int_proc
import CCB_pat_sim
import CCB_streak_det

os.system('rm -f K_map_sim_fr101.txt')
#CCB_pat_sim.sim_txts2h5('./K_map_sim_fr*.txt',(2167,2070))


#os.system('python /home/lichufen/CCB_ind/scripts/CCB_streak_det.py ./sim_data.h5 15 20 None')
os.system('python /home/lichufen/CCB_ind/scripts/CCB_streak_det.py ~/CCB_ind/scan_corrected_00135.h5  15 20 ~/CCB_ind/mask.h5')
#os.system('cp ./k_out_sim.txt ./k_out.txt')

os.system('for i in {0..100};do sbatch /home/lichufen/CCB_ind/scripts/b_refine.sh $i $i;done')
now = datetime.datetime.now()
now_str = now.strftime('%Y %m %d %H:%M:%S')
print('Now batch jobs submitted for K-DE-refinement')
print('the submission time is %s'%(now_str))


