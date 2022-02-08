'''
This is a wrapper to run the CCB data analysis pipeline.

before the K_refinement is done.
'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import datetime
import CCB_int_proc
import subprocess


os.system('python /home/lichufen/CCB_ind/scripts/gen_best_GA_match.py /home/lichufen/CCB_ind/exp_res_files /home/lichufen/CCB_ind/scan_corrected_00135.h5 0 1')

now = datetime.datetime.now()
now_str = now.strftime('%Y %m %d %H:%M:%S')
print('Now batch jobs submitted for simulating K_sim_fr files')
print('the submission time is %s'%(now_str))


