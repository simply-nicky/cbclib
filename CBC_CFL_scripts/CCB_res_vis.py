'''
CCB_res_vis.py is to visulize the the indexing solutions 
taking the Best_GA*.txt as input file.

'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import CCB_ref
#import CCB_pred
#import CCB_pat_sim
import CCB_read
import h5py
import re
#import gen_match_figs as gm
import glob2

class Res_File:
	
	def __init__(self, file_name):
		file_name = os.path.abspath(file_name)
		self.file_name = file_name
		with open(file_name,'r') as f:
			file_string_list = f.readlines()
		self.file_string_list = file_string_list
		self.get_info_list()
			
	def get_info_list(self):

		ind_list=[]
		for ind, l in enumerate(self.file_string_list):
			if l.startswith('frame'):
				ind_list.append(ind)
		self.ind_list = ind_list
		
		frame_list=[]
		ini_TG_list=[]
		final_TG_list=[]
		theta_list=[]
		phi_list=[]
		alpha_list=[]
		cam_len_list=[]
		kosx_list=[]
		kosy_list=[]
		for ind in self.ind_list:
			frame = int(self.file_string_list[ind].strip().split(' ')[1])
			frame_list.append(frame)
			ini_TG_list.append(float(self.file_string_list[ind+1].strip().split(': ')[1]))
			final_TG_list.append(float(self.file_string_list[ind+2].strip().split(': ')[1]))
			theta_list.append(float(self.file_string_list[ind+4].strip().split(' ')[0]))
			phi_list.append(float(self.file_string_list[ind+4].strip().split(' ')[1]))
			alpha_list.append(float(self.file_string_list[ind+4].strip().split(' ')[2]))
			cam_len_list.append(float(self.file_string_list[ind+4].strip().split(' ')[3]))
			kosx_list.append(float(self.file_string_list[ind+4].strip().split(' ')[4]))
			kosy_list.append(float(self.file_string_list[ind+4].strip().split(' ')[5]))
		self.frame_list = frame_list
		self.ini_TG_list = ini_TG_list
		self.final_TG_list = final_TG_list
		self.cam_len_list = cam_len_list
		self.kosx_list = kosx_list
		self.kosy_list = kosy_list	

	def plot_geom(self):

		fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(5,8))
		ax[0].plot(self.frame_list,1/np.array(self.cam_len_list),'o',ms=3)
		ax[0].set_title('Cam_len')
		ax[1].plot(self.frame_list,self.kosx_list,'o',ms=3)
		ax[1].set_title('kosx')
		ax[2].plot(self.frame_list,self.kosy_list,'o',ms=3)
		ax[2].set_title('kosy')
		ax[2].set_xlabel('frame')
		i = 0
		while os.path.isfile('res_geom_'+str(i)+'.pdf'):
			i = i + 1
		
			
		plt.savefig('res_geom_'+str(i)+'.pdf')
	def plot_TG(self):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
		ax.plot(self.frame_list,self.ini_TG_list,'bx',self.frame_list,self.final_TG_list,'go',ms=3)
		ax.legend(['initial_TG','final_TG'])
		ax.set_xlabel('frame')
		ax.set_ylabel(r'$residual(m^{-1})$')
		plt.tight_layout()
		i = 0
		while os.path.isfile('res_TG_'+str(i)+'.pdf'):
			i = i + 1
		plt.savefig('res_TG_'+str(i)+'.pdf')
if __name__=='__main__':
	res_file = os.path.abspath(sys.argv[1])
	res = Res_File(res_file)
	res.plot_geom()
	res.plot_TG()
	plt.show()

