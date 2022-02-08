'''
CCB_streak_det.py is to dectect the diffraction streaks
for Convergent beam X-ray diffration images.
'''
import sys,os
sys.path.append(os.path.realpath(__file__))
os.system('export PYTHONUNBUFFERED=1')
import numpy as np
from skimage import measure, morphology, feature
import scipy
import glob
import h5py
import re
import CCB_ref
import CCB_pred
import CCB_pat_sim
import CCB_read
import gen_match_figs as gm
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import Xtal_calc_util as xu

#OR_mat=np.array([[ 4.47536571e+08,-1.33238725e+08,0.00000000e+00],\
#[9.38439088e+07,6.35408337e+08,0.00000000e+00],\
#[0.00000000e+00,0.00000000e+00,4.00000000e+08]])
#OR_mat=OR_mat/1.03

E_ph=17.4 #in keV
#wave_len=12.40/E_ph #in Angstrom
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),1/wave_len*np.ones((1500,1))))
k0 = 1/wave_len
k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),np.sqrt(k0**2-(3e8)**2-(2.2e8)**2)*np.ones((1500,1))))
k_cen = k_cen/(np.linalg.norm(k_cen,axis=1).reshape(-1,1))*1/wave_len

def single_peak_finder(exp_img_file,frame_no,thld=10,min_pix=15,mask_file='None',interact=False):
	img_arry=gm.read_frame(exp_img_file,frame_no,h5path='/entry/data/data')
	#img_arry=gm.read_frame(exp_img_file,frame_no,h5path='/data/simulated_data')#change path for sim data
	#bimg=(img_arry>thld)

	if mask_file!='None':
		mask_file=os.path.abspath(mask_file)
		m=h5py.File(mask_file,'r')
		mask=np.array(m['/data/data']).astype(bool)
		bkg = np.array(m['/data/bkg'])
		m.close()
	elif mask_file=='None':
		mask=np.ones_like(img_arry).astype(bool)
		bkg = 0
	else:
		sys.exit('the mask file option is inproper.')


	img_arry = img_arry - bkg
	bimg = (img_arry>thld)
	bimg=bimg*mask
	if 'sim' in exp_img_file:
		all_labels=measure.label(bimg,connectivity=2)
	else:
		all_labels=measure.label(bimg,connectivity=1)
	#all_labels=measure.label(bimg,connectivity=1) #connectivity is important here, for sim data,use 2, for exp data use 1
	props=measure.regionprops(all_labels,img_arry)

	area=np.array([r.area for r in props]).reshape(-1,)
	max_intensity=np.array([r.max_intensity for r in props]).reshape(-1,)
	
	major_axis_length=np.array([r.major_axis_length for r in props]).reshape(-1,)
	minor_axis_length=np.array([r.minor_axis_length for r in props]).reshape(-1,)
	aspect_ratio = major_axis_length/(minor_axis_length+1)
	#coords=np.array([r.coords for r in props]).reshape(-1,)
	label=np.array([r.label for r in props]).reshape(-1,)
	centroid=np.array([np.array(r.centroid).reshape(1,2) for r in props]).reshape((-1,2))
	weighted_centroid=np.array([r.weighted_centroid for r in props]).reshape(-1,)
	label_filtered=label[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
	area_filtered=area[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
	area_sort_ind=np.argsort(area_filtered)[::-1]
	label_filtered_sorted=label_filtered[area_sort_ind]
	area_filtered_sorted=area_filtered[area_sort_ind]
	weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))
	for index,value in enumerate(label_filtered_sorted):

        	weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
#	print('In image: %s \n %5d peaks are found' %(img_file_name, len(label_filtered_sorted)))
	#beam_center=np.array([1492.98,2163.41])

	if interact:
		plt.figure(figsize=(15,15))
		plt.imshow(img_arry*(mask.astype(np.int16)),cmap='viridis',origin='lower')
		plt.colorbar()
	#	plt.clim(0,0.5*thld)
		plt.clim(0,10)
		#plt.xlim(250,2100)
		#plt.ylim(500,2300)
		plt.scatter(weighted_centroid_filtered[:,1],weighted_centroid_filtered[:,0],edgecolors='r',facecolors='none')
		for label in label_filtered_sorted:
			plt.scatter(props[label-1].coords[:,1],props[label-1].coords[:,0],s=0.5)
	#	plt.scatter(beam_center[1],beam_center[0],marker='*',color='b')
		title_Str=exp_img_file+'\nEvent: %d '%(frame_no)
		plt.title(title_Str)
		plt.show()
	return label_filtered_sorted,weighted_centroid_filtered,props,img_arry,all_labels


def kout(exp_img_file,thld,min_pix,cam_len=0.15,pix_size=75e-6,beam_cx=1505,beam_cy=1182,mask_file='/home/lichufen/CCB_ind/mask.h5'):
	#exp_img_file=os.path.abspath(sys.argv[1])
	#thld=int(sys.argv[2])
	#min_pix=int(sys.argv[3])
	

	f=h5py.File(exp_img_file,'r')
	if 'sim_data' in exp_img_file:
		total_frame=f['data/simulated_data'].shape[0] # correct h5 path for sim data 
	else:
		total_frame=f['/entry/data/data'].shape[0]
	f.close()
	if 'sim_data' in exp_img_file:
		o=open('k_out_sim.txt','w',1)
	else:
		o=open('k_out.txt','w',1)
	for frame_no in range(total_frame):
		print('frame %d in process'%(frame_no))
		label_filtered_sorted,weighted_centroid_filtered,props,img_arry,all_labels=single_peak_finder(exp_img_file,frame_no,thld=thld,min_pix=min_pix,mask_file=mask_file)
		num_s=weighted_centroid_filtered.shape[0]

		end_point1 = np.array([[props[label-1].coords.min(axis=0)[0], props[label-1].coords.min(axis=0)[1]] if props[label-1].orientation>=0 else [props[label-1].coords.min(axis=0)[0], props[label-1].coords.max(axis=0)[1]]  for label in label_filtered_sorted])
		end_point2 = np.array([[props[label-1].coords.max(axis=0)[0], props[label-1].coords.max(axis=0)[1]] if props[label-1].orientation>=0 else [props[label-1].coords.max(axis=0)[0], props[label-1].coords.min(axis=0)[1]]  for label in label_filtered_sorted])
		
		
		k_out=np.hstack(((weighted_centroid_filtered[:,-1::-1]-np.array([beam_cx,beam_cy]).reshape(-1,2))*pix_size/cam_len,np.ones((num_s,1))))
		k_out=k_out/(np.linalg.norm(k_out,axis=-1).reshape(-1,1))
		
		end_vector1 = np.hstack(((end_point1[:,-1::-1]-np.array([beam_cx,beam_cy]).reshape(-1,2))*pix_size/cam_len,np.ones((num_s,1))))
		end_vector2 = np.hstack(((end_point2[:,-1::-1]-np.array([beam_cx,beam_cy]).reshape(-1,2))*pix_size/cam_len,np.ones((num_s,1))))
		end_vector1 = end_vector1/(np.linalg.norm(end_vector1,axis=-1).reshape(-1,1))
		end_vector2 = end_vector2/(np.linalg.norm(end_vector2,axis=-1).reshape(-1,1))
		diff_vector = end_vector2 - end_vector1
		diff_vector = diff_vector/(np.linalg.norm(diff_vector,axis=-1).reshape(-1,1))
		
		o.write('# Frame %d\n'%(frame_no))
		for m in range(num_s):

			o.write('%6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f\n'%(k_out[m,0],k_out[m,1],k_out[m,2],end_vector1[m,0],end_vector1[m,1],end_vector1[m,2],end_vector2[m,0],end_vector2[m,1],end_vector2[m,2],diff_vector[m,0],diff_vector[m,1],diff_vector[m,2]))
		print('Done.')
	o.close()
	print('ALL Done!')
	return None

if __name__=='__main__':
	exp_img_file=os.path.abspath(sys.argv[1])
	thld=int(sys.argv[2])
	min_pix=int(sys.argv[3])
	mask_file=os.path.abspath(sys.argv[4])
	os.system('export PYTHONUNBUFFERED=1')
	kout(exp_img_file,thld,min_pix,cam_len=0.2,pix_size=75e-6,beam_cx=1908,beam_cy=2207,mask_file=mask_file)		
	#kout(exp_img_file,thld,min_pix,cam_len=0.10,pix_size=75e-6,beam_cx=1594,beam_cy=1764,mask_file=mask_file)


		

