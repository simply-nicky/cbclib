'''
movie_mk.py read all images and makes a movie out these.

'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import cv2
import glob2

def get_frame(file_name):
	frame=file_name.split('fr')[-1].split('.')[0] 
	return int(frame) 
def mk_movie(file_name_pattern,out_name):
	img_list=[]

	file_name_list=glob2.glob(file_name_pattern)
	#print(file_name_list)
	file_name_list.sort(key=get_frame)
	#print(file_name_pattern)
	print(file_name_list)
	for filename in file_name_list:
		#filename=filename[:-1]
		print(filename)
		img = cv2.imread(filename)
		print(type(img))
		height, width, layers = img.shape
		size = (width,height)
		img_list.append(img)

	out = cv2.VideoWriter(out_name+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 6, size)
	for i in range(len(img_list)):
		out.write(img_list[i])
		print('frame %d witten!'%(i))
	out.release()
	return None

if __name__=='__main__':

	file_name_pattern=os.path.abspath(sys.argv[1])
	out_name=sys.argv[2]
	mk_movie(file_name_pattern,out_name)

	
