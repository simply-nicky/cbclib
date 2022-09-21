'''
topo_compute generates the topogram images in batch
according to a .txt list of of reflections.
'''

## set the libaries and dependencies
import sys,os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
import pickle as pk
import glob2

def single_topogram(K_map_dir,hkl):
    img_dim=(4362,4148)
    topogram_img = np.zeros(img_dim)
    hkl = np.array(hkl).astype(np.int)
    streak_file = K_map_dir+f'K_streak_sofar.txt'
    streak_arry = np.genfromtxt(streak_file)
    ind = (streak_arry[:,4:7]==hkl).all(axis=1).nonzero()[0] ##check the robustness when hkl is out of range.
    frame_arry = streak_arry[ind,0].astype(np.int)
    for frame in frame_arry:
        K_map_file = K_map_dir+f'K_map_scan207_file5_fr{frame:d}.txt'
        if os.path.exists(K_map_file)==False:
            continue
        print(frame)
        K_map_arry = np.genfromtxt(K_map_file)
        ind1 = (K_map_arry[:,4:7]==hkl).all(axis=1).nonzero()[0]
        K_map_arry = K_map_arry[ind1]
        for m in range(K_map_arry.shape[0]):
            x = int(K_map_arry[m,1])
            y = int(K_map_arry[m,2])
            pix_sig = K_map_arry[m,3]
            topogram_img[y,x] += pix_sig

    return frame_arry,topogram_img

def kin2ind(K_in_arry,xbins=np.arange(-6e8,0,2.5e6),ybins=np.arange(0,6e8,2.5e6)):
    indx = np.digitize(K_in_arry[:,0],xbins) - 1
    indy = np.digitize(K_in_arry[:,1],ybins) - 1
    return indx, indy

def single_topogram_kin(K_map_dir,hkl):
    img_dim=(240,240)
    topogram_img_sum = np.zeros(img_dim)
    counter = np.zeros(img_dim).astype(np.int8)
    hkl = np.array(hkl).astype(np.int)
    streak_file = K_map_dir+f'K_streak_sofar.txt'
    streak_arry = np.genfromtxt(streak_file)
    ind = (streak_arry[:,4:7]==hkl).all(axis=1).nonzero()[0] ##check the robustness when hkl is out of range.
    frame_arry = streak_arry[ind,0].astype(np.int)
    print(hkl,frame_arry)
    for frame in frame_arry:
        K_map_file = K_map_dir+f'K_map_scan207_file5_fr{frame:d}.txt'
        if os.path.exists(K_map_file)==False:
            continue
        print(frame)
        K_map_arry = np.genfromtxt(K_map_file)
        ind1 = (K_map_arry[:,4:7]==hkl).all(axis=1).nonzero()[0]
        K_map_arry = K_map_arry[ind1]
        for m in range(K_map_arry.shape[0]):
#             x = int(K_map_arry[m,1])
#             y = int(K_map_arry[m,2])
            K_in_arry = K_map_arry[m,10:13]
            indx,indy = kin2ind(K_in_arry.reshape(1,3))
            pix_sig = K_map_arry[m,3]
#             topogram_img[y,x] += pix_sig
            topogram_img_sum[indy,indx] += pix_sig
            counter[indy,indx] += 1
    topogram_img_ave = topogram_img_sum/(counter+sys.float_info.epsilon)
    return frame_arry,topogram_img_sum,topogram_img_ave

def multiple_topogram(K_map_dir,start_reflection_ID,end_reflection_ID):
    full_topo_lst_file = K_map_dir+'full_topo_list.txt'
    full_topo_arry = np.genfromtxt(full_topo_lst_file)
    print(full_topo_arry.shape[0])
    if end_reflection_ID>=full_topo_arry.shape[0]:
        sys.exit('check reflection ID')
    for m in range(start_reflection_ID,end_reflection_ID+1):
        hkl = full_topo_arry[m].astype(np.int)
        print(f'computing hkl=({hkl[0]:d},{hkl[1]:d},{hkl[2]:d}):')
        frame_arry,topo_img = single_topogram(K_map_dir,hkl)
        per_v= np.percentile(topo_img,99.999)
        max_v = topo_img.max()
        indd = (topo_img>0).nonzero()
        xc = int(indd[1].mean())
        yc = int(indd[0].mean())

        plt.figure()
        plt.imshow(topo_img,origin='lower')
        plt.xlim(xc-120,xc+120)
        plt.ylim(yc-120,yc+120)
        plt.clim(0,per_v)
        plt.title(f'hkl=({hkl[0]:d},{hkl[1]:d},{hkl[2]:d})'+'\n'+f'I_sum_raw={topo_img.sum():7.2e}')
        plt.colorbar()
        plt.savefig(f'topo_{hkl[0]:d}_{hkl[1]:d}_{hkl[2]:d}.png')
        plt.close('all')
    return

def multiple_topogram_kin(K_map_dir,start_reflection_ID,end_reflection_ID):
    full_topo_lst_file = K_map_dir+'full_topo_list.txt'
    full_topo_arry = np.genfromtxt(full_topo_lst_file)
    print(full_topo_arry.shape[0])
    if end_reflection_ID>=full_topo_arry.shape[0]:
        sys.exit('check reflection ID')
    for m in range(start_reflection_ID,end_reflection_ID+1):
        hkl = full_topo_arry[m].astype(np.int)
        print(f'computing hkl=({hkl[0]:d},{hkl[1]:d},{hkl[2]:d}):')
        frame_arry,topo_img_sum,topo_img_ave = single_topogram_kin(K_map_dir,hkl)
        per_v= np.percentile(topo_img_ave,99)
        max_v = topo_img_ave.max()


        plt.figure()
        plt.imshow(topo_img_ave,origin='lower')
        plt.clim(0,per_v)
        plt.title(f'hkl=({hkl[0]:d},{hkl[1]:d},{hkl[2]:d})'+'\n'+f'I_ave_img_sum={topo_img_ave.sum():7.2e}')
        plt.colorbar()
        plt.savefig(f'topo_kin_ave_{hkl[0]:d}_{hkl[1]:d}_{hkl[2]:d}.png')
        plt.close('all')
    return
if __name__=='__main__':
    K_map_dir = sys.argv[1]
    start_reflection_ID = int(sys.argv[2])
    end_reflection_ID = int(sys.argv[3])
    #multiple_topogram(K_map_dir,start_reflection_ID,end_reflection_ID)
    multiple_topogram_kin(K_map_dir,start_reflection_ID,end_reflection_ID)
