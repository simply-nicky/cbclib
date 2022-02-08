'''
====================================================
**hkl2powder3**: converts hkl list to powder profile
====================================================

put the integrated structure factor list into the
1-D prowder profile form, and plot it to compare with the available
experimental data.
'''
import sys,os
import numpy as np
import matplotlib.pyplot as plt


E_ph=17
wav_len=12.40/E_ph
pix_size=75e-6
cam_len=0.1025

def hkl_read(hkl_file,cam_len=0.1025,pix_size=75e-6,E_ph=17):
    hkl_arry=np.genfromtxt(hkl_file,skip_header=3,skip_footer=2,usecols=(0,1,2,3))
    lat_par=[16.00,21.02,24.54]
    res=1/np.sqrt((hkl_arry[:,0]/lat_par[0])**2+(hkl_arry[:,1]/lat_par[1])**2+(hkl_arry[:,2]/lat_par[2])**2)
    res.shape
    res=res.reshape(-1,1)
    res.shape
    wav_len=12.40/E_ph
    theta=np.arcsin(wav_len/2/res)
    pix_num=cam_len*np.tan(2*theta)/pix_size
    pix_num=pix_num.reshape(-1,1)
    hkl_arry=np.concatenate((hkl_arry,theta,res,pix_num),axis=-1)
    ind_temp=np.argsort(res,axis=0)
    ind_temp=ind_temp[::-1,:]
    hkl_arry=hkl_arry[ind_temp.reshape(-1,),:]
    ### hkl_arry: col0~2:hkl, col3: Intensity, col4: theta, col5:resolution in A, col6:pix_num from center.
    return hkl_arry



def get_FWHM_sq(theta,LX=100):  #LX is the crystal domain/grain size in nm
    FWHM = 0.94*wav_len/(LX*10*np.cos(theta)) # Sherrer's formulaß
    FWHM_sq = FWHM**2
    return FWHM_sq


def gen_Int(hkl_arry,pix_intv):
    pix_arry=np.arange(10,2000,pix_intv)
    theta_arry=np.arctan(pix_arry*pix_size/cam_len)/2
    Int_arry=np.zeros((pix_arry.shape[0],),dtype=np.float64)
    for m in range(hkl_arry.shape[0]):
        hkl=hkl_arry[m,0:3]
        de_gen=int(int(np.round(hkl[0])!=0)+int(np.round(hkl[1])!=0)+int(np.round(hkl[2])!=0))
        #print(hkl,de_gen)
        I=hkl_arry[m,3]
        theta=hkl_arry[m,4]
        FWHM_sq=get_FWHM_sq(theta,LX=500)
        #Int_arry1=(2**de_gen)*I*np.sqrt(4*np.log(2)/np.pi/FWHM_sq)*np.exp(-4*np.log(2)/FWHM_sq*(2*theta_arry-2*theta)**2)*\
        #(np.cos(2*theta_arry))**3/(np.tan(2*theta_arry))*np.exp(-25/(np.cos(2*theta_arry)*41))
        
        #Int_arry1=(2**de_gen)*I*np.sqrt(4*np.log(2)/np.pi/FWHM_sq)*np.exp(-4*np.log(2)/FWHM_sq*(2*theta_arry-2*theta)**2)\
        #*(np.cos(2*theta_arry))**3\
        #/(np.sin(2*theta_arry))
        Int_arry1=(2**de_gen)*I*np.sqrt(4*np.log(2)/np.pi/FWHM_sq)*np.exp(-4*np.log(2)/FWHM_sq*(2*theta_arry-2*theta)**2)

        Int_arry = Int_arry + Int_arry1
        TTtheta_arry = np.rad2deg(2*theta_arry)
        Res_arry = 12.40/E_ph/(2*np.sin(theta_arry))
    return Int_arry,pix_arry,TTtheta_arry,Res_arry



if __name__=='__main__':
    pix_intv=float(sys.argv[1])
    num_input_var=len(sys.argv)-2
    file_name_list=sys.argv[2:]
    file_name_list=[os.path.abspath(file_name) for file_name in file_name_list]



    plt.figure(figsize=(15,8))
    #plt.subplots(int(num_input_var),1)
    for ind,file_name in enumerate(file_name_list):
        vars()['hkl_arry_%d'%(ind)]=hkl_read(file_name)
        vars()['New_arry_%d'%(ind)],pix_arry,TTheta,Res=gen_Int(vars()['hkl_arry_%d'%(ind)],pix_intv)
        #plt_powder(Int_binned,np.arange(100,1800,bin_intv))
        plt.subplot(int(num_input_var),1,ind+1)
        #plt.plot(pix_arry[1:-10], vars()['New_arry_%d'%(ind)][1:-10])
        #plt.plot(TTheta[1:-10], vars()['New_arry_%d'%(ind)][1:-10])
        #ind1 = np.argsort(1/Res)
        
        plt.plot((1/Res)[1:], vars()['New_arry_%d'%(ind)][1:],linewidth=0.8)
        plt.legend(['Powder_'+(os.path.basename(file_name))])
        #plt.xticks(np.arange(0,1000,200))
        #plt.xticks(np.arange(0,40,2))
        plt.xticks(np.arange(0,1.2,0.1))
    plt.xlabel('q (1/A)')
    plt.savefig('powder_profile.png')



    plt.show()
