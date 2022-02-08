import sys,os
sys.path.append(os.path.realpath(__file__))
import h5py
import numpy as np
import scipy
import CCB_read
import CCB_pred
import CCB_pat_sim
import Xtal_calc_util as xu
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plot

import gen_match_figs as gm

#a=15.4029218
#b=21.86892773
#c=25
#Alpha=90
#Beta=90
#Gamma=90
#OR_mat=np.array([[-2.05112078e+08,-3.89499652e+08,-1.22739594e+08],
#[-2.21095123e+08,1.95462074e+08,-3.25490173e+08],
#[ 5.57086534e+08,-8.72801443e+07,-1.74338424e+08]])
OR_mat = np.genfromtxt('../../OR.txt')
R_mat=OR_mat/1.0


###################
# for expanding lattice constants
expanding_const = 1
OR_mat = OR_mat/expanding_const
##################


E_ph=17.4 #in keV
wave_len=12.40/E_ph #in Angstrom
wave_len=1e-10*wave_len # convert to m
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
#k_cen=1/wave_len*np.array([-0.03115,-0.02308,0.999248]).reshape(3,1)
#k_cen = np.genfromtxt('/home/lichufen/CCB_ind/k_cen.txt')
#k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),1/wave_len*np.ones((1500,1))))
k0 = 1/wave_len
k_cen = np.hstack((-3e8*np.ones((1500,1)),2.2e8*np.ones((1500,1)),np.sqrt(k0**2-(3e8)**2-(2.2e8)**2)*np.ones((1500,1))))
k_cen = k_cen/(np.linalg.norm(k_cen,axis=1).reshape(-1,1))*1/wave_len


def rot_mat_yaxis(theta_deg):
    Rot_mat=np.zeros((3,3))
    Rot_mat[0,0]=np.cos(np.deg2rad(theta_deg))
    Rot_mat[1,1]=1
    Rot_mat[2,0]=-np.sin(np.deg2rad(theta_deg))
    Rot_mat[0,2]=np.sin(np.deg2rad(theta_deg))
    Rot_mat[2,2]=np.cos(np.deg2rad(theta_deg))
    return Rot_mat
def rot_mat_xaxis(theta_deg):
    Rot_mat=np.zeros((3,3))
    Rot_mat[1,1]=np.cos(np.deg2rad(theta_deg))
    Rot_mat[0,0]=1
    Rot_mat[1,2]=-np.sin(np.deg2rad(theta_deg))
    Rot_mat[2,1]=np.sin(np.deg2rad(theta_deg))
    Rot_mat[2,2]=np.cos(np.deg2rad(theta_deg))
    return Rot_mat

def rot_mat_zaxis(theta_deg):
    Rot_mat=np.zeros((3,3))
    Rot_mat[0,0]=np.cos(np.deg2rad(theta_deg))
    Rot_mat[2,2]=1
    Rot_mat[0,1]=-np.sin(np.deg2rad(theta_deg))
    Rot_mat[1,0]=np.sin(np.deg2rad(theta_deg))
    Rot_mat[1,1]=np.cos(np.deg2rad(theta_deg))
    return Rot_mat

def solve_k(q_vec,wave_len,theta_0,plot_flag=False):
    k0=1/wave_len
    q_len=np.linalg.norm(q_vec.reshape(3,1))

    func= lambda dir_ang: k0-\
    np.linalg.norm(q_vec+np.array([k0*np.sin(dir_ang),0,k0*np.cos(dir_ang)]).reshape(3,1))
    ##plot it
    dir_ang=np.linspace(-180,180,361)
    if plot_flag==True:
        plt.plot(dir_ang,func(np.deg2rad(dir_ang)))
        plt.xlabel('direction angle/deg')
        plt.ylabel('function')
        plt.grid()
        plt.show()
    ###solve it
    dir_ang_guess=np.deg2rad(theta_0)
    dir_ang_solution=scipy.optimize.fsolve(func,dir_ang_guess)
    k_in=np.array([k0*np.sin(dir_ang_solution),0,k0*np.cos(dir_ang_solution)]).reshape(3,1)
    k_out=q_vec.reshape(3,1)+np.array([k0*np.sin(dir_ang_solution),0,k0*np.cos(dir_ang_solution)]).reshape(3,1)
    return k_in,k_out,np.rad2deg(dir_ang_solution)

def get_HKL(OR_mat,Q_arry,frac_offset):

    frac_offset=frac_offset.reshape(-1,)
    num_q=Q_arry.shape[0]
    HKL_frac=np.zeros_like(Q_arry)
    HKL_int=np.zeros_like(Q_arry)
    Q_int=np.zeros_like(Q_arry)
    Q_resid=np.zeros_like(Q_arry)

    for num in range(num_q):
        q=Q_arry[num,:].reshape(3,1)
        hkl_frac=(np.linalg.inv(OR_mat))@q

        HKL_int[num,:]=np.rint((hkl_frac-frac_offset.reshape(3,1)).reshape(-1,))
        HKL_frac[num,:]=hkl_frac.reshape(-1,)
        Q_int[num,:]=(OR_mat@(HKL_int[num,:].reshape(3,1))).reshape(-1,)

    Q_resid=Q_arry-(OR_mat@(frac_offset.reshape(3,1))).reshape(-1,)-Q_int

    return HKL_frac,HKL_int,Q_int,Q_resid

def exctn_error(OR_mat,Q_arry,Q_int,frac_offset):
    num_q=Q_arry.shape[0]

    if Q_arry.shape[0]!=Q_int.shape[0]:
        sys.exit('input Q vector arrays inconsistent!')
    K_in_cen=np.zeros((num_q,3))
    K_out=np.zeros((num_q,3))
    Ang_deg=np.zeros((num_q,1))

    Delta_k=np.zeros((num_q,3))
    Dist=np.zeros((num_q,1))
    Dist_1=np.zeros((num_q,1))
    valid_ind=np.ones((num_q,))
    for num in range(num_q):
        q_vec=Q_arry[num,:].reshape(3,1)
        theta_0=100/num_q*(num+1)
        k_in_cen,k_out,ang_deg=solve_k(q_vec,wave_len,np.deg2rad(theta_0),plot_flag=False)
        K_in_cen[num,:]=k_in_cen.reshape(3,)
        K_out[num,:]=k_out.reshape(3,)
        Ang_deg[num,:]=ang_deg
        if (ang_deg<theta_0-10) or (ang_deg>theta_0+10):
            valid_ind[num]=0
            continue
        #k_out=k_in_cen.reshape(3,1)+Q_arry[num,:].reshape(3,1)
        #k_out=(1/wave_len)*(k_out/np.linalg.norm(k_out))#normalize the kout so that it is elastic
        #print('%7.3e'%(np.linalg.norm(Q_int[num,:])))
        #print('%7.3e'%(np.linalg.norm(k_out-Q_int[num,:].reshape(3,1))),'-','%7.3e'%(np.linalg.norm(k_out)))
        dist=np.linalg.norm(k_out-Q_int[num,:].reshape(3,1))-np.linalg.norm(k_out.reshape(3,1))


        delta_k=Q_int[num,:].reshape(3,1)-(Q_arry[num,:].reshape(3,1)-(OR_mat@(frac_offset.reshape(3,1))))


        delta_k_par=(k_in_cen/np.linalg.norm(k_in_cen))*np.dot(k_in_cen.T/np.linalg.norm(k_in_cen),delta_k)  #the component parallel to k_in_cen
        delta_k_per=delta_k-delta_k_par   #the component perpendicular to k_in_cen
        Dist[num,:]=dist
        Dist_1[num,:]=np.linalg.norm(delta_k_par)
        Delta_k[num,:]=(np.matmul(rot_mat_yaxis(-ang_deg),delta_k_per)).reshape(3,)
    return K_in_cen, K_out, Ang_deg ,Delta_k, Dist, Dist_1,valid_ind


def exctn_error_nr(k_cen,OR_mat,Q_arry,Q_int,frac_offset,E_ph):
    wave_len=12.40/E_ph
    #k_cen=np.array([0,0,1e10/wave_len]).reshape(3,1)
    #k_cen=np.array([0,0,1e10/wave_len]).reshape(3,1)
    k_cen=k_cen.reshape(3,1)
    num_q=Q_arry.shape[0]

    if Q_arry.shape[0]!=Q_int.shape[0]:
        sys.exit('input Q vector arrays inconsistent!')

    Ang_deg=np.zeros((num_q,1))

    Delta_k=np.zeros((num_q,3))
    Dist=np.zeros((num_q,1))
    Dist_1=np.zeros((num_q,1))

    for num in range(num_q):
        q_vec=Q_arry[num,:].reshape(3,1)
        k_out=k_cen+q_vec
        dist=np.linalg.norm(k_out-Q_int[num,:].reshape(3,1))-np.linalg.norm(k_out.reshape(3,1))
        delta_k=Q_int[num,:].reshape(3,1)-(Q_arry[num,:].reshape(3,1)-(OR_mat@(frac_offset.reshape(3,1))))
        delta_k_par=(k_cen/np.linalg.norm(k_cen))*np.dot(k_cen.T/np.linalg.norm(k_cen),delta_k)  #the component parallel to k_in_cen
        delta_k_per=delta_k-delta_k_par   #the component perpendicular to k_in_cen
        Dist[num,:]=np.abs(dist)
        Dist_1[num,:]=np.linalg.norm(delta_k_par)
        Delta_k[num,:]=(np.matmul(rot_mat_yaxis(0),delta_k_per)).reshape(3,)
    return  Delta_k, Dist, Dist_1

###################################
#calculate K_in_cen, K_out, Ang_deg, Delta_k, Dist, valid_ind
def get_HKL8(OR_mat,Q_arry,frac_offset):
    frac_offset=frac_offset.reshape(-1,)
    num_q=Q_arry.shape[0]
    HKL_frac=np.zeros_like(Q_arry)
    HKL_int=np.zeros((num_q,3,8))
    Q_int=np.zeros((num_q,3,8))
    Q_resid=np.zeros((num_q,3,8))

    for num in range(num_q):
        q=Q_arry[num,:].reshape(3,1)
        hkl_frac=(np.linalg.inv(OR_mat))@q

        #HKL_int[num,:]=np.rint((hkl_frac-frac_offset.reshape(3,1)).reshape(-1,))
        hkl_fl=np.floor((hkl_frac-frac_offset.reshape(3,1)).reshape(-1,)).astype(np.int)
        hkl_ce=np.ceil((hkl_frac-frac_offset.reshape(3,1)).reshape(-1,)).astype(np.int)
        hkl1=np.array([hkl_fl[0],hkl_fl[1],hkl_fl[2]])
        hkl2=np.array([hkl_fl[0],hkl_fl[1],hkl_ce[2]])
        hkl3=np.array([hkl_fl[0],hkl_ce[1],hkl_fl[2]])
        hkl4=np.array([hkl_ce[0],hkl_fl[1],hkl_fl[2]])
        hkl5=np.array([hkl_ce[0],hkl_ce[1],hkl_fl[2]])
        hkl6=np.array([hkl_ce[0],hkl_fl[1],hkl_ce[2]])
        hkl7=np.array([hkl_fl[0],hkl_ce[1],hkl_ce[2]])
        hkl8=np.array([hkl_ce[0],hkl_ce[1],hkl_ce[2]])

        HKL_int[num,:,:]=np.stack((hkl1,hkl2,hkl3,hkl4,hkl5,hkl6,hkl7,hkl8),axis=1)

        HKL_frac[num,:]=hkl_frac.reshape(-1,)
        Q_int[num,:,:]=OR_mat@(HKL_int[num,:,:])
    Q_resid=np.tile(Q_arry.reshape(-1,3,1),(1,1,8))-np.tile((OR_mat@(frac_offset.reshape(3,1))).reshape(1,3,1),(Q_int.shape[0],1,Q_int.shape[2]))-Q_int

    return HKL_frac,HKL_int,Q_int,Q_resid

def exctn_error8(OR_mat,Q_arry,Q_int,frac_offset):
    num_q=Q_arry.shape[0]

    if Q_arry.shape[0]!=Q_int.shape[0]:
        sys.exit('input Q vector arrays inconsistent!')
    K_in_cen=np.zeros((num_q,3))
    K_out=np.zeros((num_q,3))
    Ang_deg=np.zeros((num_q,1))

    Delta_k=np.zeros((num_q,3,8))
    Dist=np.zeros((num_q,8))
    Dist_1=np.zeros((num_q,8))
    valid_ind=np.ones((num_q,))
    for num in range(num_q):
        q_vec=Q_arry[num,:].reshape(3,1)
        theta_0=100/num_q*(num+1)
        k_in_cen,k_out,ang_deg=solve_k(q_vec,wave_len,np.deg2rad(theta_0),plot_flag=False)
        K_in_cen[num,:]=k_in_cen.reshape(3,)
        K_out[num,:]=k_out.reshape(3,)
        Ang_deg[num,:]=ang_deg
        if (ang_deg<theta_0-10) or (ang_deg>theta_0+10):
            valid_ind[num]=0
            continue
        #k_out=k_in_cen.reshape(3,1)+Q_arry[num,:].reshape(3,1)
        #k_out=(1/wave_len)*(k_out/np.linalg.norm(k_out))#normalize the kout so that it is elastic
        #print('%7.3e'%(np.linalg.norm(Q_int[num,:])))
        #print('%7.3e'%(np.linalg.norm(k_out-Q_int[num,:].reshape(3,1))),'-','%7.3e'%(np.linalg.norm(k_out)))


        for m in range(8):
            dist=np.linalg.norm(k_out-Q_int[num,:,m].reshape(3,1))-np.linalg.norm(k_out.reshape(3,1))
            delta_k=Q_int[num,:,m].reshape(3,1)-(Q_arry[num,:].reshape(3,1)-(OR_mat@(frac_offset.reshape(3,1))))
            delta_k_par=(k_in_cen/np.linalg.norm(k_in_cen))*np.dot(k_in_cen.T/np.linalg.norm(k_in_cen),delta_k)  #the component parallel to k_in_cen
            delta_k_per=delta_k-delta_k_par   #the component perpendicular to k_in_cen
            Dist[num,m]=nps.abs(dist)
            Dist_1[num,m]=np.linalg.norm(delta_k_par)
            Delta_k[num,:,m]=(np.matmul(rot_mat_yaxis(-ang_deg),delta_k_per)).reshape(3,)


    return K_in_cen, K_out, Ang_deg ,Delta_k, Dist, Dist_1,valid_ind

def exctn_error8_nr(k_cen,OR_mat,Q_arry,Q_int,frac_offset,E_ph):
    wave_len=1e-10*12.40/E_ph
    #k_cen=np.array([0,0,1e10/wave_len]).reshape(3,1)
    #k_cen=np.array([0,0,1/wave_len]).reshape(3,1)
    k_cen=k_cen.reshape(3,1)
    num_q=Q_arry.shape[0]

    if Q_arry.shape[0]!=Q_int.shape[0]:
        sys.exit('input Q vector arrays inconsistent!')




    Delta_k=np.zeros((num_q,3,8))
    Dist=np.zeros((num_q,8))
    Dist_1=np.zeros((num_q,8))

    for num in range(num_q):
        q_vec=Q_arry[num,:].reshape(3,1)
        k_out=k_cen+q_vec


        for m in range(8):
            dist=np.linalg.norm(k_out-Q_int[num,:,m].reshape(3,1))-np.linalg.norm(k_out.reshape(3,1))
            delta_k=Q_int[num,:,m].reshape(3,1)-(Q_arry[num,:].reshape(3,1)-(OR_mat@(frac_offset.reshape(3,1))))
            delta_k_par=(k_cen/np.linalg.norm(k_cen))*np.dot(k_cen.T/np.linalg.norm(k_cen),delta_k)  #the component parallel to k_in_cen
            delta_k_per=delta_k-delta_k_par   #the component perpendicular to k_in_cen
            Dist[num,m]=np.abs(dist)
            Dist_1[num,m]=np.linalg.norm(delta_k_par)
            Delta_k[num,:,m]=(np.matmul(rot_mat_yaxis(0),delta_k_per)).reshape(3,)


    return Delta_k, Dist, Dist_1


###################################


def _TG_func(x,*argv):
    asx,bsx,csx,asy,bsy,csy,asz,bsz,csz=x
    HKL_int, K_out = argv

    num_r=HKL_int.shape[0]
    if HKL_int.shape[0]!=K_out.shape[0]:
        sys.exit('Please check the dimensions of the input arrays!')
    TG=0
    for num in range(num_r):
        h=HKL_int[num,0]
        k=HKL_int[num,1]
        l=HKL_int[num,2]
        k_outx=K_out[num,0]
        k_outy=K_out[num,1]
        k_outz=K_out[num,2]
        k0=np.sqrt(k_outx**2+k_outy**2+k_outz**2)
        A1=h*asx+k*bsx+l*csx-k_outx
        A2=h*asy+k*bsy+l*csy-k_outy
        A3=h*asz+k*bsz+l*csz-k_outz
        dist=np.sqrt(A1**2+A2**2+A3**2)-k0
        TG=TG+dist**2
    return TG

def _TG_grad(x,*argv):
    asx,bsx,csx,asy,bsy,csy,asz,bsz,csz=x
    HKL_int, K_out = argv
    num_r=HKL_int.shape[0]
    if HKL_int.shape[0]!=K_out.shape[0]:
        sys.exit('Please check the dimensions of the input arrays!')
    grad0=0
    grad1=0
    grad2=0
    grad3=0
    grad4=0
    grad5=0
    grad6=0
    grad7=0
    grad8=0
    for num in range(num_r):
        h=HKL_int[num,0]
        k=HKL_int[num,1]
        l=HKL_int[num,2]
        k_outx=K_out[num,0]
        k_outy=K_out[num,1]
        k_outz=K_out[num,2]
        k0=np.sqrt(k_outx**2+k_outy**2+k_outz**2)
        A1=h*asx+k*bsx+l*csx-k_outx
        A2=h*asy+k*bsy+l*csy-k_outy
        A3=h*asz+k*bsz+l*csz-k_outz
        dist=np.sqrt(A1**2+A2**2+A3**2)-k0
        grad0=grad0+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A1)*h
        grad1=grad1+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A1)*k
        grad2=grad2+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A1)*l
        grad3=grad3+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A2)*h
        grad4=grad4+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A2)*k
        grad5=grad5+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A2)*l
        grad6=grad6+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A3)*h
        grad7=grad7+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A3)*k
        grad8=grad8+dist*(A1**2+A2**2+A3**2)**(-0.5)*(2*A3)*l
    return np.asarray((grad0,grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8))

##################################
#The following implements the T.G. function based on which
#12 parameters are refined using gradient-based algorithm:
#9 elements of the orientation matrix, asx,bsx,csx,asy,bsy,csy,asz,bsz,csz,
#and the 3 elements of the k_out correction term, which is equivalent to the uncertainty
#in the choice of "pupip center", and the wavelength(corresponding to koutz)
##################################
def _TG_func1(x,*argv):
    asx,bsx,csx,asy,bsy,csy,asz,bsz,csz,kosx,kosy,kosz=x
    HKL_int, K_out, Ang_deg = argv

    num_r=HKL_int.shape[0]
    if HKL_int.shape[0]!=K_out.shape[0]:
        sys.exit('Please check the dimensions of the input arrays!')
    TG=0
    Dist_q=np.zeros((num_r,))
    for num in range(num_r):
        h=HKL_int[num,0]
        k=HKL_int[num,1]
        l=HKL_int[num,2]
        k_outx=K_out[num,0]
        k_outy=K_out[num,1]
        k_outz=K_out[num,2]
        k0=np.sqrt(k_outx**2+k_outy**2+k_outz**2)

        theta_deg=Ang_deg[num]

        R00=rot_mat_yaxis(theta_deg)[0,0]
        R01=rot_mat_yaxis(theta_deg)[0,1]
        R02=rot_mat_yaxis(theta_deg)[0,2]
        R10=rot_mat_yaxis(theta_deg)[1,0]
        R11=rot_mat_yaxis(theta_deg)[1,1]
        R12=rot_mat_yaxis(theta_deg)[1,2]
        R20=rot_mat_yaxis(theta_deg)[2,0]
        R21=rot_mat_yaxis(theta_deg)[2,1]
        R22=rot_mat_yaxis(theta_deg)[2,2]


        A1=h*asx+k*bsx+l*csx
        A2=h*asy+k*bsy+l*csy
        A3=h*asz+k*bsz+l*csz

        B1=R00*kosx+R01*kosy+R02*kosz
        B2=R10*kosx+R11*kosy+R12*kosz
        B3=R20*kosx+R21*kosy+R22*kosz

        C1=A1+B1-k_outx
        C2=A2+B2-k_outy
        C3=A3+B3-k_outz
        #print(C1,C2,C3,kosz)
        dist=np.sqrt(C1**2+C2**2+C3**2)-k0+kosz
        #print('%7.3e'%dist)
        Dist_q[num]=dist
        TG=TG+dist**2
    return TG

def _TG_func2(x,*argv):
    asx,bsx,csx,asy,bsy,csy,asz,bsz,csz,kosx,kosy,kosz=x
    HKL_int, K_out, Ang_deg = argv

    num_r=HKL_int.shape[0]
    if HKL_int.shape[0]!=K_out.shape[0]:
        sys.exit('Please check the dimensions of the input arrays!')
    TG=0
    Dist_q=np.zeros((num_r,))
    for num in range(num_r):
        h=HKL_int[num,0]
        k=HKL_int[num,1]
        l=HKL_int[num,2]
        k_outx=K_out[num,0]
        k_outy=K_out[num,1]
        k_outz=K_out[num,2]
        k0=np.sqrt(k_outx**2+k_outy**2+k_outz**2)

        theta_deg=Ang_deg[num]

        R00=rot_mat_yaxis(theta_deg)[0,0]
        R01=rot_mat_yaxis(theta_deg)[0,1]
        R02=rot_mat_yaxis(theta_deg)[0,2]
        R10=rot_mat_yaxis(theta_deg)[1,0]
        R11=rot_mat_yaxis(theta_deg)[1,1]
        R12=rot_mat_yaxis(theta_deg)[1,2]
        R20=rot_mat_yaxis(theta_deg)[2,0]
        R21=rot_mat_yaxis(theta_deg)[2,1]
        R22=rot_mat_yaxis(theta_deg)[2,2]


        A1=h*asx+k*bsx+l*csx
        A2=h*asy+k*bsy+l*csy
        A3=h*asz+k*bsz+l*csz

        B1=R00*kosx+R01*kosy+R02*kosz
        B2=R10*kosx+R11*kosy+R12*kosz
        B3=R20*kosx+R21*kosy+R22*kosz

        C1=A1+B1-k_outx
        C2=A2+B2-k_outy
        C3=A3+B3-k_outz
        #print(C1,C2,C3,kosz)
        dist=np.sqrt(C1**2+C2**2+C3**2)-k0+kosz
        #print('%7.3e'%dist)
        Dist_q[num]=dist
        TG=TG+dist**2
    return TG, Dist_q


def _TG_grad1(x,*argv):
    asx,bsx,csx,asy,bsy,csy,asz,bsz,csz,kosx,kosy,kosz=x
    HKL_int, K_out, Ang_deg = argv
    num_r=HKL_int.shape[0]
    if HKL_int.shape[0]!=K_out.shape[0]:
        sys.exit('Please check the dimensions of the input arrays!')
    grad0=0
    grad1=0
    grad2=0
    grad3=0
    grad4=0
    grad5=0
    grad6=0
    grad7=0
    grad8=0
    grad9=0
    grad10=0
    grad11=0
    for num in range(num_r):
        h=HKL_int[num,0]
        k=HKL_int[num,1]
        l=HKL_int[num,2]
        k_outx=K_out[num,0]
        k_outy=K_out[num,1]
        k_outz=K_out[num,2]
        k0=np.sqrt(k_outx**2+k_outy**2+k_outz**2)

        theta_deg=Ang_deg[num]

        R00=rot_mat_yaxis(theta_deg)[0,0]
        R01=rot_mat_yaxis(theta_deg)[0,1]
        R02=rot_mat_yaxis(theta_deg)[0,2]
        R10=rot_mat_yaxis(theta_deg)[1,0]
        R11=rot_mat_yaxis(theta_deg)[1,1]
        R12=rot_mat_yaxis(theta_deg)[1,2]
        R20=rot_mat_yaxis(theta_deg)[2,0]
        R21=rot_mat_yaxis(theta_deg)[2,1]
        R22=rot_mat_yaxis(theta_deg)[2,2]


        A1=h*asx+k*bsx+l*csx
        A2=h*asy+k*bsy+l*csy
        A3=h*asz+k*bsz+l*csz

        B1=R00*kosx+R01*kosy+R02*kosz
        B2=R10*kosx+R11*kosy+R12*kosz
        B3=R20*kosx+R21*kosy+R22*kosz

        C1=A1+B1-k_outx
        C2=A2+B2-k_outy
        C3=A3+B3-k_outz
        dist=np.sqrt(C1**2+C2**2+C3**2)-k0+kosz

        grad0=grad0+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C1)*h
        grad1=grad1+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C1)*k
        grad2=grad2+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C1)*l
        grad3=grad3+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C2)*h
        grad4=grad4+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C2)*k
        grad5=grad5+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C2)*l
        grad6=grad6+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C3)*h
        grad7=grad7+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C3)*k
        grad8=grad8+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C3)*l
        grad9=grad9+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C1*R00+2*C2*R10+2*C3*R20)
        grad10=grad10+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C1*R01+2*C2*R11+2*C3*R21)
        grad11=grad11+dist*(C1**2+C2**2+C3**2)**(-0.5)*(2*C1*R02+2*C2*R12+2*C3*R22)
    return np.asarray((grad0,grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8,grad9,grad10,grad11))

def Rot_mat_gen(theta,phi,alpha):
    ux=np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi))
    uy=np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi))
    uz=np.cos(np.deg2rad(theta))
    #print(ux,uy,uz)
    alpha=np.deg2rad(alpha)
    Rot_mat=np.zeros((3,3))

    Rot_mat[0,0]=np.cos(alpha)+ux**2*(1-np.cos(alpha))
    Rot_mat[0,1]=ux*uy*(1-np.cos(alpha))-uz*np.sin(alpha)
    Rot_mat[0,2]=ux*uz*(1-np.cos(alpha))+uy*np.sin(alpha)
    Rot_mat[1,0]=uy*ux*(1-np.cos(alpha))+uz*np.sin(alpha)
    Rot_mat[1,1]=np.cos(alpha)+uy**2*(1-np.cos(alpha))
    Rot_mat[1,2]=uy*uz*(1-np.cos(alpha))-ux*np.sin(alpha)
    Rot_mat[2,0]=uz*ux*(1-np.cos(alpha))-uy*np.sin(alpha)
    Rot_mat[2,1]=uz*uy*(1-np.cos(alpha))+ux*np.sin(alpha)
    Rot_mat[2,2]=np.cos(alpha)+uz**2*(1-np.cos(alpha))

    return Rot_mat

def _TG_func3(x,frame):
    E_ph=17.4
    wave_len=1e-10*12.40/E_ph
    theta, phi, alpha, amp_fact, kosx, kosy =x

    Rot_mat=Rot_mat_gen(theta,phi,alpha)
    #frame,  = argv

    OR_start=rot_mat_zaxis(0)@rot_mat_xaxis(0)@rot_mat_yaxis(0.5*frame)@OR_mat
    OR=Rot_mat@OR_start
    kout_dir_dict=CCB_read.kout_read('../../k_out.txt')#changed for batch mode
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)

    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    Diff_vector = kout_dict['diff_vector_'+str(frame)] # This is for q,streak constraint.
    #HKL_frac, HKL_int, Q_int, Q_resid = get_HKL(OR,Q_arry,np.array([0,0,0]))
    HKL_frac, HKL_int, Q_int, Q_resid = get_HKL8(OR,Q_arry,np.array([0,0,0]))
    Delta_k, Dist, Dist_1=exctn_error8_nr(k_cen[frame,:],OR,Q_arry,Q_int,np.array([0,0,0]),E_ph)
	
    K_in_arry = K_out.reshape(-1,3,1) - Q_int #the shape of Q_int and HKL_int is (num,3,8)
    

	
    #ind=np.argsort(Dist,axis=1)
    ind=np.argsort(np.linalg.norm(K_in_arry-k_cen[frame,:].reshape(-1,3,1),axis=1),axis=1)
  


    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])



    #K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    K_in_pred,K_out_pred=CCB_pat_sim.kout_pred(OR,k_cen[frame,:],HKL_int)
    valid_value=(K_in_pred[:,0]<15e8)*(K_in_pred[:,0]>-15e8)*(K_in_pred[:,1]<15e8)*(K_in_pred[:,1]>-15e8)
    K_in_pred=K_in_pred[valid_value,:]
    K_out_pred=K_out_pred[valid_value,:]
    K_out=K_out[valid_value,:]
    #print(K_out_pred.shape)
    ###############CHECK THE CODES
    #Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_in_new=K_in_pred-k_cen[frame,:].reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out

    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<10e8
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<10e8
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]

    TG=(np.linalg.norm(Delta_k_out_new,axis=1)**2).sum()
    num_q=Delta_k_out_new.shape[0]
    TG_norm=np.sqrt(TG/num_q)
    return TG_norm

def _TG_func4(x,*argv):
    E_ph=17.4
    wave_len=1e-10*12.40/E_ph
    asx,bsx,csx,asy,bsy,csy,asz,bsz,csz,kosx,kosy,amp_fact=x
    frame, = argv
    OR_mat=np.array([[asx,bsx,csx],[asy,bsy,csy],[asz,bsz,csz]])
    OR=rot_mat_yaxis(-frame)@OR_mat
    kout_dir_dict=CCB_read.kout_read('../k_out.txt')#changed for batch mode.
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)

    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    #HKL_frac, HKL_int, Q_int, Q_resid = get_HKL(OR,Q_arry,np.array([0,0,0]))
    HKL_frac, HKL_int, Q_int, Q_resid = get_HKL8(OR,Q_arry,np.array([0,0,0]))
    Delta_k, Dist, Dist_1=exctn_error8_nr(OR,Q_arry,Q_int,np.array([0,0,0]),E_ph)
    ind=np.argsort(Dist_1,axis=1)

    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist_1=np.array([Dist_1[m,ind[m]] for m in range(Dist_1.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])



    K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out

    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<5e8
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<5e8
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]

    TG=(np.linalg.norm(Delta_k_out_new,axis=1)**2).sum()
    num_q=Delta_k_out_new.shape[0]
    TG_norm=np.sqrt(TG/num_q)
    return TG_norm

def _TG_func5(x,frame):
    E_ph=17
    wave_len=1e-10*12.40/E_ph
    theta, phi, alpha, amp_fact, kosx, kosy, a, b, c, Alpha, Beta, Gamma =x

    Rot_mat=Rot_mat_gen(theta,phi,alpha)
    #frame,  = argv
    lp=np.array([a*1e-10,b*1e-10,c*1e-10,Alpha,Beta,Gamma])
    _,OR_mat=xu.A_gen(lp)
    OR_start=rot_mat_xaxis(0)@rot_mat_yaxis(-frame)@rot_mat_zaxis(11.84)@OR_mat
    OR=Rot_mat@OR_start
    kout_dir_dict=CCB_read.kout_read('/home/lichufen/CCB_ind/k_out.txt')#changed for batch mode
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)

    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    #HKL_frac, HKL_int, Q_int, Q_resid = get_HKL(OR,Q_arry,np.array([0,0,0]))
    HKL_frac, HKL_int, Q_int, Q_resid = get_HKL8(OR,Q_arry,np.array([0,0,0]))
    Delta_k, Dist, Dist_1=exctn_error8_nr(OR,Q_arry,Q_int,np.array([0,0,0]),E_ph)
    ind=np.argsort(Dist,axis=1)

    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])
    K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    valid_value=(K_in_pred[:,0]<6e8)*(K_in_pred[:,0]>-6e8)*(K_in_pred[:,1]<6e8)*(K_in_pred[:,1]>-6e8)
    K_in_pred=K_in_pred[valid_value,:]
    K_out_pred=K_out_pred[valid_value,:]
    K_out=K_out[valid_value,:]
    #print(K_out_pred.shape)
    ###############CHECK THE CODES
    Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out

    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<10e8
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<10e8
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]

    TG=(np.linalg.norm(Delta_k_out_new,axis=1)**2).sum()
    num_q=Delta_k_out_new.shape[0]
    TG_norm=np.sqrt(TG/num_q)
    return TG_norm

def _TG_func6(x,frame,x0_GA):
    E_ph=17
    wave_len=1e-10*12.40/E_ph
    asx, asy, asz, bsx, bsy, bsz, csx, csy, csz, amp_fact, kosx, kosy = x

    theta, phi, alpha = x0_GA[0:3]
    amp_fact_0,kosx_0,kosy_0 = x0_GA[3:6]

    Rot_mat=Rot_mat_gen(theta,phi,alpha)
    OR_start=rot_mat_zaxis(0)@rot_mat_xaxis(0)@rot_mat_yaxis(-frame)@OR_mat
    OR0=Rot_mat@OR_start

    OR=np.array([[asx,bsx,csx],[asy,bsy,csy],[asz,bsz,csz]])*1e8
    kosx=kosx*1e-2
    kosy=kosy*1e-2

    kout_dir_dict=CCB_read.kout_read('/home/lichufen/CCB_ind/k_out.txt')#changed for batch mode
    #kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact_0,kosx_0,kosy_0)
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)

    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    HKL_frac, HKL_int, Q_int, Q_resid = get_HKL8(OR,Q_arry,np.array([0,0,0]))
    #HKL_frac, HKL_int, Q_int, Q_resid = get_HKL8(OR0,Q_arry,np.array([0,0,0])) # This is to fix the HKL which is determined from OR0.
    Delta_k, Dist, Dist_1=exctn_error8_nr(OR,Q_arry,Q_int,np.array([0,0,0]),E_ph)
    #Delta_k, Dist, Dist_1=exctn_error8_nr(OR0,Q_arry,Q_int,np.array([0,0,0]),E_ph)  # this is for fixed HKL.
    ind=np.argsort(Dist,axis=1)
    #print(OR0.reshape(-1,))
    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])
    #print(HKL_int.shape)
    #print(HKL_int[:10,:])

    kout_dir_dict=CCB_read.kout_read('/home/lichufen/CCB_ind/k_out.txt')#changed for batch mode
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)
    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]

    K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    #print(K_out_pred.shape)
    #print(K_out.shape)
    #print(OR.reshape(-1,))
    valid_value=(K_in_pred[:,0]<6e8)*(K_in_pred[:,0]>-6e8)*(K_in_pred[:,1]<6e8)*(K_in_pred[:,1]>-6e8)
    K_in_pred=K_in_pred[valid_value,:]
    K_out_pred=K_out_pred[valid_value,:]
    K_out=K_out[valid_value,:]
    #print(K_out_pred.shape)
    ###############CHECK THE CODES
    Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out
    #print(K_out_pred.shape)
    #print(K_out.shape)
    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<10e10
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<10e10
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]

    TG=(np.linalg.norm(Delta_k_out_new,axis=1)**2).sum()
    num_q=Delta_k_out_new.shape[0]
    #print(num_q)
    TG_norm=np.sqrt(TG/num_q)
    print(TG_norm)
    return TG_norm


def TG_func7(x,frame,res_file):
    E_ph=17
    wave_len=1e-10*12.40/E_ph

    res_arry = gm.read_res(res_file)
    ind = (res_arry[:,0]==int(frame)).nonzero()[0][0]
    theta = res_arry[ind,1]
    phi = res_arry[ind,2]
    alpha = res_arry[ind,3]
    amp_fact = res_arry[ind,4]
    kosx = res_arry[ind,5]
    kosy = res_arry[ind,6]
    
    ax, ay, az, bx, by, bz, cx, cy, cz =x
    OR_mat1 = np.array([[ax,bx,cx],[ay,by,cy],[az,bz,cz]])
    Rot_mat = Rot_mat_gen(theta,phi,alpha)

    OR_start=rot_mat_zaxis(0)@rot_mat_xaxis(0)@rot_mat_yaxis(-frame)@OR_mat1
    OR=Rot_mat@OR_start
    kout_dir_dict=CCB_read.kout_read('../../k_out.txt')#changed for batch mode
    kout_dir_dict=CCB_read.kout_dir_adj(kout_dir_dict,amp_fact,kosx,kosy)

    kout_dict,q_dict=CCB_read.get_kout_allframe(kout_dir_dict,E_ph)
    Q_arry=q_dict['q_'+str(frame)]
    K_out=kout_dict['kout_'+str(frame)]
    Diff_vector = kout_dict['diff_vector_'+str(frame)] # This is for q,streak constraint.
    #HKL_frac, HKL_int, Q_int, Q_resid = get_HKL(OR,Q_arry,np.array([0,0,0]))
    HKL_frac, HKL_int, Q_int, Q_resid = get_HKL8(OR,Q_arry,np.array([0,0,0]))
    Delta_k, Dist, Dist_1=exctn_error8_nr(k_cen[frame,:],OR,Q_arry,Q_int,np.array([0,0,0]),E_ph)

    K_in_arry = K_out.reshape(-1,3,1) - Q_int #the shape of Q_int and HKL_int is (num,3,8)
    ind=np.argsort(np.linalg.norm(K_in_arry-k_cen[frame,:].reshape(-1,3,1),axis=1),axis=1)


    ind=np.array([ind[m,0] for m in range(ind.shape[0])])
    Dist=np.array([Dist[m,ind[m]] for m in range(Dist.shape[0])])
    HKL_int=np.array([HKL_int[m,:,ind[m]] for m in range(HKL_int.shape[0])])
    Delta_k=np.array([Delta_k[m,:,ind[m]] for m in range(Delta_k.shape[0])])



    #K_in_pred,K_out_pred=CCB_pred.kout_pred(OR,[0,0,1/wave_len],HKL_int)
    K_in_pred,K_out_pred=CCB_pat_sim.kout_pred(OR,k_cen[frame,:],HKL_int)
    valid_value=(K_in_pred[:,0]<15e8)*(K_in_pred[:,0]>-15e8)*(K_in_pred[:,1]<15e8)*(K_in_pred[:,1]>-15e8)
    K_in_pred=K_in_pred[valid_value,:]
    K_out_pred=K_out_pred[valid_value,:]
    K_out=K_out[valid_value,:]
    #print(K_out_pred.shape)
    ###############CHECK THE CODES
    #Delta_k_in_new=K_in_pred-np.array([0,0,1/wave_len]).reshape(1,3)
    Delta_k_in_new=K_in_pred-k_cen[frame,:].reshape(1,3)
    Delta_k_out_new=K_out_pred-K_out

    ind_filter_1=np.linalg.norm(Delta_k_out_new,axis=1)<10e8
    ind_filter_2=np.linalg.norm(Delta_k_in_new,axis=1)<10e8
    ind_filter=ind_filter_1*ind_filter_2
    Delta_k_in_new=Delta_k_in_new[ind_filter,:]
    Delta_k_out_new=Delta_k_out_new[ind_filter,:]
    K_out=K_out[ind_filter,:]
    K_out_pred=K_out_pred[ind_filter,:]

    TG=(np.linalg.norm(Delta_k_out_new,axis=1)**2).sum()
    num_q=Delta_k_out_new.shape[0]
    TG_norm=np.sqrt(TG/num_q)
    return TG_norm, TG, num_q

def _TG_func8(x,frame_list,res_file):
    TG_total = 0
    num_q_total = 0
    for frame in frame_list:
        _,TG,num_q = TG_func7(x,frame,res_file)
        TG_total = TG_total + TG
        num_q_total = num_q_total + num_q
    
    TG_norm = np.sqrt(TG_total/num_q_total)
    #print('num_q_total:',num_q_total)
    
    return TG_norm
