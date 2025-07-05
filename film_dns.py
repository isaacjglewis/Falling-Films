import dedalus.public as d3
import numpy as np 

import logging

import matplotlib.pyplot as plt 
import matplotlib.ticker as tck
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.colors as colors

from sklearn.utils.extmath import randomized_svd

import h5py

from numpy.fft import fft
from numpy.fft import ifft
from numpy import pi

def out(Lx,Ly,delta,E_total):
    fname = 'dns_characterizations.txt'
    with open(fname,'a') as f:
        f.write("Lx ="+str(Lx)+", Ly ="+str(Ly)+", delta="+str(delta)+", Total Energy ="+str(E_total)+'\n')

def EnergyCalc(ut,N,reduced):
    # print(N)
    utfft=fft(ut,axis=0)
    # print(utfft.shape)

    dudx=np.zeros((N,reduced))
    dudx2=np.zeros((N,reduced))
    # term1=np.zeros((N,reduced))
    # term2=np.zeros((N,reduced))   
    # term3=np.zeros((N,reduced))   
    L=22
    kn = 2*pi/L*np.append(np.linspace(0,int(N/2-1),int(N/2)),np.linspace(int(-N/2),-1,int(N/2)))
    # kn = np.linspace(0,N/2-1,N/2)
    for m in range(reduced-1):
        #First derivative
        # dudx[:,m]=ifft(utfft[m,:]*(1j*kn))
       # print("energy shapes:")
       # print(utfft[:,m].shape)
       # print(kn.shape)
        dudx[:,m]=ifft(utfft[:,m]*(1j*kn))
        # temp = ifft(utfft[m,:]*(1j*kn))
        # print(temp.shape)
        # exit()

        #Second derivative
        dudx2[:,m]=ifft(utfft[:,m]*(1j*kn)**2)
        # dudx2[:,m]=ifft(utfft[m,:]*(1j*kn)**2)

        # #Term1
        # term1[:,m]=dudx[:,m]**2
        # #Term2
        # fft2=fft(term1[:,m])
        # fft3=fft(utplot[:,m]**3)
        # term2[:,m]=2*ifft(fft2*(1j*kn/pi)**2)-1/3*ifft(fft3*(1j*kn/pi))
        # #Term3
        # term3[:,m]=-dudx2[:,m]**2

    return dudx,dudx2

def e_t(E,dt):
    dE = np.diff(E)
    return dE/dt

def energy(u,Lx,Ly,Mx,My):
    dx = Lx/Mx
    dy = Ly/My

    return 0.5*np.sum(np.abs(u)**2,axis=(1,2))*dx*dy/(Lx*Ly)

def steporbit(ndts_,x,u,solver,dt):
    if ndts_ != 1:
        dt = x[0]/ndts_

    a = x[1:] # Use only spatial terms in timestepping

    for i in range(ndts_): # Predictor-corrector method 
        fa = kse_step(x,2,u,solver,dt)
        fa1 = kse_step(a1,2,u,solver,dt)
        a = a+0.5*dt*(fa+fa1)

    return a

def kse_step(x0,ndts,u,solver,dt):
    u['g'] = x0
    u_list = [np.copy(u['g'])]
    solver.sim_time = 0
    solver.iteration = 0
    solver.stop_sim_time = ndts*dt
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 100 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration,solver.sim_time,timestep))
        if solver.iteration % 25 == 0:
            u.change_scales(1)
            u_list.append(np.copy(H['g']))
            t_list.append(solver.sim_time)

    print(u_list)
    return np.squeeze(np.array(u_list))

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    F = 3.5
    W = 1.2
    R = 20
    Lx = 40
    Ly = 40
    Mx = 64
    My = 64

    a = 5/(2*np.abs(F**2-5/2))
    #d = 2*np.sqrt(5)*F/(R*np.sqrt(np.abs(F**2-5/2))*np.sqrt(2*W))
    d = 0.002

    k_1 = np.linspace(0,1.4,1000)
    k_2 = np.linspace(0,0.8,1000)

    A = np.zeros((1000,1000))

    for i in range(1000):
        for j in range(1000):
            A[i,j]=k_1[i]**2+(k_1[i])*(k_1[i]**2+k_2[j]**2)*1j-(k_1[i]**4+k_1[i]**2*k_2[j]**2+k_2[j]**4)

    A = np.real(A)

    plt.figure()
    plt.contourf(k_2,k_1,A,levels=15,cmap='viridis')
    plt.xlabel('$k_1$')
    plt.ylabel('$k_2$')
    plt.colorbar()
    #plt.show()


    dealias = 3/2

    stop_sim_time = 20000
    timestepper = d3.RK222
    timestep = 0.1
    init_t = 0.1
    dtype = np.float64

    coords = d3.CartesianCoordinates('X','Y')
    dist = d3.Distributor(coords,dtype=dtype) 
    xbasis = d3.Fourier(coords['X'],size=Mx,bounds=(-Lx/2,Lx/2),dealias=dealias,dtype=dtype)
    ybasis = d3.Fourier(coords['Y'],size=My,bounds=(-Ly/2,Ly/2),dealias=dealias,dtype=dtype)
    H = dist.Field(name='H',bases=(xbasis,ybasis))

    X,Y = dist.local_grids(xbasis,ybasis)

    dX = lambda A: d3.Differentiate(A,coords['X'])
    dY = lambda A: d3.Differentiate(A,coords['Y'])

    problem = d3.IVP([H],namespace=locals())

    problem.add_equation("dt(H)+dX(dX(H))+d*(dX(dX(dX(H)))+dX(dY(dY(H))))+dX(dX(dX(dX(H))))+2*dX(dX(dY(dY(H))))+dY(dY(dY(dY(H))))=-H*dX(H)")

    init_H = np.zeros((Mx,My))



    init_H[1,1]=0.1
    #for i in range(Mx):
    #    for j in range(My):
    #        init_H[i,j] = np.random.randint(-5,5)*1e-3
    #H['g'] = np.real(np.fft.ifft(init_H))

    #with h5py.File("IC_L_d_plot.h5",'w') as f:
    #    f.create_dataset("ic",data=init_H)
    
    #with h5py.File("IC_L_d_plot.h5",'r') as f:
    #    init_H = np.array(f.get("ic"))
    H['g'] = init_H

    solver = problem.build_solver(timestepper)
    solver.stop_sim_time=stop_sim_time

    data = solver.evaluator.add_file_handler('films_Mx_'+str(Mx)+'_My_'+str(My)+'Lx_'+str(int(Lx))+'Ly_'+str(int(Ly))+'d_002',iter=1,max_writes=20*stop_sim_time)
    #data = solver.evaluator.add_file_handler('films_Mx_'+str(Mx)+'_My_'+str(My)+'Lx_14_9Ly_14_9d_03',iter=10,max_writes=20*stop_sim_time)
    data.add_task(H,name='H')
    H_list = [np.copy(H['g'])]
    t_list = [solver.sim_time]
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 100 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration,solver.sim_time,timestep))
        #if solver.iteration % 1 == 0:
        H.change_scales(1)
        H_list.append(np.copy(H['g']))
        t_list.append(solver.sim_time)

    H = np.array(H_list)

    x = np.linspace(-Lx/2,Lx/2,Mx)
    y = np.linspace(-Ly/2,Ly/2,My)

    snap_period = 2000
    snap_freq = 100
    T_snapshots = [i*snap_period//snap_freq for i in range(snap_freq)]

    #for i in range(len(T_snapshots)):
    #    fig,ax = plt.subplots()
    #    true_cmap = ax.pcolormesh(x,y,H[T_snapshots[i]],cmap='twilight')
    #    ax.set_ylabel("y")
    #    ax.set_aspect("equal","box")
    #    fig.colorbar(true_cmap,ax=ax)

    #    plt.savefig("snaps/Snapshot_{:02f}_L_{:02f}_d_{:03f}.png".format(T_snapshots[i]*timestep,Lx,d),bbox_inches='tight',pad_inches=0.0)
    #    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.azim=-45
    ax.elev=60
    ax.set_box_aspect((np.ptp(16*X),np.ptp(16*X),300))
    X,Y = np.meshgrid(X.ravel(),Y.ravel())
    ax.plot_surface(X,Y,H[-1],cmap='twilight',rstride=1,cstride=1,linewidth=0,antialiased=False)
    ax.set_xlabel('\nx')
    ax.xaxis.set_major_locator(tck.MultipleLocator(20))
    ax.yaxis.set_major_locator(tck.MultipleLocator(20))
    ax.set_zticks([-2,0,2])
    ax.set_ylabel('\ny')
    ax.set_zlabel('\nH')
    ax.set_title("Films "+r"$L_x$="+str(Lx)+", "+r"$L_y=$"+str(Ly)+", T="f"{stop_sim_time:0.2f}")
    #fig.subplots_adjust(top=1,bottom=0,left=0,right=1,wspace=0)
    fig.set_figwidth(7)
    fig.set_figheight(7)
    plt.savefig("u_Lx_"+str(Lx)+"_Ly_"+str(Ly)+"Mx_"+str(Mx)+"_My_"+str(My)+"T_"+str(stop_sim_time)+"_d_"+str(d)+".png")
    print(H.shape)
    #E = energy(H,Lx,Ly,Mx,My)
    E = np.linalg.norm(H,axis=(1,2))
    fig,(ax1,ax2) = plt.subplots(1,2,sharey=False)
    ax1.plot(t_list,E,'k-')
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$E(t)$")
    dEdt = e_t(E,timestep)
    ax2.plot(E[:-1],dEdt,'k-')
    ax2.set_xlabel(r"$E(t)$")
    ax2.set_ylabel(r"$dE/dt$")
    plt.tight_layout()
    plt.savefig("energy_phase_Lx_22_Ly_22_d_002_end.png")
    #plt.show()

    #autocorr = np.mean(H[0]*H,axis=(1,2))/np.mean(H[0]**2,axis=(0,1))

   # plt.figure()
    #plt.plot(t_list,autocorr,'k-')
    #plt.savefig("autocorrelation_Lx__Ly_22_d_002.png")

    #total_E = np.mean(E) # time average of energy - for plotting
    #print("Total energy:",total_E)
    E_total = np.mean(E)
    out(Lx,Ly,d,E_total)

    #H = np.reshape(H,(H.shape[0],H.shape[1]*H.shape[2]))

    ux,uxx = EnergyCalc(H[:,:,My//2].T,Mx,H.shape[0])
    uy, uyy = EnergyCalc(H[:,Mx//2,:].T,My,H.shape[0])
    ux = ux.flatten()
    uxx = uxx.flatten()
    uy = uy.flatten()
    uyy = uyy.flatten()
    plt.figure()
    h, xedges, yedges, image = plt.hist2d(ux,uxx,bins=(100,100),density=True,norm=colors.LogNorm(vmin=1e-8, vmax=1.0),range=[[-3,2],[-3,3]])

    plt.figure()
    h2, xedges2, yedges2, image2 = plt.hist2d(uy,uyy,bins=(100,100),density=True,norm=colors.LogNorm(vmin=1e-8, vmax=1.0),range=[[-3,2],[-3,3]])

    plt.show()




    x = [Lx*(i/Mx) for i in range(Mx)]
    y = [Ly*(i/My) for i in range(My)]

    fig,ax = plt.subplots()
    cax = ax.pcolormesh(x,y,H[0,:,:],vmin=-2,vmax=2,cmap='twilight')
    fig.colorbar(cax)

    def animate(i):
        cax.set_array(H[i,:,:].flatten())
    anim = animation.FuncAnimation(fig,animate,interval=1,frames=int(stop_sim_time/timestep))
    anim.save('H_d_03_TW.gif')



    

