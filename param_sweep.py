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

def energy(u,Lx,Ly,Mx,My):
    dx = Lx/Mx
    dy = Ly/My

    return 0.5*np.sum(np.abs(u)**2,axis=(1,2))*dx*dy/(Lx*Ly)

def out(Lx,Ly,delta):
    fname = 'param_sweep/dns_characterizations.txt'
    #with open(fname,'a') as f:
    #    f.write(f"Running Lx = {Lx:.2f}, Ly = {Ly:.2f}, delta = {delta:.5f}")
    print(f"Running Lx = {Lx:.2f}, Ly = {Ly:.2f}, delta = {delta:.5f}")

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    #Lx = 40
    #Ly = 40
    Lx = np.linspace(5,50,100)
    Ly = np.linspace(5,50,100)
    Mx = 64
    My = 64

    delta = np.linspace(0.001,2,100)
    stop_sim_time = 2000
    timestepper = d3.RK222
    timestep = 0.1
    init_t = 0.1
    dtype = np.float64
    dealias = 3/2
    E_vals = np.zeros((len(delta),len(Lx)))

    for i,d in zip(range(len(delta)),delta):
        for j,l in zip(range(len(Lx)),Lx):
            coords = d3.CartesianCoordinates('X','Y')
            dist = d3.Distributor(coords,dtype=dtype) 
            xbasis = d3.Fourier(coords['X'],size=Mx,bounds=(-l/2,l/2),dealias=dealias,dtype=dtype)
            ybasis = d3.Fourier(coords['Y'],size=My,bounds=(-l/2,l/2),dealias=dealias,dtype=dtype)
            H = dist.Field(name='H',bases=(xbasis,ybasis))

            X,Y = dist.local_grids(xbasis,ybasis)

            dX = lambda A: d3.Differentiate(A,coords['X'])
            dY = lambda A: d3.Differentiate(A,coords['Y'])

            problem = d3.IVP([H],namespace=locals())

            problem.add_equation("dt(H)+dX(dX(H))+d*(dX(dX(dX(H)))+dX(dY(dY(H))))+dX(dX(dX(dX(H))))+2*dX(dX(dY(dY(H))))+dY(dY(dY(dY(H))))=-H*dX(H)")

            #init_H = np.zeros((Mx,My))
            with h5py.File("IC_L_d_plot.h5",'r') as f:
                init_H = np.array(f.get("ic"))

            H['g'] = init_H


    #init_H[1,1]=0.1
    #for i in range(Mx):
    #    for j in range(My):
    #        init_H[i,j] = np.random.randint(-5,5)*1e-3
    #H['g'] = np.real(np.fft.ifft(init_H))

    #with h5py.File("IC_L_d_plot.h5",'w') as f:
    #    f.create_dataset("ic",data=init_H)


            solver = problem.build_solver(timestepper)
            solver.stop_sim_time=stop_sim_time

            #data = solver.evaluator.add_file_handler('films_Mx_'+str(Mx)+'_My_'+str(My)+'Lx_'+str(int(Lx))+'Ly_'+str(int(Ly))+'d_002',iter=1,max_writes=20*stop_sim_time)
            #data = solver.evaluator.add_file_handler('films_Mx_'+str(Mx)+'_My_'+str(My)+'Lx_14_9Ly_14_9d_03',iter=10,max_writes=20*stop_sim_time)
            #data.add_task(H,name='H')
            H_list = [np.copy(H['g'])]
            t_list = [solver.sim_time]
            out(l,l,d)
            while solver.proceed:
                solver.step(timestep)
                #if solver.iteration % 100 == 0:
                #    logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration,solver.sim_time,timestep))
                #if solver.iteration % 1 == 0:
                H.change_scales(1)
                H_list.append(np.copy(H['g']))
                t_list.append(solver.sim_time)

            H = np.array(H_list)
            
            E_vals[i,j] = np.mean(energy(H,l,l,Mx,Mx)) # mean energy for each soln
            x = np.linspace(-l/2,l/2,Mx)
            y = np.linspace(-l/2,l/2,My)
            snap_period = 200 # in ndts
            snap_freq = 15 # num samples to take
            T_snapshots = [i*snap_period//snap_freq for i in range(snap_freq)]
            
            for k in range(len(T_snapshots)):
                fig,ax = plt.subplots()
                true_cmap = ax.pcolormesh(x,y,H[T_snapshots[k]],cmap='twilight')
                ax.set_ylabel("y")
                ax.set_aspect("equal","box")
                fig.colorbar(true_cmap,ax=ax)

                plt.savefig("param_sweep/snaps/Snapshot_{:.2f}_L_{:.2f}_d_{:.5f}.png".format(T_snapshots[k]*timestep,l,d),bbox_inches='tight',pad_inches=0.0)
                plt.close()

    with h5py.File("param_sweep/param_energy.h5",'w') as f:
        f.create_dataset("E",data=E_vals)
        f.create_dataset("L",data=Lx)
        f.create_dataset("d",data=delta)

    plt.figure()
    plt.pcolormesh(delta,Lx,E_vals,cmap='inferno')
    plt.xlabel(r'$\delta$')
    plt.ylabel("L")
    plt.show()