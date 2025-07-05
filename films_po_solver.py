import numpy as np 
from numpy.fft import fft 
from numpy.fft import ifft
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from math import pi as pi

from sklearn.decomposition import PCA

import matplotlib.animation as animation

import dedalus.public as d3

import logging

import h5py

import argparse

global new_x 
global new_fx
global new_tol
global new_delt
global new_nits
global new_gits
global epsJ
global ndts
global fixT
global p
global dt
global v

global converged

parser = argparse.ArgumentParser("PO solver")
parser.add_argument("--load",nargs=1)

args = parser.parse_args()

filename = "out_newton.txt"

# L = 12 TW, L = 29.3 BTW
logger = logging.getLogger(__name__)

def write_file(filename,text):
	f = open(filename,'w')
	f.write(text)
	f.close()

def e_t(E,dt):
	dE = np.diff(E)
	return dE/dt

def energy(u,Lx,Ly,Mx,My):
	dx = Lx/Mx
	dy = Ly/My

	return 0.5*np.sum(np.abs(u)**2,axis=(1,2))*dx*dy

def step(sim_time,u_0,d,verbose=1): # set verbose=1 to show timestepping information
	H = dist.Field(name='H',bases=(xbasis,ybasis))
	dX = lambda A: d3.Differentiate(A,coords['X'])
	dY = lambda A: d3.Differentiate(A,coords['Y'])
	problem = d3.IVP([H],namespace=locals())
	problem.add_equation("dt(H)+dX(dX(H))+d*(dX(dX(dX(H)))+dX(dY(dY(H))))+dX(dX(dX(dX(H))))+2*dX(dX(dY(dY(H))))+dY(dY(dY(dY(H))))=-H*dX(H)")
	solver = problem.build_solver(timestepper)
	print("sim_time:", sim_time)
	solver.stop_sim_time=sim_time
	H['g'] = u_0
	H_list = [np.copy(H['g'])]
	t_list = [solver.sim_time]
	while solver.proceed:
		solver.step(timestep)
		if solver.iteration % 100 == 0 and verbose==1:
			logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration,solver.sim_time,timestep))
		if solver.iteration % 1 == 0:
			H.change_scales(1)
			H_list.append(np.copy(H['g']))
			t_list.append(solver.sim_time)
	return H_list,t_list

# assume (ndts,Nx,Ny)
def reshape(u):
	u_square = np.reshape(u,(u.shape[0],u.shape[1]*u.shape[2]))
	return u_square

def i_reshape(u,ndts,Nx,Ny):
	if ndts == 1:
		u_full = np.reshape(u,(Nx,Ny))
	else:
		u_full = np.reshape(u,(ndts,Nx,Ny))
	return u_full

#def appendt(u,dt):
#	if u.ndim == 2:
#		u_t = np.zeros((u.shape[0]+1,u.shape[1]+1))
#		u_t[0,0] = dt
#	else:
#		u_t = np.zeros(u.shape[0])
#		u_t[0] = dt
#	return u_t

#def popt(u):
#	if u.ndim == 2:
#		return u[1:,1:]
#	else:

#		return u[1:]

def guess_residual(T): # find residual for initial guess. will use to optimize guess
	u,_ = step(T,i_reshape(new_x[1:],1,Mx,My),delta) # u0 and d outer
	u = np.array(u)
	return np.linalg.norm(u[0]-u[1])

def optim_residual(T,r):
	opt = minimize_scalar(guess_residual,bounds=(T-r,T+r),method='bounded')
	write_file(filename,"optimized period: T = {:.5f}, res = {:.7f}".format(opt.x,guess_residual(opt.x))+'\n')
	return opt.x




def dotprd(n_,v1,v2):
	n1 = 0
	if n_ == -1:
		n1 = 1
	return np.sum(np.asarray(v1[n1:])*np.asarray(v2[n1:]))

def l_inf(x):
	return np.max(np.absolute(x))

def multJ(n_,dx):
	eps = np.sqrt(dotprd(1,dx,dx))
	eps = epsJ * np.sqrt(dotprd(1,new_x,new_x))/eps
	y = new_x+eps*dx
	s = getrhs(n_,y)
	y = (s-new_fx)/eps 

	if fixT:
		y[0] = 0
	else:
		s = steporbit(1,new_x)
		dt = np.abs(new_x[0]/ndts)
		s = (s-new_x)/dt 
		y[0] = dotprd(-1,s,dx)
	return y

def multJp(n,x):
	return x

def steporbit(ndts_,x):
	global dt
	if ndts_ != 1:
		dt = x[0]/ndts_
		#dt = 0.25
	#dt = timestep
	print(x.shape)
	print(x.ndim)

	if x.ndim == 2:
		if x.shape[0] == n and x.shape[1] == n:
			a = x[:1,-1] # Use last element of solution to step
		else:
			a = x[:,-1]
		a = i_reshape(a,1,Mx,My)
	elif x.ndim == 1: # Last solution is already returned
		a = x
		if len(x) == n:
			a = a[1:]
		a = i_reshape(a,1,Mx,My)
	else:
		a = x
	#for i in range(ndts_):
	#fa,_ = step(int(ndts_*dt),a,delta)
	fa,_ = step(ndts_*dt,a,delta)
	a = fa
	y = np.zeros(len(x))
	a = np.array(a)
	a = reshape(a)
	#y[1:] = a[-1]
	a = a[-1].T
	a_new = np.zeros(n)
	a_new[1:] = a
	#a_new[0] = ndts_
	return a_new

def getrhs(n_,x):
	global ndts
	y_ = steporbit(ndts,x.T)
	#y = y_ - x[-1,:]
	y = y_ - x
	y[0] = 0
	return y

def GMRES(m,n,x,b,res,delt,iters,info):
	#global p
	global v
	global j_
	global h
	global beta_
	if info == 2:
		[y,delt] = GMREShook(j_,h,m,beta_,delt)
		z = v[:,0:j_]@y[0:j_]
		x = multJp(n,z)
		info = 0 
		return x,res,delt,iters,info

	v = np.zeros((n,m+1)) #guess vector

	tol = res 
	imx = iters
	iters = 0
	while 1:

		res_ = 1e99
		stgn = 1e0-1e-14

		beta_ = np.sqrt(dotprd(n,x,x)) 
		if beta_ == 0:
			w = 0
		else:
			w = multJ(n,x)

		w = b - w
		beta_ = np.sqrt(dotprd(n,w,w))
		v[:,0] = w/beta_

		h = np.zeros((m+1,m),dtype=np.float64)
		for j in range(m):
			j_ = j
			iters = iters + 1
			z = v[:,j]
			z = multJp(n,z)
			w = multJ(n,z)
			for i in range(j+1):
				h[i,j] = dotprd(n,w,v[:,i])
				w = w - h[i,j]*v[:,i]
			h[j+1,j] = np.sqrt(dotprd(n,w,w))
			v[:,j+1] = w/h[j+1,j]

			p = np.zeros(j+2)
			p[0] = beta_
			h_ = np.zeros(h[0:j+2,0:j+1].shape)
			h_[0:j+2,0:j+1] = h[0:j+2,0:j+1]
			y = np.linalg.pinv(h_)@p

			p = (-1)*h[0:j+2,0:j+1]@y
			p[0] = p[0]+beta_
			print(beta_)
			print(p[0])
			res = np.sqrt(np.sum(p*p))
			if info==1:
				#print(f'gmresm: it={iters},res={res}')
				write_file(filename,f'gmresm: it={iters},res={res}\n')

			done = ((res<=tol) or (iters==imx) or (res > res_))

			if done or j == m:
				if delt > 0:
					[y,delt] = GMREShook(j,h,m,beta_,delt)
				z = v[:,0:j+1]@y[0:j+1]
				z = multJp(n,z)
				x = x.flatten()

				x = x.T + z.T
				if iters == imx:
					info = 2
				if res>res_:
					info = 1
				if res <= tol:
					info = 0
				if done:
					return x,res,delt,iters,info
				if delt>0:
					#print("gmres: WARNING: m too small")
					write_file(filename,"gmres: WARNING: m too small\n")
			res_ = res*stgn

def saveorbit():
	#print(f"newton iteration: {new_nits}")
	write_file(filename,f"newton iteration: {new_nits}\n")

	norm_x = np.sqrt(dotprd(-1,new_x,new_x))
	relative_err = new_tol/norm_x

def GMREShook(j,h,m,beta,delt):

	a = h[0:j+2,0:j+1]

	[u,s,v] = np.linalg.svd(a)
	s = np.array(s)

	p = beta * u[0,0:j+1]
	p = p.T

	mu = np.max((s[j]*s[j]*1e-6,1e-99))
	qn = 1e99 
	while qn > delt:
		mu = mu * 1.1
		q = p*s/(mu+s*s)
		qn = np.sqrt(np.sum(q*q))

	y = v.T@q
	p = (-1)*h[0:j+2,0:j+1]@y
	p[0] = p[0] + beta
	delt = np.sqrt(np.sum(p*p))
	return y,delt

def NewtonHook(f,m,n,gtol,tol,delt,mndl,mxdl,nits,info):
	global new_x 
	global new_fx
	global new_tol
	global new_delt
	global new_nits 
	global new_gits
	new_nits = 0
	new_gits = 0
	new_delt = delt
	mxdl_ = mxdl
	ginfo = info
	new_fx = f(n,new_x)
	new_tol = np.sqrt(dotprd(n,new_fx,new_fx))

	if delt < 0:
		new_delt = new_tol/10
		mxdl_ = 1e99
	if info == 1:
		#print(f'newton: nits={new_nits} res={new_tol}')
		write_file(filename,f'newton: nits={new_nits} res={new_tol}\n')
	saveorbit()
	x_ = new_x
	fx_ = new_fx
	tol_ = new_tol
	tol__ = 1e99

	if new_tol < tol:
		if info == 1:
			#print("newton: input already converged")
			write_file(filename,"newton: input already converged\n")
			final_res = np.copy(new_tol)
		info = 0
		return info,final_res
	while 1:
		if new_delt<mndl:
			if info == 1:
				#print("newton: trust region too small")
				write_file(filename,"newton: trust region too small\n")
			info = 3
			final_res = -1
			return info,final_res
		s = np.zeros((n,1))
		gres = gtol*new_tol
		gdelt = new_delt
		if ginfo != 2:
			new_gits = m
		if delt==0:
			new_gits = 9999

		s,gres,gdel,new_gits,ginfo = GMRES(m,n,s,fx_,gres,gdelt,new_gits,ginfo)
		ginfo  = info
		s = s.flatten()
		new_x = x_ - s


		new_fx = f(n,new_x)
		new_tol = np.sqrt(dotprd(n,new_fx,new_fx))

		snrm = np.sqrt(dotprd(n,s,s))
		ared = tol_ - new_tol
		pred = tol_ - gdel 

		if info == 1:
			#print(f'newton: nits={new_nits}  res={new_tol}') 
			#print(f'newton: gits={new_gits}  del={new_delt}') 
			#print(f'newton: |s|={snrm}  pred={pred}') 
			#print(f'newton: ared/pred={ared/pred}') 			

			write_file(filename,f'newton: nits={new_nits}  res={new_tol}\n') 
			write_file(filename,f'newton: gits={new_gits}  del={new_delt}\n') 
			write_file(filename,f'newton: |s|={snrm}  pred={pred}\n') 
			write_file(filename,f'newton: ared/pred={ared/pred}\n') 

		if delt == 0:
			if info == 1:
				write_file(filename,"newton: took full newton step\n")

		elif new_tol>tol__:
			if info == 1:
				write_file(filename,"newton: accepting previous step\n")
			new_x = x__ 
			new_fx = fx__
			new_tol = tol__
			new_delt = delt__
		elif ared < 0:
			if info == 1: # changed from new_delt=snrm*0.5
				if new_tol > 10*tol_:
					new_tol *= 0.5
				else:
					new_delt = snrm*0.5
				ginfo = 2
				#print("newton: norm increased, try smaller step")
				write_file(filename,"newton: norm increased, try smaller step\n")
			
		elif ared/pred<0.75:
			if info == 1:
				#print("newton: step ok, trying smaller step")
				write_file(filename,"newton: step ok, trying smaller step\n")
			x__ = new_x 
			fx__ = new_fx 
			tol__ = new_tol
			if ared/pred>0.1:
				delt__ = snrm
			if ared/pred<=0.1:
				delt__ = snrm*0.5
			new_delt = snrm*0.5
			ginfo = 2
		elif snrm<new_delt*0.9:
			if info==1:
				#print('newton:step good, took full newton step')
				write_file(filename,'newton:step good, took full newton step\n')
			new_delt = np.min((mxdl_,snrm*2))
		elif snrm<mxdl_*0.9:
			if info == 1:
				#print("newton: step good, trying larger step")
				write_file(filename,"newton: step good, trying larger step\n")
			x__ = new_x
			fx__ = new_fx
			tol__ = new_tol
			delt__ = new_delt
			new_delt = np.min((mxdl_,snrm*2))
			ginfo = 2 

		if ginfo == 2:
			continue

		new_nits = new_nits+1
		saveorbit()
		x_ = new_x 
		fx_ = new_fx 
		tol_ = new_tol
		tol__ = 1e99 
		if new_tol<tol:
			if info==1:
				#print("newton: converged")
				write_file(filename,"newton: converged\n")
				converged = True
				final_res = np.copy(new_tol)
			info = 0
			return info,final_res
		elif new_nits == nits:
			if info==1:
				#print("newton: reached max iterations")
				write_file(filename,"newton: reached max iterations\n")
				final_res = np.copy(new_tol)
			info = 2
			return info,final_res

if __name__ == "__main__":
	n = 4097
	mgmres = 4096
	nits = 100
	rel_err = 1e-6
	fixT = 0

	delt = -1
	mndl = 1e-20
	mxdl = 1e20
	gtol = 1e-3
	epsJ = 1e-6
	delta = 1.0
	Mx = 64
	My = 64
	Lx = 10
	Ly = 10
	M = Mx*My
	dealias = 3/2
	T_guess = 85.29 # period to step for calculating orbit, set short for now. give a reasonable guess
	write_file(filename,f"Initial T={T_guess}\n")

	# refine period before solving

	stop_sim_time = 62 # generate initial data, up to starting condition
	timestepper = d3.RK222
	timestep = 1e-3
	init_t = 1e-3
	#init_time = 2
	dtype = np.float64

	N_ics = 50 # number of initial conditions from data to solve from

	coords = d3.CartesianCoordinates('X','Y')
	dist = d3.Distributor(coords,dtype=dtype) 
	xbasis = d3.Fourier(coords['X'],size=Mx,bounds=(-Lx/2,Lx/2),dealias=dealias,dtype=dtype)
	ybasis = d3.Fourier(coords['Y'],size=My,bounds=(-Ly/2,Ly/2),dealias=dealias,dtype=dtype)
	H = dist.Field(name='H',bases=(xbasis,ybasis))

	X,Y = dist.local_grids(xbasis,ybasis)

	dX = lambda A: d3.Differentiate(A,coords['X'])
	dY = lambda A: d3.Differentiate(A,coords['Y'])

	problem = d3.IVP([H],namespace=locals())

	problem.add_equation("dt(H)+dX(dX(H))+delta*(dX(dX(dX(H)))+dX(dY(dY(H))))+dX(dX(dX(dX(H))))+2*dX(dX(dY(dY(H))))+dY(dY(dY(dY(H))))=-H*dX(H)")

	#init_H = np.zeros((Mx,My))

	#for i in range(Mx):
	#	for j in range(My):
	#		init_H[i,j] = np.sin(i*np.pi/Mx)*np.cos(j*np.pi/My)
	#init_H[1,1] = 0.1
	if args.load is not None:
		f = h5py.File("PO_T_"+str(args.load[0])+".h5",'r')
		f.visit(lambda x:print(x))
		#init_H = np.array(f['loc']['u0'])
		traj = np.array(f['data']['PO'])
		init_H = traj[0]
		T_orbit = np.array(f['loc']['T'])
		stop_sim_time = np.array(f['loc']['sim_time'])
		print("loading T="+args.load[0])
	else:
		#for (i,j) in zip(range(init_H.shape[0]),range(init_H.shape[1])):
	 	#	init_H[i,j] = np.random.randint(-10,10)*0.001
		with h5py.File("IC_L_d_plot.h5",'r') as f:
			init_H = np.array(f.get("ic"))
			#H['g'] = init_H

	init_H_save = np.copy(init_H)
	#H['g'] = np.real(np.fft.ifft(init_H))
	#H['g'] = init_H
	
	#solver.stop_sim_time = init_time
	if args.load is not None:
		po_init = traj
		t_init = np.linspace(0,T_guess,po_init.shape[0])
	else:
		po_init,t_init = step(stop_sim_time,init_H,delta)

	E_init = energy(po_init,Lx,Ly,Mx,My)
	dEdt = e_t(E_init,timestep)
	plt.figure()
	plt.plot(t_init,E_init,'k')
	plt.show()
	#plt.figure()
	#plt.plot(t_init[int(0/timestep):int(stop_sim_time/timestep)],E_init[int(0/timestep):int(stop_sim_time/timestep)])
	plt.figure()
	plt.plot(E_init[:-1],dEdt,'k')
	plt.show()

	po_init = np.array(po_init)
	#plt.figure()
	#plt.pcolormesh(po_init[-1,:,:])
	#plt.show()
	po_init = reshape(po_init)
	po_init = po_init.T
	true,t_true = step(T_guess,i_reshape(po_init[:,-1],1,Mx,My),delta)
	E_true = energy(true,Lx,Ly,Mx,My)
	dEdt_true = e_t(E_true,timestep)
	plt.figure()
	plt.plot(E_true[:-1],dEdt_true,'g-')
	plt.plot(E_true[0],dEdt_true[0],'gx')
	plt.plot(E_true[-1],dEdt_true[-2],'bx')
	#plt.savefig("phase_plot_L20_d_00294.png")
	plt.show()

	#new_x = np.zeros((M,ndts+2))
	new_x = np.zeros(n)
	#new_x = np.copy(po_init[:,-1]) # Start stepping after transient period
	#print(new_x)
	new_x[1:] = po_init[:,-1]
	T_orbit = optim_residual(T_guess,3*timestep)
	ndts = int(T_orbit/timestep)
	new_x[0] = T_orbit
	#new_x = new_x.T
	#new_x[0] = stop_sim_time*timestep**2

	#iv_plot = i_reshape(new_x,new_x.shape[0],Mx,My)

	#print("plot surface:",iv_plot.shape)

	#plt.figure()
	#plt.pcolormesh(iv_plot[-1])
	#plt.show()

	start_x = np.zeros(n)

	#start_x = np.copy(po_init[:,-1])
	start_x[1:]=po_init[:,-1]
	start_x[0] = T_orbit
	end_x = steporbit(T_orbit/timestep,start_x.T)

	#print("initial residual: ",np.linalg.norm(start_x[1:]-end_x[1:]))
	fig,(ax1,ax2)=plt.subplots(1,2)
	ax1.pcolormesh(np.reshape(start_x[1:],(Mx,My)),cmap='twilight')
	ax2.pcolormesh(np.reshape(end_x[1:],(Mx,My)),cmap='twilight')
	plt.show()


	d = np.sqrt(dotprd(n,new_x,new_x))
	tol = rel_err * d 
	delt = delt * d 
	mndl = mndl * d 
	mxdl = mxdl * d 


	info = 1 
	converged = False
	info,final_res = NewtonHook(getrhs,mgmres,n,gtol,tol,delt,mndl,mxdl,nits,info)

	start_x = new_x 
	#dt = .1
	end_x = steporbit(ndts,start_x)
	true,t_true = step(T_orbit,i_reshape(po_init[:,-1],1,Mx,My),delta)
	E_true = energy(true,Lx,Ly,Mx,My)
	result,t_result = step(ndts*dt,i_reshape(start_x[1:],1,Mx,My),delta)
	E_result = energy(result,Lx,Ly,Mx,My)
	plt.figure()
	plt.plot(t_result,E_result,'k-')
	plt.plot(t_true,E_true,'g-')
	plt.show()
	dEdt=e_t(E_result,timestep)
	dEdt_true = e_t(E_true,timestep)
	plt.figure()
	plt.plot(E_result[:-2],dEdt[:-1],'k-')
	plt.plot(E_true[:-2],dEdt_true[:-1],'g-')
	plt.plot(E_true[0],dEdt_true[0],'gx')
	plt.plot(E_true[-2],dEdt_true[-1],'bx')
	plt.show()

	true = np.array(true)
	result = np.array(result)
	def animate_true(i):
		cax.set_array(true[i,:,:].flatten())
	def animate_pred(i):
		cax.set_array(result[i,:,:].flatten())

	f = h5py.File(f"PO_T_"+"{:0.3f}".format(ndts*dt)+".h5",'w')
	f.require_group('loc')
	f.require_group('data')

	if final_res == -1:
		print("warning: newton solver failed")

	f['loc']['T'] = ndts*dt
	f['loc']['u0'] = init_H_save
	f['loc']['sim_time'] = stop_sim_time
	f['data']['PO'] = result
	f['data']['res'] = final_res
	f['data']['E'] = E_result
	f.close()


	fig,ax = plt.subplots()
	cax = ax.pcolormesh(true[0,:,:],cmap='twilight')
	fig.colorbar(cax)

	result_square = reshape(result)
	#result_square = result_square-np.mean(result_square,axis=0)
	#f_pca = PCA(n_components=3)

	#r_pca = f_pca.fit_transform(result_square)

	print(result_square.shape)


	ax = plt.figure().add_subplot(projection='3d')
	ax.plot(result_square[:,0],result_square[:,1],result_square[:,2])
	plt.show()


	anim = animation.FuncAnimation(fig,animate_true,interval=30,frames=true.shape[0])
	anim.save('true_po_T205_d_002_L22.gif')
	fig,ax = plt.subplots()
	cax = ax.pcolormesh(result[0,:,:],cmap='twilight')
	fig.colorbar(cax)



	anim = animation.FuncAnimation(fig,animate_pred,interval=30,frames=result.shape[0])
	anim.save('pred_po_T205_d_002.gif')





	