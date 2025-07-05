import numpy as np 
from numpy.fft import fft, ifft
from math import pi

import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
import seaborn as sns
import pickle
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import torch
import torch.nn as nn
import torch.optim as optim

from numpy import genfromtxt

import h5py

from sklearn.utils.extmath import randomized_svd

import argparse
import time

parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--data_size', type=int, default=100)  #IC from the simulation
parser.add_argument('--step',type=float,default=1) # This is steps between snaps
parser.add_argument('--sim_time', type=float, default=10)  #Time of simulation to draw samples from
parser.add_argument('--batch_time', type=int, default=9)   # batch time 9 -> 90
parser.add_argument('--batch_size', type=int, default=300)   #Number of IC to calc gradient with each iteration

parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--niters', type=int, default=1500)	   #Iterations of training
parser.add_argument('--test_freq', type=int, default=20)	#Frequency for outputting test loss
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--restart',action='store_true')
parser.add_argument('--checkpoint',type=int,default=1000) # Number of epochs to checkpoint NN
parser.add_argument('--notrain',action='store_true')
args = parser.parse_args()

#Add 1 to batch_time to include the IC
args.batch_time+=1 

# Determines what solver to use
if args.adjoint:
	from torchdiffeq import odeint_adjoint as odeint
else:
	# This is the default
	from torchdiffeq import odeint

# Check if there are gpus
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

class ODEFunc(nn.Module):
	def __init__(self,trunc,a):
		super(ODEFunc, self).__init__()
		self.trunc=trunc
		# Change the NN architecture here
		self.net = nn.Sequential(
			nn.Linear(trunc, 500), # 500
			nn.Sigmoid(),
			nn.Linear(500, 500),
			nn.Sigmoid(),
			nn.Linear(500, 500),
			nn.Sigmoid(),
			nn.Linear(500, trunc),
		)

		self.lin=nn.Sequential(nn.Linear(trunc, trunc,bias=False),)

		for m in self.net.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0, std=0.1)
				nn.init.constant_(m.bias, val=0)
		
		for m in self.lin.modules():
			if isinstance(m, nn.Linear):
				m.weight=nn.Parameter(torch.from_numpy(Linear(trunc,a)).float())
				m.weight.requires_grad=False

	def forward(self, t, y):
		# This is the evolution with the NN
		return self.lin(y)+self.net(y)

def Linear(N,a):
	A=np.diag(-a*np.ones(N))
	return A

def get_batch(t,true_y):
	rand=np.random.choice(np.arange(np.floor(args.data_size/args.step) - args.batch_time, dtype=np.int64), args.batch_size, replace=False)

	s = torch.from_numpy(rand)
	batch_y0 = true_y[s]  # (M, D)
	batch_t = t[:args.batch_time]  # (T)
	batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
	return batch_y0, batch_t, batch_y

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

class Autoencoder(nn.Module):
	def __init__(self,trunc,N):
		super(Autoencoder,self).__init__()
		self.encode = nn.Sequential(
			nn.Linear(N, 500),
			nn.Sigmoid(),
			nn.Linear(500,trunc),
		) 
		self.decode = nn.Sequential(
			nn.Linear(trunc, 500),
			nn.Sigmoid(),
			nn.Linear(500,N),

		) 

		self.lin=nn.Sequential(nn.Linear(trunc,trunc,bias=False),
							   nn.Linear(trunc,trunc,bias=False),
							   nn.Linear(trunc,trunc,bias=False),
							   nn.Linear(trunc,trunc,bias=False),
							   nn.Linear(trunc,trunc,bias=False),
							  )
	def forward(self, y):
		# This is the evolution with the NN
		return self.decode(self.lin(self.encode(y)))

def getSVD(code_data):
	covMatrix = (code_data.T @ code_data) / len(code_data)
	u,s,v = np.linalg.svd(covMatrix,full_matrices=True)
	return u,s,v

def Out(text):
	name = "Out_node_L22.txt"
	newfile=open(name,'+a')
	newfile.write(text+'\n')
	newfile.close()

def EnergyCalc(ut,N,reduced):
	utfft=fft(ut,axis=0)

	dudx = np.zeros((N,reduced-1))
	dudx2=np.zeros((N,reduced-1))

	L = 22
	kn = 2*pi/L*np.append(np.linspace(0,int(N/2-1),int(N/2)),np.linspace(int(-N/2),-1,int(N/2)))
	print(kn.shape)

	for m in range(reduced-1):
		dudx[:,m]=ifft(utfft[:,m]*(1j*kn))
		dudx2[:,m]=ifft(utfft[:,m]*(1j*kn)**2)

	return dudx,dudx2

if __name__ == "__main__":
	Lx = 22
	Ly = 22
	Mx = 64
	My = 64
	dh = 22
	dt = 0.1
	file = h5py.File('films_Mx_'+str(Mx)+'_My_'+str(My)+'Lx_'+str(Lx)+'Ly_'+str(Ly)+'d_002/films_Mx_'+str(Mx)+'_My_'+str(My)+'Lx_'+str(Lx)+'Ly_'+str(Ly)+'d_002_s1.h5')
	auto = torch.load('autos/model_autoencoder_irmae_2d_dh_'+str(dh)+'_Lx_22_Ly_22d_002_3.pt')

	u = file['tasks']['H']
	u = np.array(u)
	u = u[50000:,:,:]
	print(device)
	print(u.shape)
	[M,N1,N2] = u.shape
	u = np.reshape(u,(M,N1*N2))
	[_,N] = u.shape 
	frac = .8

	u_mean = np.mean(u,axis=0)
	u_std = np.std(u,axis=0)

	u_norm = (u-u_mean)/u_std
	u_norm = u_norm.T

	max_trunc = 4096
	trunc = 1000
	with h5py.File("pca_L22_d_002.h5",'r') as f:
		a = np.array(f.get('PCA'))
		U_full = np.array(f.get('U'))
		S = np.array(f.get('S'))
	#U,S,VT = randomized_svd(u_norm,n_components=max_trunc)

	a = U_full[:,:trunc].T @ u_norm
	u_pred = U_full[:,:trunc] @ a 

	error = np.mean((u_norm-u_pred)**2)

	svs = np.sqrt(S)

	a = a.T
	print(a.shape)

	u_train = a[:round(M*frac),:]
	u_test = a[round(M*frac):M,:]

	u_train_t = torch.tensor(u_train,dtype=torch.float64)
	u_test_t = torch.tensor(u_test,dtype=torch.float64)

	data_h = auto.lin(auto.encode(torch.tensor(a,dtype=torch.float64)))
	latent_data = data_h.detach().numpy()
	Out(str(latent_data.shape))
	hstd = np.max(np.std(latent_data,axis=0))
	hmean = np.mean(latent_data,axis=0)
	latent_data = (latent_data-hmean[np.newaxis,:])/hstd

	#U,S,VT = getSVD(latent_data)
	#with h5py.File("pca_red_L30_d_002.h5",'w') as file:
	#	file.create_dataset("U",data=U)
	#	file.create_dataset("S",data=S)
	#	file.create_dataset("VT",data=VT)

	with h5py.File("pca_red_L22_d_002.h5",'r') as file:
		U = np.array(file.get('U'))

	h_train = U[:,:dh].T @ latent_data[:round(M*frac),:].T
	h_test = U[:,:dh].T @ latent_data[round(M*frac):M,:].T
	#h_test = (h_test - np.mean(h_test,axis=0))/np.max(np.std(h_test,axis=0))
	h_test = h_test.T

	#data_norm=(h_train-hmean[:,np.newaxis])/hstd
	#data_norm = data_norm.T

	true_y = torch.tensor(h_train[:,np.newaxis,:],dtype=torch.float64)
	true_y = torch.tensor(true_y.detach().numpy().T,dtype=torch.float64)

	#iters = 500
	t = np.arange(round(M*frac))*dt 
	args.data_size=round(M*frac)
	t = torch.tensor(t,dtype=torch.float64)
	if args.restart == True:
		func = torch.load('model_node_irmae_2d_dh_'+str(dh)+'d_002_L_22_test.pt')
		print("loaded from file")
	else:
		func=ODEFunc(dh,.75*np.max(np.std(data_h.detach().numpy()))).double()
		
	optimizer=optim.Adam(func.parameters(),lr=1e-3)
	scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=int(args.niters/3),gamma=0.1)
	criterion = torch.nn.MSELoss()
	end = time.time()

	err = []
	ii = 0

	for itr in range(1, args.niters + 1):
		# Get the batch and initialzie the optimizer
		optimizer.zero_grad()
		batch_y0, batch_t, batch_y = get_batch(t,true_y)
		if itr==1:
			Out('Batch Time Units: '+str(batch_t.detach().numpy()[-1])+'\n')

		# Make a prediction and calculate the loss
		pred_y = odeint(func, batch_y0, batch_t)
		#loss = torch.mean((pred_y - batch_y)**2) # Compute the mean (because this includes the IC it is not as high as it should be)
		loss = criterion(pred_y,batch_y)
		loss.backward() 
		# Use the optimizer to update the model
		optimizer.step()
		scheduler.step()

		# Print out the Loss and the time the computation took

		if itr % args.test_freq == 0:
			print(f'Epoch {itr}, Loss: {loss.item()}')
		if itr % args.test_freq == 0:
			with torch.no_grad():
				err.append(loss.item())
				Out('Iter {:04d} | Total Loss {:.6f} | Time {:.6f} | LR {:.6f}'.format(itr, loss.item(),time.time() - end,optimizer.param_groups[0]['lr']))
				ii += 1
			
			stop=open('stop.txt','a+')
			stop.seek(0)
			if stop.read()[:4]=='stop':
				break
		if itr % args.checkpoint == 0:
			torch.save(func,'model_node_irmae_2d_dh_'+str(dh)+'d_002_L_22_test.pt')
			Out('Write at {:04d} epochs'.format(itr))

		end = time.time()

	torch.save(func, 'model_node_irmae_2d_dh_'+str(dh)+'d_002_L_22_test.pt')

	pred_test = auto(u_test_t)
	#h_test = auto.lin(auto.encode(u_test_t))
	#h_test = (h_test.detach().numpy() - hmean)/hstd
	#print(h_test.detach().numpy().shape)
	#h_test = U[:,:dh].T @ h_test.detach().numpy().T
	#h_test = h_test.T 

	T = 3400
	ex = 0
	y0=torch.tensor(h_test[ex:ex+1,np.newaxis,:],dtype=torch.float64)
	tt = np.arange(T)*dt
	t = torch.tensor(tt,dtype=torch.float64)
	hNN = odeint(func,y0,t)
	hNN=np.squeeze(hNN.detach().numpy())
	hNN = U[:,:dh]@hNN.T
	hNN = hNN.T
	print(hNN.shape)
#	hNN = auto.decode(torch.tensor(hNN,dtype=torch.float64)).detach().numpy()


	T_plot = 200
	plt.figure()
	plt.plot(tt,np.linalg.norm(h_test[ex:ex+T,:],axis=-1),'g',label='Truth')
	plt.plot(tt,np.linalg.norm(hNN[ex:ex+T,:],axis=-1),'b',label='NODE')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$E(t)$')
	plt.legend()
	#plt.xlim([0,T*dt])
	plt.savefig('Energy_d_002_L22.png')

	T = 3400
	ex = 0 
	h_train = h_train.T
	print(h_train.shape)
	y0_train = torch.tensor(h_train[ex:ex+1,np.newaxis,:],dtype=torch.float64)
	tt = np.arange(T)*dt 
	t = torch.tensor(tt,dtype=torch.float64)
	hNN_train = odeint(func,y0_train,t)
	hNN_train = np.squeeze(hNN_train.detach().numpy()) 
	hNN_train = U[:,:dh] @ hNN_train.T
	hNN_train = hNN_train.T 
	print(hNN_train.shape)
	plt.figure()
	plt.plot(tt,np.linalg.norm(h_train[ex:ex+T,:],axis=-1))
	plt.plot(tt,np.linalg.norm(hNN_train[ex:ex+T,:],axis=-1))
	plt.savefig("Energy_train_d_002_L22.png")

	plt.figure()
	plt.semilogy(np.arange(args.test_freq,args.niters+1,args.test_freq),np.asarray(err),'.-')
	plt.xlabel('Epoch')
	plt.ylabel('MAE')
	plt.savefig('Error_v_Epochs_node_d_002_L22.png')

	hNN = hNN*hstd+hmean[np.newaxis,:]

	uNN = auto.decode(torch.tensor(hNN,dtype=torch.float64)).detach().numpy()
	u_test = (U_full[:,:trunc]@u_test.T).T
	u_test_reshape = np.reshape(u_test,(u_test.shape[0],N1,N2))

	uNN = (U_full[:,:trunc]@uNN.T).T 
	uNN = uNN*u_std+u_mean
	uNN = np.reshape(uNN,(uNN.shape[0],N1,N2))

	x = np.linspace(-Lx/2,Lx/2,N1)
	y = np.linspace(-Ly/2,Ly/2,N2)

	T_snapshots = [i*T//5 for i in range(5)]
	for i in range(len(T_snapshots)):
		fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
		true_cmap = ax1.pcolormesh(x,y,u_test_reshape[T_snapshots[i]],cmap='twilight')
		pred_cmap = ax2.pcolormesh(x,y,uNN[T_snapshots[i]],cmap='twilight')
		ax2.set_xlabel("x")
		ax1.set_ylabel("y")
		ax2.set_ylabel("y")
		ax1.set_aspect("equal","box")
		ax2.set_aspect("equal","box")
		ax1.set_title("True")
		ax2.set_title("NODE")
		fig.colorbar(true_cmap,ax=ax1)
		fig.colorbar(pred_cmap,ax=ax2)

		plt.savefig("Snapshot_L22_"+str(T_snapshots[i]*dt)+".png",bbox_inches='tight',pad_inches=0.0)
		plt.close()

	uxt,uxxt = EnergyCalc(u_test_reshape[:T,:,My//2].T,Mx,u_test.shape[0])
	uyt,uyyt = EnergyCalc(u_test_reshape[:T,Mx//2,:].T,My,u_test.shape[0])
	uxt = uxt.flatten()
	uxxt = uxxt.flatten()

	uyt = uyt.flatten()
	uyyt = uyyt.flatten()

	uxp,uxxp = EnergyCalc(uNN[:,:,My//2].T,Mx,uNN.shape[0])
	uyp,uyyp = EnergyCalc(uNN[:,Mx//2,:].T,My,uNN.shape[0])
	uxp = uxp.flatten()
	uxxp = uxxp.flatten()

	uyp = uyp.flatten()
	uyyp = uyyp.flatten()

	fig = plt.figure(figsize=(3,3),dpi=200)
	### Data
	# ax = plt.subplot(131)
	plt.title('True')
	hx, xedgesx, yedgesx, imagex = plt.hist2d(uxt,uxxt,bins=(100,100),density=True,norm=colors.LogNorm(vmin=1e-6, vmax=1.0),range=[[-4,4],[-4,4]])
	# h, xedges, yedges, image = plt.hist2d(upt,uppt,bins=(200,200),density=True)
	# plt.xlim([-25,25])
	# plt.ylim([-25,25])
	# ax.set_aspect('equal')
	plt.xlabel(r'$u_x$')
	plt.ylabel(r'$u_{xx}$')
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# clb = fig.colorbar(image,cax=cax)
	plt.set_cmap('inferno')
	plt.tight_layout()
	plt.savefig(open('PDF_x_truth_L22.png','wb'))
	fig = plt.figure(figsize=(3,3),dpi=200)
	### Data
	# ax = plt.subplot(131)
	plt.title('True')
	hy, xedgesy, yedgesy, imagey = plt.hist2d(uyt,uyyt,bins=(100,100),density=True,norm=colors.LogNorm(vmin=1e-6, vmax=1.0),range=[[-4,4],[-4,4]])
	# h, xedges, yedges, image = plt.hist2d(upt,uppt,bins=(200,200),density=True)
	# plt.xlim([-25,25])
	# plt.ylim([-25,25])
	# ax.set_aspect('equal')
	plt.xlabel(r'$uy$')
	plt.ylabel(r'$u_{yy}$')
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# clb = fig.colorbar(image,cax=cax)
	plt.set_cmap('inferno')
	plt.tight_layout()
	plt.savefig(open('PDF_y_truth_L22.png','wb'))

	fig = plt.figure(figsize=(3,3),dpi=200)
	### Data
	# ax = plt.subplot(131)
	plt.title('NODE')
	h, xedges, yedges, image = plt.hist2d(uxp,uxxp,bins=(100,100),density=True,norm=colors.LogNorm(vmin=1e-6, vmax=1.0),range=[[-4,4],[-4,4]])
	# h, xedges, yedges, image = plt.hist2d(upt,uppt,bins=(200,200),density=True)
	# plt.xlim([-25,25])
	# plt.ylim([-25,25])
	# ax.set_aspect('equal')
	plt.xlabel(r'$u_x$')
	plt.ylabel(r'$u_{xx}$')
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# clb = fig.colorbar(image,cax=cax)
	plt.set_cmap('inferno')
	plt.tight_layout()
	plt.savefig(open('PDF_x_NODE_L22.png','wb'))
	fig = plt.figure(figsize=(3,3),dpi=200)
	### Data
	# ax = plt.subplot(131)
	plt.title('NODE')
	h, xedges, yedges, image = plt.hist2d(uyp,uyyp,bins=(100,100),density=True,norm=colors.LogNorm(vmin=1e-6, vmax=1.0),range=[[-4,4],[-4,4]])
	# h, xedges, yedges, image = plt.hist2d(upt,uppt,bins=(200,200),density=True)
	# plt.xlim([-25,25])
	# plt.ylim([-25,25])
	# ax.set_aspect('equal')
	plt.xlabel(r'$u_y$')
	plt.ylabel(r'$u_{yy}$')
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# clb = fig.colorbar(image,cax=cax)
	plt.set_cmap('inferno')
	plt.tight_layout()
	plt.savefig(open('PDF_y_NODE_L22.png','wb'))






