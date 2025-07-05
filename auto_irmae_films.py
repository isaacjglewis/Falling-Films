import numpy as np 
import pickle
import matplotlib.ticker as tck
from matplotlib import cm

import h5py

import time
import torch
import torch.nn as nn
import torch.optim as optim

import os
#import psutil
import sys
import math

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

#import tensorflow as tf
#from tensorflow import keras
#import tensorflow.keras.backend as K

import scipy.io
from sklearn.utils.extmath import randomized_svd
import tables

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from numpy import genfromtxt

parser = argparse.ArgumentParser('IRMAE Autoencoder')
# These are the relevant sampling parameters
parser.add_argument('--restart',action='store_true')
args = parser.parse_args()

def getSVD(code_data):
	covMatrix = (code_data.T @ code_data) / len(code_data)
	u,s,v = np.linalg.svd(covMatrix,full_matrices=True)
	return u,s,v

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
							   #nn.Linear(trunc,trunc,bias=False),
                              )
    def forward(self, y):
        # This is the evolution with the NN
        return self.decode(self.lin(self.encode(y)))
	
def Out(text):
	name = "Out_auto_L22_6LL_wd-7.txt"
	newfile=open(name,'+a')
	newfile.write(text+'\n')
	newfile.close()

def shift(w,theta1=0,theta2=0):
	[Nx,Ny]=w.shape
	W=1/Nx*1/Ny*np.fft.fftn(w,axes=(0,1))
	kx = np.append(np.linspace(0,Nx/2-1,Nx/2),np.linspace(-Nx/2,-1,Nx/2))
	ky = np.append(np.linspace(0,Ny/2-1,Ny/2),np.linspace(-Ny/2,-1,Ny/2))
	[Kx,Ky]=np.meshgrid(kx,ky)
	if theta1==0 and theta2==0:
		theta1=np.arctan2(np.imag(W[0,1]),np.real(W[0,1]))
		theta2=np.arctan2(np.imag(W[1,0]),np.real(W[1,0]))
	wshift=np.real(np.fft.ifftn(W*np.exp(-1j*Kx*theta1)*np.exp(-1j*Ky*theta2),axes=(0,1)))

	return wshift

def plot_parity(utt_test,test_predictions):
    plt.plot(utt_test.flatten(), test_predictions.flatten(),'o',color='black',markersize=.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([np.min(utt_test.flatten()),np.max(utt_test.flatten())])
    plt.ylim([np.min(utt_test.flatten()),np.max(utt_test.flatten())])
    plt.plot([-100, 100], [-100, 100])

def plot_hist(error,title,bins=40,ymax=50,xmax=.05):
    plt.hist(error, bins = bins,density=True)
    plt.xlim([-xmax,xmax])
    plt.ylim([0,ymax])
    plt.xlabel("Prediction Error")
    variance=np.mean(error**2)
    plt.title('MSE='+str(round(variance,7)))
    plt.ylabel(title+' PDF')
    
    return variance    	

def plot_trajectories(t,x,y,u,u_pred):
	fig = plt.figure()
	ax1 = fig.add_subplot(121,projection='3d')
	ax1.azim = -45
	ax1.elev = 60
	ax1.set_box_aspect((np.ptp(16*x),np.ptp(16*y),300))
	X,Y = np.meshgrid(x.ravel(),y.ravel())
	pred = func[8](u_test_t)
	pred = pred.detach().numpy()
	pred = np.reshape(pred,(819,N1,N2))
	pred = pred.T
	ax1.plot_surface(X,Y,u_test[:,:,M],cmap=cm.coolwarm)
	ax1.xaxis.set_major_locator(tck.MultipleLocator(20))
	ax1.yaxis.set_major_locator(tck.MultipleLocator(20))
	ax1.set_zticks([-2,0,2])
	ax1.set_ylabel('\ny')
	ax1.set_zlabel('\nu')
	ax1.set_xlabel('\nx')
	fig.subplots_adjust(top=1,bottom=0,left=0,right=1,wspace=0)
	ax2=fig.add_subplot(122,projection='3d')
	ax2.plot_surface(X,Y,pred[:,:,M],cmap=cm.coolwarm)
	ax2.azim=-45
	ax2.elev=60
	ax2.set_box_aspect((np.ptp(16*x),np.ptp(16*y),300))
	ax2.xaxis.set_major_locator(tck.MultipleLocator(20))
	ax2.yaxis.set_major_locator(tck.MultipleLocator(20))
	ax2.set_zticks([-2,0,2])
	ax2.set_xlabel("\nx")
	ax2.set_ylabel("\ny")
	ax2.set_zlabel("\nu")
	plt.savefig("pred_Lx_"+str(Lx)+"_Ly_"+str(Ly)+"T_"+str(plot_t*0.01)+".png")

if __name__ == "__main__":
	Lx = 22
	Ly = 22
	Mx = 64
	My = 64
	file = h5py.File("films_Mx_"+str(Mx)+"_My_"+str(My)+"Lx_"+str(Lx)+"Ly_"+str(Ly)+"d_002/films_Mx_"+str(Mx)+"_My_"+str(My)+"Lx_"+str(Lx)+"Ly_"+str(Ly)+"d_002_s1.h5")
	print("loaded")
	u = file['tasks']['H']
	u = np.array(u)
	u = u[400:,:,:]
	plt.figure()
	plt.pcolormesh(u[0])
	plt.show()
	print(u.shape)
	[M,N1,N2] = u.shape
	u = np.reshape(u,(M,N1*N2)) 
	[_,N] = u.shape 
	frac = .8

	if args.restart:
		print("restarting from file")

	u_mean = np.mean(u,axis=0)
	u_std = np.std(u,axis=0)

	u_norm = (u-u_mean)/u_std
	u_norm = u_norm.T 

	max_trunc = 4096
	#U,S,VT = randomized_svd(u_norm,n_components=max_trunc)

	trunc = 1000
	with h5py.File("pca_L22_d_002.h5",'r') as f:
		a = np.array(f.get('PCA'))
		U = np.array(f.get('U'))
		S = np.array(f.get('S'))

	#a = U[:,:trunc].T @ u_norm
	#u_pred = U[:,:trunc] @ a

	#error = np.mean((u_norm - u_pred) ** 2) 
	#print('PCA reconstruction error:', error)

	svs = np.sqrt(S)

	#u_norm_reconstruction = u_norm.T*u_std + u_mean
	#u_pred_reconstruction = u_pred.T*u_std + u_mean

	#data_final = np.reshape(u_norm_reconstruction, (M, N1, N2))
	#pred_final = np.reshape(u_pred_reconstruction, (M, N1, N2))

	a = a.T 
	u_train = a[:round(M*frac),:]
	u_test = a[round(M*frac):M,:]

	u_train_t = torch.tensor(u_train,dtype=torch.float64)
	u_test_t = torch.tensor(u_test,dtype=torch.float64)
	loader = torch.utils.data.DataLoader(
		u_train,
		batch_size = 1000,
		shuffle=True)

	dh = 26 
	iters = 5000
	weight_decay = 1e-4
	test_freq = 10
	freq = int(iters/10)
	if args.restart:
		auto = torch.load('autos_for_comparison/model_autoencoder_irmae_2d_dh_'+str(dh)+'_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'d_002.pt')
	else:
		auto=Autoencoder(dh,trunc).double()

	optimizer = optim.AdamW(auto.parameters(),lr=1e-3,weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(iters/2), gamma=0.1)
	end = time.time()

	err = []

	for itr in range(1, iters + 1):
		print("iteration: ", itr)
	# Loop over the batch of data
		ii=0
		avgloss=0
		for u in loader:
			ii+=1
			# Get the batch and initialize the optimizer
			optimizer.zero_grad()
			pred=auto(u)
			loss = torch.mean((pred - u)**2) # Compute the mean (because this includes the IC it is not as high as it should be)
			avgloss+=loss.item()
			loss.backward() #Computes the gradient of the loss (w.r.t to the parameters of the network?)
			# Use the optimizer to update the model
			optimizer.step()

		avgloss=avgloss/ii # compute the average loss over the epoch
		scheduler.step() # update the scheduler every epoch

		if itr % test_freq == 0:
			with torch.no_grad():
				err.append(avgloss)
				name='autos_for_comparison/Out_auto_L22_6LL_wd-7.txt'
				newfile=open(name,'a+')
				newfile.write('Iter {:04d} | Total Loss {:.6f} | Time {:.6f} | LR {:.6f} | Weight Decay {:.2e}'.format(itr, avgloss,time.time() - end,optimizer.param_groups[0]['lr'],weight_decay)+'\n')
				newfile.close()

				print("Total loss: {:03f}".format(avgloss))

		stop=open('stop.txt','a+')
		stop.seek(0)
		if stop.read()[:4]=='stop':
			break
		end = time.time()

	torch.save(auto,'autos_for_comparison/model_autoencoder_irmae_2d_dh_'+str(dh)+'_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'d_002_6LL_wd-7.pt')
	print("autoencoder saved")

	plt.figure()
	plt.semilogy(np.arange(test_freq,(len(err)+1)*test_freq,test_freq),np.asarray(err),'.-')
	plt.xlabel('Epoch')
	plt.ylabel('MSE')
	plt.savefig('autos_for_comparison/Error_v_Epochs_irmae_L22_d002_6LL_wd-7.png')

	plt.figure()
	pred=auto(u_test_t)
	MSE=plot_parity(u_test_t.detach().numpy().flatten(),pred.detach().numpy().flatten())
	plt.title(str(MSE))
	plt.tight_layout()
	plt.savefig(open('autos_for_comparison/Parity_irmae_L22_d002_6LL_wd-7.png','wb'))
	plt.close()

	plt.figure()
	#pred_a=auto(u_test_t)
	error = u_test_t.detach().numpy().flatten() - pred.detach().numpy().flatten()
	MSE=plot_hist(error,'NN')
	plt.savefig(open('autos_for_comparison/Statistics_L22_d002_6LL_wd-7.png','wb'))
	plt.close()

	u_test_plot = torch.tensor(u_test,dtype=torch.float64)
	u_norm_plot = torch.tensor(a,dtype=torch.float64)
	pred_reduced = auto.lin(auto.encode(u_norm_plot))
	pred_reduced = pred_reduced.detach().numpy()
	_,S_pred,_ = randomized_svd(pred_reduced,n_components=trunc)
	pred_test = auto(u_test_t)
	test_loss = torch.mean((pred_test-u_test_t)**2)
	Out('Test loss: '+str(test_loss))
	pred_test = pred_test.detach().numpy()
	pred_test = (U[:,:trunc]@pred_test.T).T

	pred_test_temp = np.copy(pred_test)

	pred_test = pred_test*u_std+u_mean
	pred_test = pred_test.T
	#pred = auto.lin(auto.encode(torch.tensor(u_norm,dtype=torch.float64)))

	pred = pred.detach().numpy()
	pred = (U[:,:trunc]@pred.T).T

	#pred = pred*u_std + u_mean
	#pred = pred.T

	#_,S_pred,_ = getSVD(pred_test_temp)
	#_,S_true,_ = getSVD(u_test)

	

	plt.figure()
	plt.semilogy(S/S[0],marker='.',markersize='10',linestyle='-',color='k',label='PCA')
	plt.semilogy(np.arange(1,dh+1),S_pred/S_pred[0],marker='.',markersize='10',linestyle='-',color='r',label="IRMAE")
	plt.xlim([1,40])
	plt.title("Singular Values")
	plt.xlabel(r'$i$')
	plt.ylabel(r'$\sigma$')
	plt.legend()
	plt.savefig("autos_for_comparison/sv_comp_films_dh_"+str(dh)+"L22_6LL_2d-7.png")

	#u_norm = np.reshape(u_norm,(u_norm.shape[0],N1,N2))
	u_test = (U[:,:trunc]@u_test.T).T
	u_test = np.reshape(u_test,(u_test.shape[0],N1,N2))
	#pred_temp = pred
	pred = np.reshape(pred,(pred.shape[0],N1,N2))
	plt.figure()
	fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(6, 3))

	x = np.linspace(-Lx/2,Lx/2,N1)
	y = np.linspace(-Ly/2,Ly/2,N2)
	ax1.pcolormesh(x,y,u_test[5000,:,:],cmap='twilight')
	ax2.pcolormesh(x,y,pred[5000,:,:],cmap='twilight')
	ax1.set_title('True')
	ax1.set_xlim([-Lx/2,Lx/2])
	ax2.set_xlim([-Lx/2,Lx/2])
	ax1.set_ylim([-Ly/2,Ly/2])
	ax2.set_ylim([-Ly/2,Ly/2])
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax2.set_xlabel("x")
	#ax2.set_ylabel("y")
	ax2.set_title('Autoencoder PCA '+str(trunc)+' SV')
	plt.savefig('autos_for_comparison/irmae_auto_result_films_L22_6LL_wd-7.png') 

	# Singular value plot
	#U_pred,S_pred,VT_pred = randomized_svd(pred_temp,n_components=trunc)
	#S_pred = np.sqrt(S_pred)
