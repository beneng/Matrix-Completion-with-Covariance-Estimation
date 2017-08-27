import numpy as np
from scipy.sparse import random
import scipy.sparse as spa
from scipy.sparse.linalg import cg


#Functions used for covariance estimation in paper "Covariance Estimation Using Conjugate Gradient For 3D Classification in Cryo-EM"



#Generates linear operator L
# mat1 is  a p x p matrix
def genL (mat1):
    dims = mat1.shape;
    rows = dims[0];
    cols = dims[1];
    
    LL = np.zeros([rows**2, rows**2]);
    cnt= 0;
    for i in range(rows):
        for j in range(cols):
            row_idx = i*rows+j
            
            roi = mat1[i,:];
            coi = mat1[:,j];
            cnt = j*rows+i
            LL[row_idx,cnt]= roi[i]*coi[j];
            
    return LL;



#Calculate the average projected covariance
# projI , a list of projected images
# meas , number of measurements of a vector made
# dim, the dimension of the vector
def calc_Bn(projI,meas,mu,dim):
    
    obs_cov = np.zeros([dim,dim]);
    num_meas = len(meas);
    
    for i in range(num_meas):
        Is = projI[i,:].reshape([dim,1]);
        Ms = meas[i]
        A = Ms.T
        
        B = Is-np.matmul(Ms,mu);
        C = (Is-np.matmul(Ms,mu)).T;
        D = Ms;
        T1 = np.matmul(A,B);
        
        T2 = np.matmul(T1,C);
       
        Mi = np.matmul(T2,D);
        obs_cov = obs_cov+Mi;
    obs_cov = obs_cov/num_meas;
    return obs_cov;


# Generate Measurement matrices - for experimentation 
# DIM - dimensionality of the vector
# numMeas - Number of measurements made
# num_replace, number of data points per vector to be replaced 

def createMeas (DIM, numMeas,num_replace):
    M =[];
    indices = np.arange(0,DIM,1)
    for i in range(numMeas):
        Ms = np.eye(DIM);
        val_pick = np.random.choice(indices,[num_replace,],replace=False);
        Ms[val_pick,val_pick]=0;
        M.append(Ms)
    return M;

#Average Imaging Operator 

#DIM - dimensionality of the vector of interest
#numMeas - number of measurements made
#M - list of measurement matrices

def avgOperator (DIM, numMeas, M):
    MM = np.zeros([DIM,DIM]);
    for j in range(numMeas):
        MM= MM+np.matmul(M[j].T,M[j]);
    
    MM = MM/numMeas
    return MM;
#Function to get list of projections
#DIM - dimensionality of the vector of interest
#numMeas - number of measurements made
#TD_s - traffic data subset
def getProj (DIM, numMeas,M,TD_s):
    Is = np.zeros([numMeas,DIM]);
    for i in range(numMeas):
    
        Proj = (np.matmul(M[i],TD_s[i,:].reshape([DIM,1]))).reshape([DIM])
        Is[i,:] = Proj;
    return Is

# Average projections

# DIM - dimensionality of the vector of interest
# numMEas - number of measurements
# M - list of measurement operators
# TD_s - Traffic Data subset

# Average projection 

def avgProj (DIM, numMeas,M,TD_s):
    Is = np.zeros([numMeas,DIM]);
    for i in range(numMeas):
        Proj = (np.matmul(M[i],TD_s[i,:].reshape([DIM,1]))).reshape([DIM])
        Is[i,:] = Proj;
    return np.nanmean(Is, axis= 0 );


    # Estimate mu 
#avg_operator - average imaging operator matrix
#avg_I - average projected image
def est_mu (avg_op,avg_I):
    mu_est = np.linalg.solve(avg_op,avg_I)
    return mu_est


# Estimate covariance 
# Average imaging operator Ln in sparse format
# Average covariance


def est_cov (Ln, Bn_p):
    sA = spa.csr_matrix(Ln)
    x0 = cg(sA,Bn_p)
    return x0[0]
