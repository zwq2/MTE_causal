# -*- coding: utf-8 -*-
# @Time    : 2022/4/2 14:21
# @Author  : Zhou wanqi
# @FileName: utils.py
# @Descriptions:
import torch
import numpy as np
import scipy.spatial as sp_spatial
from numba import jit
from collections import deque
from iaaft import surrogates
def GaussianKernel(X):
    """
    Compute Gaussian Kernel
    INPUT: X in R^{Lxdim}
    OUTPUT: A in R^{LxL}
    """
    bn = X.shape[0]
    X = X.reshape(bn, -1)
    utri_ind = np.triu_indices(X.shape[0], 1)
    dist = sp_spatial.distance.cdist(X, X, 'euclidean')
    sigma = np.median(dist[utri_ind])
    K = np.exp(-1 * (dist ** 2) / (2 * sigma ** 2))
    KK = torch.tensor(K)
    return KK

def pairwise_distances(x):
    bn = x.shape[0]
    x = x.reshape(bn, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, alpha):
    # alpha = 1.01
    # k = calculate_gram_mat(x, sigma)
    k = GaussianKernel(x)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, alpha):

    # x = calculate_gram_mat(x, s_x)
    # y = calculate_gram_mat(y, s_y)
    kx = GaussianKernel(x)
    ky = GaussianKernel(y)
    k = torch.mul(kx, ky)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(x, y, alpha):
    Hx = reyi_entropy(x,alpha)
    Hy = reyi_entropy(y,alpha)
    Hxy = joint_entropy(x, y,alpha)
    Ixy = Hx + Hy - Hxy
    return Ixy
def calculate_MI_normalize(x, y, alpha):
    Hx = reyi_entropy(x,alpha)
    Hy = reyi_entropy(y,alpha)
    Hxy = joint_entropy(x, y,alpha)
    Ixy = Hx + Hy - Hxy
    normlizeIxy = Ixy / (torch.max(Hx, Hy) + 1e-16)
    return normlizeIxy
def multi_joint_entroy(variables,alpha):

    K = torch.tensor(1.)
    # for (key,value) in zip(variables,sigmas):
    for key in variables:
        # key = calculate_gram_mat(key,value)
        key = GaussianKernel(key)
        K = torch.mul(key,K)
    K = K / torch.trace(K)
    eigv = torch.abs(torch.symeig(K, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy
def embeddingX(X, tau, dim, u):
    X = torch.tensor(X)
    size_v = X.size()
    if len(size_v) == 1:
        X = X.reshape(len(X), 1)
    T = X.shape[0]  # length of full time series
    L = T - ((dim - 1) * tau)
    feature_dim = X.shape[1]
    FirstP = T - L
    X_emb = torch.zeros(L, dim, feature_dim)
    for ii in range(0, L):
        for jj in range(0, dim):
            X_emb[ii, jj, :] = X[ii + FirstP - jj * tau, :]

    num = X_emb.shape[0]

    X_emb = X_emb[0:num - u, :, :]
    X_emb = torch.from_numpy(X_emb.numpy().transpose((0, 2, 1)))
    return X_emb
def embeddingY(Y, tau, dim, u):
    Y = torch.tensor(Y)
    size_v = Y.size()
    if len(size_v) == 1:
        Y= Y.reshape(len(Y), 1)
    T = Y.shape[0]  # length of full time series
    L = T - ((dim - 1) * tau)
    feature_dim = Y.shape[1]
    FirstP = T - L
    Y_emb = torch.zeros(L, dim, feature_dim)

    for ii in range(0, L):
        for jj in range(0, dim):
            Y_emb[ii, jj, :] = Y[ii + FirstP - jj * tau, :]


    num = Y_emb.shape[0]
    Y_t = Y[FirstP + u:T, :]
    Y_emb = Y_emb[u - 1:num - 1, :, :]
    Y_emb = torch.from_numpy(Y_emb.numpy().transpose((0, 2, 1)))
    return Y_t,Y_emb

def te(X,Y,dim,tau,u,alpha):
    X= torch.tensor(X)
    Y = torch.tensor(Y)
    size_v = X.size()
    if len(size_v) == 1:
        X = X.reshape(len(X), 1)
        Y = Y.reshape(len(Y), 1)
    T = X.shape[0]  # length of full time series
    L = T - ((dim - 1) * tau)
    feature_dim = X.shape[1]
    FirstP = T - L
    Y_emb = torch.zeros(L, dim, feature_dim)
    X_emb = torch.zeros(L, dim, feature_dim)
    for ii in range(0, L):
        for jj in range(0, dim):
            Y_emb[ii, jj, :] = Y[ii + FirstP - jj * tau, :]
            X_emb[ii, jj, :] = X[ii + FirstP - jj * tau, :]

    num = Y_emb.shape[0]
    Y_t = Y[FirstP + u:T, :]
    Y_emb = Y_emb[u - 1:num - 1, :, :]
    X_emb = X_emb[0:num - u, :, :]
    Y_emb = torch.from_numpy(Y_emb.numpy().transpose((0, 2, 1)))
    X_emb = torch.from_numpy(X_emb.numpy().transpose((0, 2, 1)))
    H1 = joint_entropy(Y_t, Y_emb, alpha)  # Y和Y_emb的联合熵
    H2 = reyi_entropy(Y_emb, alpha)  # Y_emb的熵
    H3 = joint_entropy(Y_emb, X_emb, alpha)  # X_emb和Y_emb的联合熵
    H4 = multi_joint_entroy([Y_emb, X_emb, Y_t], alpha)
    Hx = H4 - H3
    Hy = H1 - H2
    te = Hy - Hx
    if te < 0:
        te = 0
    return te
def cte(X, Y, Z, dim, tau, u,alpha):

    # estimate the transfer entropy from x to y, conditioned on z
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    Z = torch.tensor(Z)

    size_v = X.size()
    if len(size_v) == 1:
        X = X.reshape(len(X), 1)
        Y = Y.reshape(len(Y), 1)
        Z = Z.reshape(len(Z), 1)
    T = X.shape[0]  # length of full time series
    L = T - ((dim - 1) * tau)
    feature_dim = X.shape[1]
    FirstP = T - L
    Y_emb = torch.zeros(L, dim, feature_dim)
    X_emb = torch.zeros(L, dim, feature_dim)
    Z_emb = torch.zeros(L, dim, feature_dim)
    for ii in range(0, L):
        for jj in range(0, dim):
            Y_emb[ii, jj, :] = Y[ii + FirstP - jj * tau, :]
            X_emb[ii, jj, :] = X[ii + FirstP - jj * tau, :]
            Z_emb[ii, jj, :] = Z[ii + FirstP - jj * tau, :]

    num = Y_emb.shape[0]
    Y_t = Y[FirstP + u:T, :]
    Y_emb = Y_emb[u - 1:num - 1, :, :]
    X_emb = X_emb[0:num - u, :, :]
    Z_emb = Z_emb[0:num - u, :, :]
        
    Y_emb = torch.from_numpy(Y_emb.numpy().transpose((0, 2, 1)))
    X_emb = torch.from_numpy(X_emb.numpy().transpose((0, 2, 1)))
    Z_emb = torch.from_numpy(Z_emb.numpy().transpose((0, 2, 1)))

    H1 = multi_joint_entroy([Y_t, Y_emb, Z_emb], alpha)  # Y和Y_emb的联合熵
    H2 = joint_entropy(Y_emb, Z_emb,alpha)
    H3 = multi_joint_entroy([Y_emb, X_emb, Z_emb], alpha)  # X_emb和Y_emb的联合熵
    H4 = multi_joint_entroy([Y_emb, X_emb, Y_t, Z_emb], alpha)
    Hx = H4 - H3
    Hy = H1 - H2
    cte = Hy - Hx
    if cte < 0:
        cte = 0
    return cte
def fully_cte(x,y,variables,dim,tau,u,alpha): # the number of conditional variables is larger than one

# variables: conditional variables
    X_emb = embeddingX(x,tau,dim,u)
    Y_t,Y_emb = embeddingY(y,tau,dim,u)
    if len(variables.shape) == 1:
        variables = np.reshape(variables,(variables.size,1))
    num_condition =variables.shape[1]
    V_emb = []
    for num_v in range(num_condition):
        V_emb.append(embeddingX(variables[:,num_v],tau,dim,u))
    H1 = multi_joint_entroy([Y_t, Y_emb]+V_emb, alpha)  
    H2 = multi_joint_entroy([Y_emb]+V_emb,alpha)
    H3 = multi_joint_entroy([Y_emb, X_emb]+V_emb,  alpha)  
    H4 = multi_joint_entroy([Y_emb, X_emb, Y_t]+V_emb, alpha)
    Hx = H4 - H3
    Hy = H1 - H2
    cte = Hy - Hx
    if cte < 0:
        cte = 0
    return cte



def autocorrelation(x):
    """
    Autocorrelation of time series
    INPUT  = x in R^{T}
    OUTPUT =
    """
    xp = (x - np.mean(x)) / np.std(x)
    result = np.correlate(xp, xp, mode='full')
    return result[int(result.size / 2):] / len(xp)

def autocorr_decay_time(x, maxlag):
    """
    Autocorrelation decay time (embedding time)
    INPUT = x in R^{T}
          = maxlag in R (maximum delay time)
    OUTPU = act in R (time delay for embedding)
    """
    autocorr = autocorrelation(x)
    thresh = np.exp(-1)
    aux = autocorr[0:maxlag];
    aux_lag = np.arange(0, maxlag)
    if len(aux_lag[aux < thresh]) == 0:
        act = maxlag
    else:
        act = np.min(aux_lag[aux < thresh])
    return act


def cao_criterion(x, d_max, tau):
    """ Cao criterion (embedding dimension)
    """
    tau = int(tau)
    N = len(x)
    d_max = int(d_max) + 1
    x_emb_lst = []

    for d in range(d_max):
        # Time embedding
        T = np.size(x)
        L = T - (d * tau)
        if L > 0:
            FirstP = T - L
            x_emb = np.zeros((L, d + 1))
            for ii in range(0, L):
                for jj in range(0, d + 1):
                    x_emb[ii, jj] = x[ii + FirstP - (jj * tau)]
            x_emb_lst.append(x_emb)

    d_aux = len(x_emb_lst)
    E = np.zeros(d_aux - 1)
    for d in range(d_aux - 1):
        emb_len = N - ((d + 1) * tau)
        a = np.zeros(emb_len)
        for i in range(emb_len):
            var_den = x_emb_lst[d][i, :] - x_emb_lst[d][0:emb_len, :]
            inf_norm_den = np.linalg.norm(var_den, np.inf, axis=1)
            inf_norm_den[inf_norm_den == 0] = np.inf
            den = np.min(inf_norm_den)
            ind = np.argmin(inf_norm_den)
            num = np.linalg.norm(x_emb_lst[d + 1][i, :] - x_emb_lst[d + 1][ind, :], np.inf)
            a[i] = num / den
        E[d] = np.sum(a) / emb_len

    E1 = np.roll(E, -1)  # circular shift
    E1 = E1[:-1] / E[:-1]

    dim_aux = np.zeros([1, len(E1) - 1])

    for j in range(1, len(E1) - 1):
        dim_aux[0, j] = E1[j - 1] + E1[j + 1] - 2 * E1[j]
    dim_aux[dim_aux == 0] = np.inf
    dim = np.argmin(dim_aux) + 1

    return dim




def Normalize(data):
    data_len,data_num = data.shape
    data = data.flatten()
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    ans = [(float(i)-m)/(mx-mn) for i in data]
    ans = np.array(ans).reshape(data_len,data_num)
    return ans
def nonzero_index(arr,num):
    
    tmp = -1
    arr = list(arr)
    index_list = []

    for i in range(arr.count(num)):
        tmp = arr.index(num,tmp+1,len(arr))
        index_list.append(tmp)
    return index_list

def p_cte(x,y,variables,p_run,dim,tau,u,alpha):
    xx = x.copy()
    yy = y.copy()
    vv = variables.copy()
    if len(variables.shape) == 1:
        variables = np.reshape(variables,(variables.size,1))
    num_v = variables.shape[1]
    temp =[]
    # yy = yy[len(yy) - 1::-1]
    xx_s = surrogates(xx,p_run,verbose=False)
    for k in range(p_run):

        temp.append(fully_cte(xx_s[k,:], yy,vv, dim, tau, u, alpha))
    temp = np.array(temp)
    surr_cte = np.mean(temp) + 2.53*np.std(temp)
    return surr_cte

@jit
def shuffle_data(input_data):
    """Returns a (seeded) randomly shuffled array of data.
    The data input needs to be a two-dimensional numpy array.
    """

    shuffled = np.random.permutation(input_data)

    shuffled_formatted = np.zeros((1, len(shuffled)))
    shuffled_formatted[0, :] = shuffled

    return shuffled_formatted


# Use jit for loop-jitting
@jit(forceobj=True)
def gen_iaaft_surrogates(data, iterations):
    """Generates iAAFT surrogates
    """
    # Make copy to  prevent rotation of array
    data_f = data.copy()
    #    start_time = time.clock()
    xs = data_f.copy()
    # sorted amplitude stored
    xs.sort()
    # amplitude of fourier transform of orig
    pwx = np.abs(np.fft.fft(data_f))

    data_f.shape = (-1, 1)
    # random permutation as starting point
    xsur = np.random.permutation(data_f)
    xsur.shape = (1, -1)

    for i in range(iterations):
        fftsurx = pwx * np.exp(1j * np.angle(np.fft.fft(xsur)))
        xoutb = np.real(np.fft.ifft(fftsurx))
        ranks = xoutb.argsort(axis=1)
        xsur[:, ranks] = xs

    #    end_time = time.clock()
    #    logging.info("Time to generate surrogates: " + str(end_time - start_time))

    return xsur


def timeshifted(timeseries, shift):
    ts = deque(timeseries)
    ts.rotate(shift)
    return np.asarray(ts)

