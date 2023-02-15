# -*- coding: utf-8 -*-
# @Time    : 2022/4/22 10:38
# @Author  : Zhou wanqi
# @FileName: Inte_CTE.py

import torch
import numpy as np
from utils import *
from teaspoon.parameter_selection.FNN_n import FNN_n
from teaspoon.parameter_selection.autocorrelation import autoCorrelation_tau
from teaspoon.parameter_selection.MI_delay import MI_for_delay
from math import frexp, ldexp
import math
from iaaft import surrogates
class CTE():
    # data : input data (for causal discovery)
    # p_s, p_t, p_run, type: the parameters of choosing different significance analysis method
    
    def __init__(self,data,p_s,p_t,p_run,p_length,type):
        self.data = data
        self.p_s = p_s
        self.p_t = p_t
        self.p_run = p_run
        self.p_length = p_length
        self.T,self.N = self.data.shape # T:time series;N:number of variables
        self.type = type
    
    # obtain time delay parameters
    def tau_matrix_generate(self):
        self.tau_matrix = np.zeros((self.N,),dtype=int)
        for i in range(0,self.N):
            self.tau_matrix[i] = autocorr_decay_time(self.data[:,i],600)
            # b = MI_for_delay(self.data[:,i], plotting=False, method='basic', h_method='sturge', k=2, ranking=True)
            # self.tau_matrix[i] = autoCorrelation_tau(self.data[:,i], cutoff=1 / np.exp(1), AC_method='pearson', plotting=False)
        return self.tau_matrix

    # obtain embedding dimension method 
    def embed_matrix_generate(self):
        self.embed_matrix = np.zeros((self.N,),dtype=int)
        for i in range(0,self.N):
            self.embed_matrix[i] = cao_criterion(self.data[:,i],40,self.tau_matrix[i])
            # _,self.embed_matrix[i] = FNN_n(self.data[:,i], self.tau_matrix[i,0], plotting=False)
        return self.embed_matrix

    # transfer entropy of each pair of variable
    def te_matrix_generate(self):
        self.tau_matrix= np.ones((self.N,)).astype(int)
        self.embed_matrix = 8*np.ones((self.N,)).astype(int)
        self.te_matrix = np.zeros((self.N,self.N))
        for i in range(0,self.N):
            for j in range(0,self.N):
                if j!=i:
                    self.te_matrix[i, j] = te(self.data[:, i], self.data[:, j], self.embed_matrix[j,],self.tau_matrix[j,], 1,1.01)  
        return self.te_matrix


    # Different significant method

    def P_test(self):
        # you can choose your method according to your dataset
        self.surr_te_matrix = np.zeros((self.N, self.N)) 
        if self.type == 'long':
            for i in range(0, self.N):
                for j in range(0, self.N):
                    temp = []
                    if j!= i :
                        a = self.p_s
                        b = self.p_t
                        for k in range(0,self.p_run):
                            target = self.data[b:b + self.p_length, j].copy()
                            temp.append(te(self.data[a:a + self.p_length, i],
                                                       target,
                                                    self.embed_matrix[j],
                                                       self.tau_matrix[j], 1,1.01))
                            
                            a = a + 1
                            b = b + 1

                        temp = np.array(temp)
                        self.surr_te_matrix[i, j] = np.mean(temp) + 3*np.std(temp)
        if self.type =='short':
            for i in range(self.N):
                for j in range(self.N):
                    if j==i:
                        self.surr_te_matrix[i,j]=0
                    else:
                        # count = 0
                        source = self.data[:,i].copy()

                        temp = []
                        # target = target[self.T-1::-1]
                        # target_1 = target.copy()
                        for k in range(self.p_run):
                            target = self.data[:, j].copy()
                            order = np.random.permutation(self.T)
                            # np.random.shuffle(source)
                            target = source[order]
                            # order = np.random.permutation(self.T)
                            # target = target[order]

                            temp.append(te(source,target,1,self.tau_matrix[j],1,2))
                            # if temp > self.te_matrix[i,j]:
                            #     count = count + 1
                        # temp = temp - self.te_matrix[i,j]
                        # count = temp.count(temp>0)
                        # self.surr_te_matrix[i,j] = count/self.p_run
                        temp = np.array(temp)
                        self.surr_te_matrix[i,j] = np.mean(temp) + 2.53*np.std(temp)
        if self.type == 'inverse':
            for i in range(self.N):
                for j in range(self.N):
                    if j==i:
                        self.surr_te_matrix[i,j]=0
                    else:
                        reverse = self.data[:,j].copy()
                        reverse = reverse[::1]
                        self.surr_te_matrix[i, j] = te(self.data[:,i],reverse,3,1,1,2)
        if self.type == 'iaaft':
            for i in range(self.N):
                for j in range(self.N):
                   if j!=i:
                        xsource = self.data[:,i].copy()
                        xs = surrogates(x= xsource,ns=self.p_run,verbose=False)
                        temp = []
                        for run in range(self.p_run):
                            temp.append(te(xs[run,:],self.data[:,j],self.embed_matrix[j],self.tau_matrix[j],1,1.01))
                        temp = np.array(temp)
                        self.surr_te_matrix[i,j] = np.mean(temp) + 2.53*np.std(temp)
        return self.surr_te_matrix



# stage 1 obtain initial dense causal matrix: row (dirver/cause); col (effect/result)
    def te_causal_matrix_generate(self):
        self.te_causal_matrix = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self.te_matrix[i,j] > self.te_matrix[j,i] and self.te_matrix[i,j] > self.surr_te_matrix[i,j] -1e-4:
                    self.te_causal_matrix[i,j] = 1
        return self.te_causal_matrix


# stage 2 remove indirect causal influence
    def del_suprious_causal(self):
        
        self.di_cuasl_matrix = self.te_causal_matrix.copy()
        for i in range(self.N):
            temp_row = self.di_causal_matrix[i,:]
            index_list = nonzero_index(temp_row,1)
            for j in index_list:
                    i_result_list = index_list.copy()
                    i_result_list.remove(j) # obtain the direct effect variables of x_i
                    j_cause_list = nonzero_index(self.di_causal_matrix[:,j],1) # obtain the direct cause variables of x_j
                    variable_list = list(set(i_result_list).intersection(set(j_cause_list)))
                    if len(variable_list) !=0:

                        cte_temp = fully_cte(self.data[:,i],self.data[:,j],self.data[:,variable_list],self.embed_matrix[j],self.tau_matrix[j],1,1.01)
                        cte_surr = p_cte(self.data[:,i],self.data[:,j],self.data[:,variable_list],self.p_run,self.embed_matrix[j],self.tau_matrix[j],1,1.01)
                        if cte_temp - cte_surr < -1e-4:
                            self.di_causal_matrix[i,j] = 0
        return self.di_causal_matrix


# stage 3 remove common causal influence
    def del_common_causal(self):
        self.dc_causal_matrix = self.di_causal_matrix.copy()
        for i in range(self.N):
            temp_col = self.dc_causal_matrix[:, i]
            temp_row = self.dc_causal_matrix[i, :]
            index_list = nonzero_index(temp_row, 1)  
            col_list = nonzero_index(temp_col, 1)
            for j in index_list:
                i_cause_list = col_list.copy() # obtain the direct cause of x_i
                j_cause_list = nonzero_index(self.dc_causal_matrix[:, j], 1) # obtain the direct cause of x_j
                variable_list = list(set(i_cause_list).intersection(set(j_cause_list)))
                cte_temp = fully_cte(self.data[:, i], self.data[:, j], self.data[:, variable_list], self.embed_matrix[j],self.tau_matrix[j], 1, 1.01) # estiamte the conditional mte
                cte_surr = p_cte(self.data[:, i], self.data[:, j], self.data[:, variable_list], self.p_run, self.embed_matrix[j],self.tau_matrix[j], 1, 1.01)
                if cte_temp - cte_surr < -1e-4:
                    self.dc_causal_matrix[i, j] = 0
        return self.dc_causal_matrix
        













