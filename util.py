# -*- coding: utf-8 -*-

"""
Util function and class.

"""

import pandas as pd
import numpy as np
import json
import os


class Option:
    def __init__(self):
        pass
    def setattr(self, attr, val):
        self.__dict__[attr] = val
    


class Dataset:
    def __init__(self,A=None,b=None,x_init=None, x_sol=None):
        self.A = A
        self.b = b
        self.x_init = x_init
        self.x_sol = x_sol
        if A is not None:
            self.As = np.linalg.norm(A,axis=1) ** 2
            self.As = self.As / (np.sum(self.As))
    
    def setrandls(self, A, b, x_init, x_sol):
        self.A = A
        self.b = b 
        self.x_init = x_init 
        self.x_sol = x_sol
        self.setAs()

    def setcovmm(self, A, b, x_init, x_sol, mapcount):
        self.A = A
        self.b = b
        self.x_init = x_init
        self.x_sol = x_sol
        self.setAs()

    def setAs(self):
        A = self.A
        assert(A is not None    )
        self.As = np.linalg.norm(A,axis=1)**2
        self.As = self.As / (np.sum(self.As))


    def saverandls(self,filename):
        np.savez(filename, self.A, self.b, \
                self.x_init, self.x_sol,self.As)

    def savecovmm(self,filename):
        np.savez(filename, self.A, self.b, \
                self.x_init, self.x_sol,self.As, self.mapcount) 

    def loadrandls(self,filename):
        try:
            npzfile = np.load(filename)
        except Exception as e:
            print(type(e))
            A = None
            return 
        A,b,x_init,x_sol,As = npzfile.files
        A,b,x_init,x_sol,As = npzfile[A],npzfile[b],npzfile[x_init],npzfile[x_sol],npzfile[As]
        self.A = A
        self.b = b
        self.x_init = x_init
        self.x_sol = x_sol
        self.As = As
        
        
    def loadcovmm(self,filename):
        try:
            npzfile = np.load(filename)
        except Exception as e:
            print(type(e), filename)
            A = None
            return 
        A,b,x_init,x_sol,As,mapcount = npzfile.files
        A,b,x_init,x_sol,As,mapcount = npzfile[A],npzfile[b],npzfile[x_init],npzfile[x_sol],npzfile[As],npzfile[mapcount]
        
        self.A = A
        self.b = b
        self.x_init = x_init
        self.x_sol = x_sol
        self.setAs()
        self.mapcount = mapcount

    def gettheory(self, algo='sgd', importance=0):
        self.rate = None
        A  = self.A
        valididx = np.where(self.b!=0)
        A = A[valididx]
        if algo == 'sgd':
            if importance:
                w =  A.T@A / (np.linalg.norm(A,ord='f')**2)
            else:
                m,n = A.shape[0], A.shape[1]
                w = np.zeros((n,n))
                for i in range(m):
                    w += np.outer(A[i],A[i]) / np.linalg.norm(A[i])**2
                w *= 1./m
            labdamin = np.linalg.svd(w)[1][-1]
            assert(labdamin > 0)
            self.rate = (1.-labdamin)**2
        

class Reporter:
    def __init__(self, filepath=None, algoname=None, dataname=None, option=None):
        self.filepath = filepath
        self.algoname = algoname
        self.dataname = dataname
        self.option = option
        self.namelist = []
        
    def registervar(self, varname):
        self.namelist.append(varname)
        self.__dict__[varname] = []

    def to_csv(self,overwrite=False):
        filename = self.filepath + self.algoname + self.dataname + '.json'
        if not overwrite and os.path.exists(filename):
            print("{} exits, choose another name or set overwrite=True.".format(filename))
            return 
        optser = json.dumps(self.option.__dict__)
        recorddiction = {}
        for e in self.__dict__:
            if e == 'option':
                continue
            else:
                recorddiction[e] = self.__dict__[e]
        recordser = json.dumps(recorddiction)
        finaldiction = {'opt': optser,'record': recordser}
        
        with open(filename,'w') as f:
            json.dump(finaldiction, f)

    def load_csv(self, algoname, dataname, filename):
        with open(filename, 'r') as f:
            finaldiction = json.load(f)
            record = json.loads(finaldiction['record'])
            opt = json.loads(finaldiction['opt'])
        self.option.__dict__ = opt
        self.algoname = algoname
        self.dataname = dataname
        for k,v in record.items():
            self.__dict__[k] = v


Colorstr = ['#E53A40','#30A9DE','#EFDC05','#379392','#791E94','#AACD6E']
Markerstr = ['D','v','*','s','o','x']

def norm2(vec):
    return np.linalg.norm(vec)**2

def norminf(vec):
    return np.linalg.norm(vec,ord=np.inf)

def getloglb(listvec):
    combind = []
    for v in listvec:
        combind.extend(v)
    minval = np.min(combind)
    logval = np.log10(minval)
    exp = None
    if logval < 0:
        exp = (int(logval)-1)
    else:
        exp = (int(logval)+1)
    return 10 ** exp