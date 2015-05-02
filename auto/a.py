from __future__ import division
try:
    from runtime import *
except ImportError:
    from smop.runtime import *

import matplotlib
import matplotlib.pyplot as plt

import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
from PIL import Image 
from copy import deepcopy

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, deepcopy(v, memo))
        return result

def defaultParameters_():

    params = AttributeDict()

    # switches
    params.DO_PREPROCESSING = 1
    params.DO_RESIZE        = 0
    params.DO_GRAYLEVEL     = 1
    params.DO_PATCHNORMALIZATION    = 1
    params.DO_SAVE_PREPROCESSED_IMG = 0
    params.DO_DIFF_MATRIX   = 1
    params.DO_CONTRAST_ENHANCEMENT  = 1
    params.DO_FIND_MATCHES  = 1


    # parameters for preprocessing
    params.downsample = AttributeDict()
    params.downsample.size = [32, 64]  # height, width
    params.downsample.method = Image.LANCZOS
    params.normalization = AttributeDict()
    params.normalization.sideLength = 8
    params.normalization.mode = 1
            
    
    # parameters regarding the matching between images
    params.matching = AttributeDict()
    params.matching.ds = 10 
    params.matching.Rrecent=5
    params.matching.vmin = 0.8
    params.matching.vskip = 0.1
    params.matching.vmax = 1.2  
    params.matching.Rwindow = 10
    params.matching.save = 1
    params.matching.load = 1
    
    # parameters for contrast enhancement on difference matrix
    params.contrastEnhancement = AttributeDict()  
    params.contrastEnhancement.R = 10

    # load old results or re-calculate? save results?
    params.differenceMatrix = AttributeDict()
    params.differenceMatrix.save = 1
    params.differenceMatrix.load = 1
    
    params.contrastEnhanced = AttributeDict()
    params.contrastEnhanced.save = 1
    params.contrastEnhanced.load = 1
    
    # suffix appended on files containing the results
    params.saveSuffix=''
    
    return params



def doContrastEnhancement_(results=None,params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 2-[results,params].count(None)+len(args)

    filename=sprintf_(char('%s/differenceEnhanced-%s-%s%s.mat'),params.savePath,params.dataset(1).saveFile,params.dataset(2).saveFile,params.saveSuffix)
    if params.contrastEnhanced.load and exist_(filename,char('file')):
        display_(sprintf_(char('Loading contrast-enhanced image distance matrix from file %s ...'),filename))
        dd=load_(filename)
        results.DD=dd.DD
    else:
        display_(char('Performing local contrast enhancement on difference matrix ...'))
        DD=zeros_(size_(results.D),char('single'))
        D=results.D
        for i in arange_(1,size_(results.D,1)).reshape(-1):
            a=max_(1,i - params.contrastEnhancement.R / 2)
            b=min_(size_(D,1),i + params.contrastEnhancement.R / 2)
            v=D[a:b,:]
            DD[i,:]=(D[i,:] - mean_(v)) / std_(v)
        results.DD=DD - min_(min_(DD))
        if params.contrastEnhanced.save:
            DD=results.DD
            save_(filename,char('DD'))
    return results
def doDifferenceMatrix_(results=None,params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 2-[results,params].count(None)+len(args)

    filename=sprintf_(char('%s/difference-%s-%s%s.mat'),params.savePath,params.dataset(1).saveFile,params.dataset(2).saveFile,params.saveSuffix)
    if params.differenceMatrix.load and exist_(filename,char('file')):
        display_(sprintf_(char('Loading image difference matrix from file %s ...'),filename))
        d=load_(filename)
        results.D=d.D
    else:
        if length_(results.dataset) < 2:
            display_(char('Error: Cannot calculate difference matrix with less than 2 datasets.'))
            return results
        display_(char('Calculating image difference matrix ...'))
        n=size_(results.dataset(1).preprocessing,2)
        m=size_(results.dataset(2).preprocessing,2)
        D=zeros_(n,m,char('single'))
        for i in arange_(1,n).reshape(-1):
            d=results.dataset(2).preprocessing - repmat_(results.dataset(1).preprocessing(arange_(),i),1,m)
            D[i,:]=sum_(abs_(d)) / n
        results.D=D
        if params.differenceMatrix.save:
            save_(filename,char('D'))
    return results
def doFindMatches_(results=None,params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 2-[results,params].count(None)+len(args)

    filename=sprintf_(char('%s/matches-%s-%s%s.mat'),params.savePath,params.dataset(1).saveFile,params.dataset(2).saveFile,params.saveSuffix)
    if params.matching.load and exist_(filename,char('file')):
        display_(sprintf_(char('Loading matchings from file %s ...'),filename))
        m=load_(filename)
        results.matches=m.matches
    else:
        matches=NaN_(size_(results.DD,2),2)
        display_(char('Searching for matching images ...'))
        params.matching.ds=params.matching.ds + mod_(params.matching.ds,2)
        DD=results.DD
        for N in arange_(params.matching.ds / 2 + 1,size_(results.DD,2) - params.matching.ds / 2).reshape(-1):
            matches[N,:]=findSingleMatch_(DD,N,params)
        if params.matching.save:
            save_(filename,char('matches'))
        results.matches=matches
    return results
def findSingleMatch_(DD=None,N=None,params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 3-[DD,N,params].count(None)+len(args)

    move_min=params.matching.vmin * params.matching.ds
    move_max=params.matching.vmax * params.matching.ds
    move=arange_(move_min,move_max)
    v=move / params.matching.ds
    idx_add=repmat_([arange_(0,params.matching.ds)],size_(v,2),1)
    idx_add=floor_(idx_add.dot(repmat_(v.T,1,length_(idx_add))))
    n_start=N - params.matching.ds / 2
    x=repmat_([arange_(n_start,n_start + params.matching.ds)],length_(v),1)
    score=zeros_(1,size_(DD,1))
    DD=matlabarray([[DD],[inf_(1,size_(DD,2))]])
    y_max=size_(DD,1)
    xx=(x - 1) * y_max
    for s in arange_(1,size_(DD,1)).reshape(-1):
        y=min_(idx_add + s,y_max)
        idx=xx + y
        score[s]=min_(sum_(DD[idx],2))
    min_value,min_idx=min_(score,nargout=2)
    window=arange_(max_(1,min_idx - params.matching.Rwindow / 2),min_(length_(score),min_idx + params.matching.Rwindow / 2))
    not_window=setxor_(arange_(1,length_(score)),window)
    min_value_2nd=min_(score[not_window])
    match=matlabarray([[min_idx + params.matching.ds / 2],[min_value / min_value_2nd]])
    return match
def doPreprocessing_(params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 1-[params].count(None)+len(args)

    for i in arange_(1,length_(params.dataset)).reshape(-1):
        filename=sprintf_(char('%s/preprocessing-%s%s.mat'),params.dataset(i).savePath,params.dataset(i).saveFile,params.saveSuffix)
        if params.dataset(i).preprocessing.load and exist_(filename,char('file')):
            r=load_(filename)
            display_(sprintf_(char('Loading file %s ...'),filename))
            results.dataset(i).preprocessing=r.results_preprocessing
        else:
            p=copy_(params)
            p.dataset=params.dataset(i)
            results.dataset(i).preprocessing=single_(preprocessing_(p))
            if params.dataset(i).preprocessing.save:
                results_preprocessing=single_(results.dataset(i).preprocessing)
                save_(filename,char('results_preprocessing'))
    return results
def preprocessing_(params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 1-[params].count(None)+len(args)

    display_(sprintf_(char('Preprocessing dataset %s, indices %d - %d ...'),params.dataset.name,params.dataset.imageIndices(1),params.dataset.imageIndices(end())))
    n=length_(params.dataset.imageIndices)
    m=params.downsample.size(1) * params.downsample.size(2)
    if not isempty_(params.dataset.crop):
        c=params.dataset.crop
        m=(c[3] - c[1] + 1) * (c[4] - c[2] + 1)
    images=zeros_(m,n,char('uint8'))
    j=1
    for i in params.dataset.imageIndices.reshape(-1):
        filename=sprintf_(char('%s/%s%05d%s%s'),params.dataset.imagePath,params.dataset.prefix,i,params.dataset.suffix,params.dataset.extension)
        img=imread_(filename)
        if params.DO_GRAYLEVEL:
            img=rgb2gray_(img)
        if params.DO_RESIZE:
            img=imresize_(img,params.downsample.size,params.downsample.method)
        if not isempty_(params.dataset.crop):
            img=img[params.dataset.crop(2):params.dataset.crop(4),params.dataset.crop(1):params.dataset.crop(3)]
        if params.DO_PATCHNORMALIZATION:
            img=patchNormalize_(img,params)
        if params.DO_SAVE_PREPROCESSED_IMG:
            pass
        images[:,j]=img[:]
        j=j + 1
    return images
def openSeqSLAM_(params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 1-[params].count(None)+len(args)

    results=matlabarray([])
    if params.DO_PREPROCESSING:
        results=doPreprocessing_(params)
    if params.DO_DIFF_MATRIX:
        results=doDifferenceMatrix_(results,params)
    if params.DO_CONTRAST_ENHANCEMENT:
        results=doContrastEnhancement_(results,params)
    else:
        if params.DO_DIFF_MATRIX:
            results.DD=results.D
    if params.DO_FIND_MATCHES:
        results=doFindMatches_(results,params)
    return results
def patchNormalize_(img=None,params=None,*args,**kwargs):
    varargin = cellarray(args)
    nargin = 2-[img,params].count(None)+len(args)

    s=params.normalization.sideLength
    n=arange_(1,size_(img,1) + 1,s)
    m=arange_(1,size_(img,2) + 1,s)
    for i in arange_(1,length_(n) - 1).reshape(-1):
        for j in arange_(1,length_(m) - 1).reshape(-1):
            p=img[n[i]:n[i + 1] - 1,m[j]:m[j + 1] - 1]
            pp=p[:]
            if params.normalization.mode != 0:
                pp=double_(pp)
                img[n[i]:n[i + 1] - 1,m[j]:m[j + 1] - 1]=127 + reshape_(round_((pp - mean_(pp)) / std_(pp)),s,s)
            else:
                f=255.0 / double_(max_(pp) - min_(pp))
                img[n[i]:n[i + 1] - 1,m[j]:m[j + 1] - 1]=round_(f * (p - min_(pp)))
    return img

def demo_():
    params = defaultParameters_()    
    
    # Nordland spring dataset
    ds = AttributeDict()
    ds.name = 'spring'
    #ds.imagePath = '../datasets/nordland/64x32-grayscale-1fps/spring'
    ds.imagePath = 'C:/Scientific Software/SeqSLAM/datasets/nordland/64x32-grayscale-1fps/spring'    
    ds.prefix='images-'
    ds.extension='.png'
    ds.suffix=''
    ds.imageSkip = 100     # use every n-nth image
    ds.imageIndices = range(1, 35700, ds.imageSkip)    
    ds.savePath = 'results'
    ds.saveFile = '%s-%d-%d-%d' % (ds.name, ds.imageIndices[0], ds.imageSkip, ds.imageIndices[-1])
    
    ds.preprocessing = AttributeDict()
    ds.preprocessing.save = 1
    ds.preprocessing.load = 1
    #ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop=[]
    
    spring=ds


    # Nordland winter dataset
    ds.name = 'winter'
    #ds.imagePath = '../datasets/nordland/64x32-grayscale-1fps/winter'
    ds.imagePath = 'C:/Scientific Software/SeqSLAM/datasets/nordland/64x32-grayscale-1fps/winter'       
    ds.saveFile = '%s-%d-%d-%d' % (ds.name, ds.imageIndices[0], ds.imageSkip, ds.imageIndices[-1])
    # ds.crop=[5 1 64 32]
    ds.crop=[]
    
    winter=ds        

    params.dataset = [spring, winter]

    # load old results or re-calculate?
    params.differenceMatrix.load = 0
    params.contrastEnhanced.load = 0
    params.matching.load = 0
    
    # where to save / load the results
    params.savePath='results'
              
    
    ## now process the dataset
   
    results = openSeqSLAM_(params)     
    
    ## show some results
    
    # set(0, 'DefaultFigureColor', 'white')
    
    # Now results.matches(:,1) are the matched winter images for every 
    # frame from the spring dataset.
    # results.matches(:,2) are the matching score 0<= score <=1
    # The LARGER the score, the WEAKER the match.
    
    if len(results.matches) > 0:
        m = results.matches[:,0]
        thresh=0.9  # you can calculate a precision-recall plot by varying this threshold
        #m[results.matches[:,1]>thresh] = float('nan')  # remove the weakest matches
        print m
        plt.plot(m[results.matches[:,1]<=thresh],'.')      # ideally, this would only be the diagonal
        plt.title('Matchings')   
        plt.show()    
    else:
        print "Zero matches"  

if __name__ == "__main__":
    demo_()