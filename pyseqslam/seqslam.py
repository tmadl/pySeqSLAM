from utils import AttributeDict
import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
from PIL import Image 
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SeqSLAM():
    params = None
    
    def __init__(self, params):
        self.params = params

    def run(self):
        # begin with preprocessing of the images
        if self.params.DO_PREPROCESSING:
            results = self.doPreprocessing()
        
        # image difference matrix             
        if self.params.DO_DIFF_MATRIX:
            results = self.doDifferenceMatrix(results)
        
        # contrast enhancement
        if self.params.DO_CONTRAST_ENHANCEMENT:
            results = self.doContrastEnhancement(results)        
        else:
            if self.params.DO_DIFF_MATRIX:
                results.DD = results.D
        
        # find the matches
        if self.params.DO_FIND_MATCHES:
            results = self.doFindMatches(results)
        return results
    
    def doPreprocessing(self):
        results = AttributeDict()
        results.dataset = []
        for i in range(len(self.params.dataset)):
            # shall we just load it?
            filename = '%s/preprocessing-%s%s.mat' % (self.params.dataset[i].savePath, self.params.dataset[i].saveFile, self.params.saveSuffix)
            if self.params.dataset[i].preprocessing.load and os.path.isfile(filename):         
                r = loadmat(filename)
                print('Loading file %s ...' % filename)
                results.dataset[i].preprocessing = r.results_preprocessing
            else:
                # or shall we actually calculate it?
                p = deepcopy(self.params)    
                p.dataset = self.params.dataset[i]
                d = AttributeDict()
                d.preprocessing = np.copy(SeqSLAM.preprocessing(p))
                results.dataset.append(d)
    
                if self.params.dataset[i].preprocessing.save:
                    results_preprocessing = results.dataset[i].preprocessing
                    savemat(filename, {'results_preprocessing': results_preprocessing})

        return results
    
    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    
    @staticmethod
    def preprocessing(params):
        print('Preprocessing dataset %s, indices %d - %d ...' % (params.dataset.name, params.dataset.imageIndices[0], params.dataset.imageIndices[-1]))
        # allocate memory for all the processed images
        n = len(params.dataset.imageIndices)
        m = params.downsample.size[0]*params.downsample.size[1] 
        
        if len(params.dataset.crop) > 0:
            c = params.dataset.crop
            m = (c[2]-c[0]+1) * (c[3]-c[1]+1)
        
        images = np.zeros((m,n), 'uint8')
        j=0
        
        # for every image ....
        for i in (params.dataset.imageIndices):
            filename = '%s/%s%04d%s%s' % (params.dataset.imagePath, \
                params.dataset.prefix, \
                i, \
                params.dataset.suffix, \
                params.dataset.extension)
            
            img = Image.open(filename)

            # resize the image
            if params.DO_RESIZE:
                img = img.resize(params.downsample.size, params.downsample.method)

            # convert to grayscale
            if params.DO_GRAYLEVEL:
                #img = img.convert('L') #LA to include alpha       
                img = SeqSLAM.rgb2gray(np.asarray(img))   
            
            img = np.copy(np.asarray(img))
            #img.flags.writeable = True
            
            # crop the image if necessary
            if len(params.dataset.crop) > 0:
                img = img[params.dataset.crop[1]:params.dataset.crop[3], params.dataset.crop[0]:params.dataset.crop[2]]            
            
            # do patch normalization
            if params.DO_PATCHNORMALIZATION:
                img = SeqSLAM.patchNormalize(img, params)            

            # shall we save the result?
            if params.DO_SAVE_PREPROCESSED_IMG:
                pass
            
            images[:,j] = img.flatten(0)
            j += 1
            
        return images
    
    
    @staticmethod
    def patchNormalize(img, params):
        s = params.normalization.sideLength    
        
        n = range(0, img.shape[0]+2, s)
        m = range(0, img.shape[1]+2, s)
            
        for i in range(len(n)-1):
            for j in range(len(m)-1):
                p = img[n[i]:n[i+1], m[j]:m[j+1]]
                
                pp=np.copy(p.flatten(1))
                
                if params.normalization.mode != 0:
                    pp=pp.astype(float)
                    img[n[i]:n[i+1], m[j]:m[j+1]] = 127+np.reshape(np.round((pp-np.mean(pp))/np.std(pp, ddof=1)), (s, s))
                else:
                    f = 255.0/np.max((1, np.max(pp) - np.min(pp)))
                    img[n[i]:n[i+1], m[j]:m[j+1]] = np.round(f * (p-np.min(pp)))
                    
                #print str((n[i], n[i+1], m[j], m[j+1]))
        return img
    
    def getDifferenceMatrix(self, data0preproc, data1preproc):
        # TODO parallelize
        n = data0preproc.shape[1]
        m = data1preproc.shape[1]
        D = np.zeros((n, m))   
    
        #parfor?
        for i in range(n):
            d = data1preproc - np.tile(data0preproc[:,i],(m, 1)).T
            D[i,:] = np.sum(np.abs(d), 0)/n
            
        return D
    
    def doDifferenceMatrix(self, results):
        filename = '%s/difference-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0].saveFile, self.params.dataset[1].saveFile, self.params.saveSuffix)  
    
        if self.params.differenceMatrix.load and os.path.isfile(filename):
            print('Loading image difference matrix from file %s ...' % filename)
    
            d = loadmat(filename)
            results.D = d.D                                    
        else:
            if len(results.dataset)<2:
                print('Error: Cannot calculate difference matrix with less than 2 datasets.')
                return None
    
            print('Calculating image difference matrix ...')
    
            results.D=self.getDifferenceMatrix(results.dataset[0].preprocessing, results.dataset[1].preprocessing)
            
            # save it
            if self.params.differenceMatrix.save:                   
                savemat(filename, {'D':results.D})
            
        return results
    
    def enhanceContrast(self, D):
        # TODO parallelize
        DD = np.zeros(D.shape)
    
        #parfor?
        for i in range(D.shape[0]):
            a=np.max((0, i-self.params.contrastEnhancement.R/2))
            b=np.min((D.shape[0], i+self.params.contrastEnhancement.R/2+1))                                                        
            v = D[a:b, :]
            DD[i,:] = (D[i,:] - np.mean(v, 0)) / np.std(v, 0, ddof=1)  
        
        return DD-np.min(np.min(DD))
    
    def doContrastEnhancement(self, results):
        
        filename = '%s/differenceEnhanced-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0].saveFile, self.params.dataset[1].saveFile, self.params.saveSuffix)  
        
        if self.params.contrastEnhanced.load and os.path.isfile(filename):    
            print('Loading contrast-enhanced image distance matrix from file %s ...' % filename)
            dd = loadmat(filename)
            results.DD = dd.DD
        else:
            print('Performing local contrast enhancement on difference matrix ...')
               
            # let the minimum distance be 0
            results.DD = self.enhanceContrast(results.D)

            # save it?
            if self.params.contrastEnhanced.save:                        
                DD = results.DD
                savemat(filename, {'DD':DD})
                
        return results
    
    def doFindMatches(self, results):
     
        filename = '%s/matches-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0].saveFile, self.params.dataset[1].saveFile, self.params.saveSuffix)  
         
        if self.params.matching.load and os.path.isfile(filename):   
            print('Loading matchings from file %s ...' % filename)
            m = loadmat(filename)
            results.matches = m.matches          
        else:
        
            print('Searching for matching images ...')
            
            # make sure ds is dividable by two
            self.params.matching.ds = self.params.matching.ds + np.mod(self.params.matching.ds,2)
        
            matches = self.getMatches(results.DD)
                   
            # save it
            if self.params.matching.save:
                savemat(filename, {'matches':matches})
            
            results.matches = matches
            
        return results
    
    def getMatches(self, DD):
        # TODO parallelize
        matches = np.nan*np.ones((DD.shape[1],2))    
        # parfor?
        for N in range(self.params.matching.ds/2, DD.shape[1]-self.params.matching.ds/2):
            # find a single match
            
            # We shall search for matches using velocities between
            # params.matching.vmin and params.matching.vmax.
            # However, not every vskip may be neccessary to check. So we first find
            # out, which v leads to different trajectories:
                
            move_min = self.params.matching.vmin * self.params.matching.ds    
            move_max = self.params.matching.vmax * self.params.matching.ds    
            
            move = np.arange(int(move_min), int(move_max)+1)
            v = move.astype(float) / self.params.matching.ds
            
            idx_add = np.tile(np.arange(0, self.params.matching.ds+1), (len(v),1))
            idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)
            
            # this is where our trajectory starts
            n_start = N + 1 - self.params.matching.ds/2    
            x= np.tile(np.arange(n_start , n_start+self.params.matching.ds+1), (len(v), 1))    
            
            #TODO idx_add and x now equivalent to MATLAB, dh 1 indexing
            score = np.zeros(DD.shape[0])    
            
            # add a line of inf costs so that we penalize running out of data
            DD=np.vstack((DD, np.infty*np.ones((1,DD.shape[1]))))
                    
            y_max = DD.shape[0]        
            xx = (x-1) * y_max
            
            flatDD = DD.flatten(1)
            for s in range(1, DD.shape[0]):   
                y = np.copy(idx_add+s)
                y[y>y_max]=y_max     
                idx = (xx + y).astype(int)
                ds = np.sum(flatDD[idx-1],1)
                score[s-1] = np.min(ds)            
            
            # find min score and 2nd smallest score outside of a window
            # around the minimum 
            
            min_idx = np.argmin(score)
            min_value=score[min_idx]
            window = np.arange(np.max((0, min_idx-self.params.matching.Rwindow/2)), np.min((len(score), min_idx+self.params.matching.Rwindow/2)))
            not_window = list(set(range(len(score))).symmetric_difference(set(window))) #xor
            min_value_2nd = np.min(score[not_window])
            
            match = [min_idx + self.params.matching.ds/2, min_value / min_value_2nd]
            matches[N,:] = match
                
        return matches
