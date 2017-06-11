from parameters import defaultParameters
from utils import AttributeDict
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os

from seqslam import *

def demo():

    # set the parameters

    # start with default parameters
    params = defaultParameters()    
    
    # Nordland spring dataset
    ds = AttributeDict()
    ds.name = 'train'
    ds.imagePath = '../datasets/loam/train'
    
    ds.prefix='left'
    ds.extension='.jpg'
    ds.suffix=''
    ds.imageSkip = 1     # use every n-nth image
    ds.imageIndices = range(10, 310, ds.imageSkip)    
    ds.savePath = 'results'
    ds.saveFile = '%s-%d-%d-%d' % (ds.name, ds.imageIndices[0], ds.imageSkip, ds.imageIndices[-1])
    
    ds.preprocessing = AttributeDict()
    ds.preprocessing.save = 1
    ds.preprocessing.load = 0 #1
    #ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop=[]
    
    train=ds

    ds2 = deepcopy(ds)
    # Nordland winter dataset
    ds2.name = 'test'
    ds2.imageSkip = 1     # use every n-nth image
    ds2.imageIndices = range(10, 310, ds.imageSkip)    

    test_name = 'test_T15_R1.5'
    ds2.imagePath = '../datasets/loam/'+test_name
    ds2.saveFile = '%s-%d-%d-%d' % (ds2.name, ds2.imageIndices[0], ds2.imageSkip, ds2.imageIndices[-1])
    # ds.crop=[5 1 64 32]
    ds2.crop=[]
    
    test=ds2      

    params.dataset = [train, test]

    # load old results or re-calculate?
    params.differenceMatrix.load = 0
    params.contrastEnhanced.load = 0
    params.matching.load = 0
    
    # where to save / load the results
    params.savePath='results'
              
    ## now process the dataset
    ss = SeqSLAM(params)  
    t1=time.time()
    results = ss.run()
    t2=time.time()          
    print "time taken: "+str(t2-t1)
    
    ## show some results
    if len(results.matches) > 0:
        m = results.matches[:,0] # The LARGER the score, the WEAKER the match.
        thresh=2  # you can calculate a precision-recall plot by varying this threshold
        m[results.matches[:,1]>thresh] = np.nan # remove the weakest matches
        plt.plot(m,'.')      # ideally, this would only be the diagonal
        plt.title('Matchings')   
        plt.title('Matching '+ test_name)
        plt.savefig(test_name+'.jpg')
    else:
        print "Zero matches"          


if __name__ == "__main__":
    demo()
