"""
     OpenSeqSLAM
     Copyright 2013, Niko Sünderhauf Chemnitz University of Technology niko@etit.tu-chemnitz.de

     pySeqSLAM is an open source Python implementation of the original SeqSLAM algorithm published by Milford and Wyeth at ICRA12 [1]. SeqSLAM performs place recognition by matching sequences of images.
     
     [1] Michael Milford and Gordon F. Wyeth (2012). SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights. In Proc. of IEEE Intl. Conf. on Robotics and Automation (ICRA)
"""
from parameters import defaultParameters

from attrdict import AttrDict

def demo():

    # set the parameters

    # start with default parameters
    params = defaultParameters()    
    
    # Nordland spring dataset
    ds = AttrDict()
    ds.name = 'spring';
    ds.imagePath = '../datasets/nordland/64x32-grayscale-1fps/spring';    
    ds.prefix='images-';
    ds.extension='.png';
    ds.suffix='';
    ds.imageSkip = 100;     # use every n-nth image
    ds.imageIndices = range(1, 35700, ds.imageSkip);    
    ds.savePath = 'results';
    ds.saveFile = sprintf('#s-#d-#d-#d', ds.name, ds.imageIndices(1), ds.imageSkip, ds.imageIndices(end));
    
    ds.preprocessing.save = 1;
    ds.preprocessing.load = 1;
    #ds.crop=[1 1 60 32];  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop=[];
    
    spring=ds;


    # Nordland winter dataset
    ds.name = 'winter';
    ds.imagePath = '../datasets/nordland/64x32-grayscale-1fps/winter';       
    ds.saveFile = sprintf('#s-#d-#d-#d', ds.name, ds.imageIndices(1), ds.imageSkip, ds.imageIndices(end));
    # ds.crop=[5 1 64 32];
    ds.crop=[];
    
    winter=ds;        

    params.dataset(1) = spring;
    params.dataset(2) = winter;

    # load old results or re-calculate?
    params.differenceMatrix.load = 0;
    params.contrastEnhanced.load = 0;
    params.matching.load = 0;
    
    # where to save / load the results
    params.savePath='results';
              
    
## now process the dataset
   
    results = openSeqSLAM(params);          
    
## show some results
    
    close all;
    # set(0, 'DefaultFigureColor', 'white');
    
    # Now results.matches(:,1) are the matched winter images for every 
    # frame from the spring dataset.
    # results.matches(:,2) are the matching score 0<= score <=1
    # The LARGER the score, the WEAKER the match.
    
    m = results.matches(:,1);
    thresh=0.9;  # you can calculate a precision-recall plot by varying this threshold
    m(results.matches(:,2)>thresh) = NaN;  # remove the weakest matches
    plot(m,'.');      # ideally, this would only be the diagonal
    title('Matchings');                 
end