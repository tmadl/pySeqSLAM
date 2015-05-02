%

% Copyright 2013, Niko Sünderhauf
% niko@etit.tu-chemnitz.de
%
% This file is part of OpenSeqSLAM.
%
% OpenSeqSLAM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% OpenSeqSLAM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with OpenSeqSLAM.  If not, see <http://www.gnu.org/licenses/>.

function results = openSeqSLAM(params)        
    
    results=[];
        
    % try to allocate 3 CPU cores (adjust to your machine) for parallel
    % processing
    try
        if matlabpool('size')==0
            matlabpool 3
        end
    catch
        display('No parallel computing toolbox installed.');
    end
    
    
    
    % begin with preprocessing of the images
    if params.DO_PREPROCESSING                
        results = doPreprocessing(params);        
    end
        
    
    % image difference matrix             
    if params.DO_DIFF_MATRIX
        results = doDifferenceMatrix(results, params);
    end
    
    
    % contrast enhancement
    if params.DO_CONTRAST_ENHANCEMENT
        results = doContrastEnhancement(results, params);        
    else
        if params.DO_DIFF_MATRIX
            results.DD = results.D;
        end
    end
    
    
    % find the matches
    if params.DO_FIND_MATCHES
        results = doFindMatches(results, params);
    end
            
end









