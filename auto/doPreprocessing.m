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
function results = doPreprocessing(params)
    for i = 1:length(params.dataset)

        % shall we just load it?
        filename = sprintf('%s/preprocessing-%s%s.mat', params.dataset(i).savePath, params.dataset(i).saveFile, params.saveSuffix);                
        if params.dataset(i).preprocessing.load && exist(filename, 'file');           
            r = load(filename);
            display(sprintf('Loading file %s ...', filename));
            results.dataset(i).preprocessing = r.results_preprocessing;
        else
            % or shall we actually calculate it?
            p = params;    
            p.dataset=params.dataset(i);
            results.dataset(i).preprocessing = single(preprocessing(p));

            if params.dataset(i).preprocessing.save                
                results_preprocessing = single(results.dataset(i).preprocessing);
                save(filename, 'results_preprocessing');
            end                
        end
    end               
end





%%
function images = preprocessing(params)
    

    display(sprintf('Preprocessing dataset %s, indices %d - %d ...', params.dataset.name, params.dataset.imageIndices(1), params.dataset.imageIndices(end)));
   % h_waitbar = waitbar(0,sprintf('Preprocessing dataset %s, indices %d - %d ...', params.dataset.name, params.dataset.imageIndices(1), params.dataset.imageIndices(end)));
    
    % allocate memory for all the processed images
    n = length(params.dataset.imageIndices);
    m = params.downsample.size(1)*params.downsample.size(2); 
    
    if ~isempty(params.dataset.crop)
        c = params.dataset.crop;
        m = (c(3)-c(1)+1) * (c(4)-c(2)+1);
    end
    
    images = zeros(m,n, 'uint8');
    j=1;
    
    % for every image ....
    for i = params.dataset.imageIndices
        filename = sprintf('%s/%s%05d%s%s', params.dataset.imagePath, ...
            params.dataset.prefix, ...
            i, ...
            params.dataset.suffix, ...
            params.dataset.extension);
        
        img = imread(filename);
        
        % convert to grayscale
        if params.DO_GRAYLEVEL
            img = rgb2gray(img);
        end
        
        
        % resize the image
        if params.DO_RESIZE
            img = imresize(img, params.downsample.size, params.downsample.method);
        end
        
        
        % crop the image if necessary
        if ~isempty(params.dataset.crop)
            img = img(params.dataset.crop(2):params.dataset.crop(4), params.dataset.crop(1):params.dataset.crop(3));            
        end
        
        % do patch normalization
        if params.DO_PATCHNORMALIZATION
            img = patchNormalize(img, params);            
        end
        
        
        % shall we save the result?
        if params.DO_SAVE_PREPROCESSED_IMG
            
        end
                    
        images(:,j) = img(:);   
        j=j+1;
        
       % waitbar((i-params.dataset.imageIndices(1)) / (params.dataset.imageIndices(end)-params.dataset.imageIndices(1)));
        
    end
    
   % close(h_waitbar);

end