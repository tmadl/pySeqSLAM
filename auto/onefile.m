
function params=defaultParameters()



    % switches
    params.DO_PREPROCESSING = 1;
    params.DO_RESIZE        = 0;
    params.DO_GRAYLEVEL     = 1;
    params.DO_PATCHNORMALIZATION    = 1;
    params.DO_SAVE_PREPROCESSED_IMG = 0;
    params.DO_DIFF_MATRIX   = 1;
    params.DO_CONTRAST_ENHANCEMENT  = 1;
    params.DO_FIND_MATCHES  = 1;


    % parameters for preprocessing
    params.downsample.size = [32 64];  % height, width
    params.downsample.method = 'lanczos3';
    params.normalization.sideLength = 8;
    params.normalization.mode = 1;
            
    
    % parameters regarding the matching between images
    params.matching.ds = 10; 
    params.matching.Rrecent=5;
    params.matching.vmin = 0.8;
    params.matching.vskip = 0.1;
    params.matching.vmax = 1.2;  
    params.matching.Rwindow = 10;
    params.matching.save = 1;
    params.matching.load = 1;
    
    % parameters for contrast enhancement on difference matrix  
    params.contrastEnhancement.R = 10;

    % load old results or re-calculate? save results?
    params.differenceMatrix.save = 1;
    params.differenceMatrix.load = 1;
    
    params.contrastEnhanced.save = 1;
    params.contrastEnhanced.load = 1;
    
    % suffix appended on files containing the results
    params.saveSuffix='';

end

function results = doDifferenceMatrix(results, params)
        
    filename = sprintf('%s/difference-%s-%s%s.mat', params.savePath, params.dataset(1).saveFile, params.dataset(2).saveFile, params.saveSuffix);  

    if params.differenceMatrix.load && exist(filename, 'file')
        display(sprintf('Loading image difference matrix from file %s ...', filename));

        d = load(filename);
        results.D = d.D;                                    
    else
        if length(results.dataset)<2
            display('Error: Cannot calculate difference matrix with less than 2 datasets.');
            return;
        end

        display('Calculating image difference matrix ...');
  %      h_waitbar = waitbar(0,'Calculating image difference matrix');

        n = size(results.dataset(1).preprocessing,2);
        m = size(results.dataset(2).preprocessing,2);

        D = zeros(n,m, 'single');   

	% parfor!
        for i = 1:n            
            d = results.dataset(2).preprocessing - repmat(results.dataset(1).preprocessing(:,i),1,m);
            D(i,:) = sum(abs(d))/n;            
            %waitbar(i/n);
        end
        results.D=D;
        
        % save it
        if params.differenceMatrix.save                   
            save(filename, 'D');
        end

     %   close(h_waitbar);
    end    


end

   
function results = doFindMatches(results, params)       
     
    filename = sprintf('%s/matches-%s-%s%s.mat', params.savePath, params.dataset(1).saveFile, params.dataset(2).saveFile, params.saveSuffix);  
     
    if params.matching.load && exist(filename, 'file')
        display(sprintf('Loading matchings from file %s ...', filename));
        m = load(filename);
        results.matches = m.matches;          
    else
    
        matches = NaN(size(results.DD,2),2);
        
        display('Searching for matching images ...');
        % h_waitbar = waitbar(0, 'Searching for matching images.');
        
        % make sure ds is dividable by two
        params.matching.ds = params.matching.ds + mod(params.matching.ds,2);
        
        DD = results.DD;
        %parfor!
        for N = params.matching.ds/2+1 : size(results.DD,2)-params.matching.ds/2
            matches(N,:) = findSingleMatch(DD, N, params);
            %   waitbar(N / size(results.DD,2), h_waitbar);
        end
               
        % save it
        if params.matching.save
            save(filename, 'matches');
        end
        
        results.matches = matches;
    end
end


%%
function match = findSingleMatch(DD, N, params)


    % We shall search for matches using velocities between
    % params.matching.vmin and params.matching.vmax.
    % However, not every vskip may be neccessary to check. So we first find
    % out, which v leads to different trajectories:
        
    move_min = params.matching.vmin * params.matching.ds;    
    move_max = params.matching.vmax * params.matching.ds;    
    
    move = move_min:move_max;
    v = move / params.matching.ds;
    
    idx_add = repmat([0:params.matching.ds], size(v,2),1);
   % idx_add  = floor(idx_add.*v);
    idx_add = floor(idx_add .* repmat(v', 1, length(idx_add)));
    
    % this is where our trajectory starts
    n_start = N - params.matching.ds/2;    
    x= repmat([n_start : n_start+params.matching.ds], length(v), 1);    
    
    
    score = zeros(1,size(DD,1));    
    
    % add a line of inf costs so that we penalize running out of data
    DD=[DD; inf(1,size(DD,2))];
            
    y_max = size(DD,1);        
    xx = (x-1) * y_max;
    
    for s=1:size(DD,1)           
        y = min(idx_add+s, y_max);                
        idx = xx + y;
        score(s) = min(sum(DD(idx),2));
    end
    
    
    % find min score and 2nd smallest score outside of a window
    % around the minimum 
    
    [min_value, min_idx] = min(score);
    window = max(1, min_idx-params.matching.Rwindow/2):min(length(score), min_idx+params.matching.Rwindow/2);
    not_window = setxor(1:length(score), window);
    min_value_2nd = min(score(not_window));
    
    match = [min_idx + params.matching.ds/2; min_value / min_value_2nd];    
end



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


function img = patchNormalize(img, params)
    s = params.normalization.sideLength;    
    
    n = 1:s:size(img,1)+1;
    m = 1:s:size(img,2)+1;
        
    for i=1:length(n)-1
        for j=1:length(m)-1
            p = img(n(i):n(i+1)-1, m(j):m(j+1)-1);
            
            pp=p(:);
            
            if params.normalization.mode ~=0
                pp=double(pp);
                img(n(i):n(i+1)-1, m(j):m(j+1)-1)  = 127+reshape(round((pp-mean(pp))/std(pp)), s, s);
            else            
                f = 255.0/double(max(pp) - min(pp));            
                img(n(i):n(i+1)-1, m(j):m(j+1)-1) = round(f * (p-min(pp)));                        
            end
            
            
        end
    end

end


function results = doContrastEnhancement(results, params)
        
    filename = sprintf('%s/differenceEnhanced-%s-%s%s.mat', params.savePath, params.dataset(1).saveFile, params.dataset(2).saveFile, params.saveSuffix);  
    
    if params.contrastEnhanced.load && exist(filename, 'file')                
        display(sprintf('Loading contrast-enhanced image distance matrix from file %s ...', filename));
        dd = load(filename);
        results.DD = dd.DD;

    else

        display('Performing local contrast enhancement on difference matrix ...');
    %    h_waitbar = waitbar(0,'Local contrast enhancement on difference matrix');

        DD = zeros(size(results.D), 'single');

        D=results.D;
        % parfor!
        for i = 1:size(results.D,1)
            a=max(1, i-params.contrastEnhancement.R/2);
            b=min(size(D,1), i+params.contrastEnhancement.R/2);                                                        
            v = D(a:b, :);
            DD(i,:) = (D(i,:) - mean(v)) ./ std(v);                                          
           % waitbar(i/size(results.D, 1));                
        end  
                      
        % let the minimum distance be 0
        results.DD = DD-min(min(DD));

        % save it?
        if params.contrastEnhanced.save                         
            DD = results.DD;
            save(filename, 'DD');
        end

  %      close(h_waitbar);

    end

end



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






function demo()

%% first set the parameters

    % start with default parameters
    params = defaultParameters();    
    
    % Nordland spring dataset
    ds.name = 'spring';
    ds.imagePath = '../datasets/nordland/64x32-grayscale-1fps/spring';    
    ds.prefix='images-';
    ds.extension='.png';
    ds.suffix='';
    ds.imageSkip = 100;     % use every n-nth image
    ds.imageIndices = 1:ds.imageSkip:35700;    
    ds.savePath = 'results';
    ds.saveFile = sprintf('%s-%d-%d-%d', ds.name, ds.imageIndices(1), ds.imageSkip, ds.imageIndices(end));
    
    ds.preprocessing.save = 1;
    ds.preprocessing.load = 1;
    %ds.crop=[1 1 60 32];  % x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop=[];
    
    spring=ds;


    % Nordland winter dataset
    ds.name = 'winter';
    ds.imagePath = '../datasets/nordland/64x32-grayscale-1fps/winter';       
    ds.saveFile = sprintf('%s-%d-%d-%d', ds.name, ds.imageIndices(1), ds.imageSkip, ds.imageIndices(end));
    % ds.crop=[5 1 64 32];
    ds.crop=[];
    
    winter=ds;        

    params.dataset(1) = spring;
    params.dataset(2) = winter;

    % load old results or re-calculate?
    params.differenceMatrix.load = 0;
    params.contrastEnhanced.load = 0;
    params.matching.load = 0;
    
    % where to save / load the results
    params.savePath='results';
              
    
%% now process the dataset
   
    results = openSeqSLAM(params);          
    
%% show some results
    
    close all;
    % set(0, 'DefaultFigureColor', 'white');
    
    % Now results.matches(:,1) are the matched winter images for every 
    % frame from the spring dataset.
    % results.matches(:,2) are the matching score 0<= score <=1
    % The LARGER the score, the WEAKER the match.
    
    m = results.matches(:,1);
    thresh=0.9;  % you can calculate a precision-recall plot by varying this threshold
    m(results.matches(:,2)>thresh) = NaN;  % remove the weakest matches
    plot(m,'.');      % ideally, this would only be the diagonal
    title('Matchings');                 
end