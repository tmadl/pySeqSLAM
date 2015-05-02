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
