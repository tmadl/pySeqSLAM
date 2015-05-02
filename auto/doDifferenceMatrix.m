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