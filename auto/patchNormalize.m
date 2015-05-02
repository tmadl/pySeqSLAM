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