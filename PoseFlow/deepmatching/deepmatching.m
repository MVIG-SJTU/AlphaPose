% mex wrapper to compute the 'DeepMatching' between two images.
%
% matches = deepmatching(image1, image2, options)
%
% Images must be HxWx3 single matrices.
% Options is an optional string argument ('' by default). 
% Availalble options are listed when calling deepmatching() without args.
%
% The function returns a matrix with 6 columns, each row being x1 y1 x2 y2 score index.
% (index refers to the local maximum from which the match was retrieved)
%
% Version 1.2.2
%
% Copyright (C) 2014 Jerome Revaud
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>
%  
