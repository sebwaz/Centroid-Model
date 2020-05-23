%W_2D  Generate the 2-D convolution matrix. W*A here is equivalent to W \ast A in the text
% w = w_2d(n, m, torus);
%
% INPUTS:
%         n : the number of columns in the input image
% (optional)
%         m : the number of rows in the input image. If this is not
%             provided, the image will be assumed square.
%     torus : string indicating which torus to generate the convolution
%             matrix for (recall that HM and VM have different connectivity
%             than P and A). If this is not provided, it is assumed that
%             connectivity for P (same as A) is desired.
%
% OUTPUTS:
%         w : The convolution matrix
%
% AUTHOR:
%   Sebastian Waz, swaz@uci.edu



function w = w_2d(n, m, torus)

% Defaults
if nargin < 3 || isempty(torus)
    torus = 'A';
end
if nargin < 2 || isempty(m)
    m = n;
end

% Specify the locations of the connections in the connectivity matrix
% Recurrency
rec_r = 1:4*n*m;
rec_c = 1:4*n*m;

% Horizontal adjacency
lat1_r = 1:4*n*m;
lat1_c = mod((0:4*n*m-1)+2*m, 4*n*m)+1;
lat2_r = lat1_c;
lat2_c = lat1_r;

% Vertical adjacency
lat3_r = 1:4*n*m;
lat3_c = mod(1:4*n*m, 2*n)+1 + repelem(1:2*m, 2*n)*2*n-2*n;
lat4_r = lat3_c;
lat4_c = lat3_r;

% Connection weights
rec = -1*ones(1, 4*n*m);
lat = 0.25*ones(1, 4*n*m);

% Generate convolution matrix
% (matrix product W * A is equivalent to W \ast A in the text)
r     = [rec_r, lat1_r, lat2_r, lat3_r, lat4_r];
c     = [rec_c, lat1_c, lat2_c, lat3_c, lat4_c];
if strcmp(torus, 'HM')
    z = [rec,   0*lat,  0*lat,  2*lat,  2*lat ];
elseif strcmp(torus, 'VM')
    z = [rec,   2*lat,  2*lat,  0*lat,  0*lat ];
else
    z = [rec,   lat,    lat,    lat,    lat   ];
end
w = sparse(r, c, z);
end

