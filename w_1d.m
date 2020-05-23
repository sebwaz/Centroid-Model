%W_1D  Generate the 1-D convolution matrix. W*A here is equivalent to W \ast A in the text
% w = w_1d(n);
%
% INPUTS:
%         n : the number of columns in the input image
%
% OUTPUTS:
%         w : The convolution matrix
%
% AUTHOR:
%   Sebastian Waz, swaz@uci.edu



function w = w_1d(n)

% Connection weights
recurrent = -1;
lateral = 0.5;

% Generate convolution matrix
% (matrix product W * A is equivalent to W \ast A in the text)
w = zeros(2*n);
w(logical(eye(size(w)))) = lateral;
w = circshift(w,1);
w(logical(eye(size(w)))) = recurrent;
w = circshift(w,1);
w(logical(eye(size(w)))) = lateral;
w = circshift(w,-1);

% Use matlab sparse matrix
w = sparse(w);
end

