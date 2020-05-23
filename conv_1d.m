%CONV_2D  Perform the 1-D convolution, given A is periodic
% w_ast_A = conv_1d(w,A);
%
% INPUTS:
%         w : the convolution kernel, represented as a matrix
%         A : the matrix that will be convolved with 
%
% OUTPUTS:
%         w_ast_A : the result of convolution of w and A on periodic domain
%
% AUTHOR:
%   Sebastian Waz, swaz@uci.edu



function w_ast_A = conv_1d(w,A)
% Expand A, obeying periodicity so that central part of
% convolution with w accounts for periodicity
buf = length(w);

% Expand left and right
A_lft = A(:,1:buf);
A_rgt = A(:,(end-buf+1):end);
wide_A = [A_rgt, A, A_lft];

% Perform the 1D periodic convolution
w_ast_A = conv(wide_A, w, 'same');
w_ast_A = w_ast_A((1+buf):(end-buf));
end

