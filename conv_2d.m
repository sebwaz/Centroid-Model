%conv_2d  Perform the 2-D convolution, given A is periodic
% w_ast_A = conv_2d(w,A);
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



function w_ast_A = conv_2d(w,A)
% Expand A, obeying periodicity so that central part of
% convolution with w accounts for periodicity
buf = max(size(w));

% Expand top and bottom
A_top = A(1:buf,:);
A_btm = A((end-buf+1):end,:);
wide_A = [A_btm; A; A_top];

% Expand left and right
A_lft = wide_A(:,1:buf);
A_rgt = wide_A(:,(end-buf+1):end);
wide_A = [A_rgt, wide_A, A_lft];

% Perform the 2D periodic convolution
w_ast_A = conv2(wide_A, w, 'same');
w_ast_A = w_ast_A((1+buf):(end-buf),(1+buf):(end-buf));
end

