%MODEL_1D  Apply the 1-D model to an input image, beta. This version makes
%use of literal convolutions with the kernel w. It may be helpful for
%understanding the model, but will likely be slower than model_1d_matrix().
% sim_obj = model_1d(beta, T, dt);
%
% INPUTS:
%      beta : an input image from which the model parameters will be inferred
% (optional)
%         T : stopping time for the simulation
%        dt : the size of the time step used for numerical integration
%
% OUTPUTS:
%   sim_obj : a matlab object containing the timecourse for each layer in
%             the network, as well as the convolution kernels that
%             represent the lateral connectivity. Use
%             sim_obj.A(:,:,sim_obj.t==5) to access the state of A at t=5.
%             Similarly for other layers.
%
% AUTHOR:
%   Sebastian Waz, swaz@uci.edu



function sim_obj = model_1d(beta, T, dt)

% Defaults (beta is required argument)
if nargin < 3 || isempty(dt)
    dt = 0.0001;
end
if nargin < 2 || isempty(T)
    T = 100;
end

% Force input to be a row
if size(beta,2)==1
    beta = beta';
end

% Initalize the t for each timestep in the simulation
tt=0:dt:T;

% Create B for all time
B = repmat([beta, zeros(size(beta))], [1, 1, length(tt)]);

% Initialize P (note that P takes as input dB/dt)
P        = nan(size(B));
P(:,:,1) = B(:,:,1);

% Initalize A
A        = nan(size(B));
A(:,:,1) = 0;

% Generate convolution kernel (this represents lateral connectivity)
w = [1/2, -1, 1/2];

% Run the simulation (use Euler's method for numerical integration)
tic
check = 0;
for t = 2:length(tt)
    % Announce progress every 1/4 second
    if toc > check
        clc
        fprintf('At iteration %d of %d.\nTime elapsed: %5.2f seconds.\nCurrent params:\n n=%d\n m=%d\n dt=%d\n T=%d\n', t, length(tt)-1, round(toc, 3), length(beta), 1, dt, T); 
        check = check + 0.25;
    end

    % Iterate A
    dA       = conv_1d(w, A(:,:,t-1)) - B(:,:,t-1) + P(:,:,t-1);
    A(:,:,t) = A(:,:,t-1) + dA * dt;

    % Iterate P
    dP       = conv_1d(w, P(:,:,t-1));
    P(:,:,t) = P(:,:,t-1) + dP * dt;
end

% Done.
% Transpose the vectors because n corresponds to number of columns in text
sim_obj.B = B;
sim_obj.P = P;
sim_obj.A = A;
sim_obj.w = w;
sim_obj.t = tt;
end