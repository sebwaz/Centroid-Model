%MODEL_2D  Apply the 2-D model to an input image, beta. This version does
%not recast layers as vectors and makes use of literal convolutions with
%the kernel w. It may helpful for understanding the model, but will likely
%be slower than model_2d_matrix().
% sim_obj = model_2d(beta, T, dt);
%
% INPUTS:
%      beta : an input image from which the model parameters will be inferred
% (optional)
%         T : stopping time for the simulation
%        dt : the size of the time step used for numerical integration
%
% OUTPUTS:
%   sim_obj : a matlab object containing the final state of each layer in
%             the network, as well as the convolution kernels that
%             represent the lateral connectivity
%
% AUTHOR:
%   Sebastian Waz, swaz@uci.edu



function sim_obj = model_2d(beta, T, dt)

% Defaults (beta is required argument)
if nargin < 3 || isempty(dt)
    dt = 0.0001;
end
if nargin < 2 || isempty(T)
    T = 1000;
end

% Initalize the t for each timestep in the simulation
tt=0:dt:T;

% Note: to reduce memory consumption, B, A, P, HM, and VM
% will only be instantiated for one timepoint and will be updated over time.

% Create B
B = [beta, zeros(size(beta)); zeros(size(beta)), zeros(size(beta))];

% Initialize P (note that P takes as input dB/dt)
P = B;

% Initalize A
A = zeros(size(B));

% Initialize HM and VM
HM = A;
VM = A;

% Generate convolution kernels (these represent lateral connectivity)
w    = [   0,  1/4,    0; ...
         1/4,   -1,  1/4; ...
           0,  1/4,    0 ];
      
w_hm = [   0,  1/2,    0; ...
           0,   -1,    0; ...
           0,  1/2,    0 ];

w_vm = [   0,    0,    0; ...
         1/2,   -1,  1/2; ...
           0,    0,    0 ];

% Run the simulation (use Euler's method for numerical integration)
tic
check = 0;
for t = 2:length(tt)
    % Announce progress every 1/4 second
    if toc > check
        clc
        fprintf('At iteration %d of %d.\nTime elapsed: %5.2f seconds.\nCurrent params:\n n=%d\n m=%d\n dt=%d\n T=%d\n', t, length(tt)-1, round(toc, 3), size(beta,1), size(beta,2), dt, T); 
        check = check + 0.25;
    end

    % Iterate A
    dA = conv_2d(w, A) - B + P;
    A  = A + dA * dt;

    % Iterate P
    dP = conv_2d(w, P);
    P  = P + dP * dt;
    
    % Iterate HM
    dHM = conv_2d(w_hm, HM) + dA;
    HM  = HM + dHM * dt;
    
    % Iterate VM
    dVM = conv_2d(w_vm, VM) + dA;
    VM  = VM + dVM * dt;
end

% Done.
sim_obj.B = B;
sim_obj.P = P;
sim_obj.A = A;
sim_obj.HM = HM;
sim_obj.VM = VM;
sim_obj.w = w;
sim_obj.w_hm = w_hm;
sim_obj.w_vm = w_vm;
end