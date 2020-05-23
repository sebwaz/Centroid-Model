%MODEL_2D_MATRIX  Apply the 2-D model to an input image, beta. This version
%casts everything as matrix operations and can (therefore) make use of
%Matlab's optimizations.
% sim_obj = model_2d_matrix(beta, T, dt, gpu_opt);
%
% INPUTS:
%      beta : an input image from which the model parameters will be inferred
% (optional)
%         T : stopping time for the simulation
%        dt : the size of the time step used for numerical integration
%   gpu_opt : boolean for whether matlab should use GPU to run simulation
%             (this should be reserved for large images; the examples
%             provided should not require this)
%
% OUTPUTS:
%   sim_obj : a matlab object containing the final state of each layer in
%             the network, as well as the convolution matrices used to
%             perform convolution as matrix multiplication
%
% AUTHOR:
%   Sebastian Waz, swaz@uci.edu



function sim_obj = model_2d_matrix(beta, T, dt, gpu_opt)

% Defaults (beta is required argument)
if nargin < 4 || isempty(gpu_opt)
    gpu_opt = false;
end
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
B = reshape(B, [numel(B), 1]);

% Initialize P (note that P takes as input dB/dt)
P = B;

% Initalize A
A = zeros(size(B));

% Initialize HM and VM
HM = A;
VM = A;

% Generate connectivity matrices (to make use of matlab optimizations,
% convolution is expressed as a matrix operation)
w    = w_2d(size(beta, 2), size(beta, 1));
w_hm = w_2d(size(beta, 2), size(beta, 1), 'HM');
w_vm = w_2d(size(beta, 2), size(beta, 1), 'VM');

% Setup GPU optimization if requested
if gpu_opt
    B = gpuArray(B);
    P = gpuArray(P);
    A = gpuArray(A);
    HM = gpuArray(HM);
    VM = gpuArray(VM);
    w = gpuArray(w);
    w_hm = gpuArray(w_hm);
    w_vm = gpuArray(w_vm);
end

% Run the simulation (use Euler's method for numerical integration)
tic
check = 0;
for t = 2:length(tt)
    % Announce progress every 1/4 second
    if toc > check
        clc
        if gpu_opt
            warning('GPU optimization is only appropriate for large images. Slow down may occur for small images.');
        end
        fprintf('At iteration %d of %d.\nTime elapsed: %5.2f seconds.\nCurrent params:\n n=%d\n m=%d\n dt=%d\n T=%d\n', t, length(tt)-1, round(toc, 3), size(beta,1), size(beta,2), dt, T); 
        check = check + 0.25;
    end

    % Iterate A
    dA = w * A - B + P;
    A  = A + dA * dt;

    % Iterate P
    dP = w * P;
    P  = P + dP * dt;
    
    % Iterate HM
    dHM = w_hm * HM + dA;
    HM  = HM + dHM * dt;
    
    % Iterate VM
    dVM = w_vm * VM + dA;
    VM  = VM + dVM * dt;
end

% Wrap up GPU optimization
if gpu_opt
    B = gather(B);
    P = gather(P);
    A = gather(A);
    HM = gather(HM);
    VM = gather(VM);
    w = gather(w);
    w_hm = gather(w_hm);
    w_vm = gather(w_vm);
end

% Done.
sim_obj.B = reshape(B, 2*size(beta));
sim_obj.P = reshape(P, 2*size(beta));
sim_obj.A = reshape(A, 2*size(beta));
sim_obj.HM = reshape(HM, 2*size(beta));
sim_obj.VM = reshape(VM, 2*size(beta));
sim_obj.w = w;
sim_obj.w_hm = w_hm;
sim_obj.w_vm = w_vm;
end