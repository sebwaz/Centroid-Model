%RUN_EXAMPLES   Script that demonstrates how to call the 1-D and 2-D model functions.
%
% AUTHOR:
%   Sebastian Waz, swaz@uci.edu


%% 1-D model example

% Generate a random dot pattern
n = 16;
beta = round(rand(1,n));

% Provide the image as input to the model and run
sim_obj = model_1d_matrix(beta);

% The model_1d_matrix() optimizes the model by using matrix operations
% instead of literal convolutions. The following function uses literal
% convolutions and may be easier to relate to the text:
%sim_obj = model_1d(beta);

% Animate the model's evolution over time
figure;
ybounds = [min(sim_obj.A(:))-1, max(sim_obj.A(:))+1];
for k = 1:5000:size(sim_obj.A, 3)
    plot(1:2*n,sim_obj.A(:,:,k), '-or'); hold on;
    plot(1:2*n, zeros(1,2*n), ':k');
    plot([n+.5, n+.5], ybounds, '--k');
    text(n+1,ybounds(1)+2,'readout \rightarrow');
    text(n,ybounds(1)+2,'\leftarrow input', 'HorizontalAlignment', 'right'); hold off;
    xlim([1,2*n]);
    ylim(ybounds);
    xlabel('Neuron index');
    ylabel('A activation');
    pause(0.01);
end



%% 2-D model examples

% Load the images used in Figures 7 and 8
load('images.mat');

% Provide first image as input to the model and run
beta = images(1:20,1:20,1);
sim_obj = model_2d_matrix(beta);

% Same as above, using literal convolutions
%sim_obj = model_2d(beta);

% Plot (like Figure 7)
figure;
subplot(3,4,1);
imagesc(sim_obj.B);
title('B');

subplot(3,4,5);
P = sim_obj.P;
% Contrast gain in imagesc makes P look less flat than it is. If difference
% between P_k and mean(P) is less than 1e-4, set P_k = mean(P) to flatten.
P(abs(P - mean(P(:))) < 1e-4) = mean(P(:));
imagesc(P);
title('P');

subplot(3,4,6);
imagesc(sim_obj.A);
title('A');

subplot(3,4,3);
imagesc(sim_obj.HM);
title('M^H');

subplot(3,4,11);
imagesc(sim_obj.VM);
title('M^V');

subplot(3,4,8);
imagesc(sim_obj.HM+sim_obj.VM);
title('readout');



% Provide dot cloud image as input to the model and run
beta = images(:,:,2);
sim_obj_dotcloud = model_2d_matrix(beta);

% Provide contour image as input to the model and run
beta = images(:,:,3);
sim_obj_contour = model_2d_matrix(beta);

% Provide noise image as input to the model and run
beta = images(:,:,4);
sim_obj_noise = model_2d_matrix(beta);

% Plot (like Figure 8)
figure;
subplot(3,3,1);
imagesc(sim_obj_dotcloud.B);
title('(a.) dot cloud');
subplot(3,3,2);
imagesc(sim_obj_contour.B);
title('(b.) contour');
subplot(3,3,3);
imagesc(sim_obj_noise.B);
title('(c.) noise');

subplot(3,3,4);
imagesc(sim_obj_dotcloud.A);
subplot(3,3,5);
imagesc(sim_obj_contour.A);
subplot(3,3,6);
imagesc(sim_obj_noise.A);

subplot(3,3,7);
imagesc(sim_obj_dotcloud.HM + sim_obj_dotcloud.VM);
subplot(3,3,8);
imagesc(sim_obj_contour.HM + sim_obj_contour.VM);
subplot(3,3,9);
imagesc(sim_obj_noise.HM + sim_obj_noise.VM);