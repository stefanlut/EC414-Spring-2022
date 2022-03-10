%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2022
% HW 4
% Stefan Lütschg (slutschg@bu.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];

% Generate dataset (i) 

lambda1 = 1;
lambda2 = 0.25;
theta = 0*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,1);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(theta),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii), (iii), and (iv)
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(ii)
theta = 1 * pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);
subplot(2,2,2);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%(iii)
theta = 2 * pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);
subplot(2,2,3);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(2),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%(iv)
lambda1 = 0.25;
lambda2 = 1;
theta = 1 * pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);
subplot(2,2,4);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal_power;
    noise_power_array(i) = noise_power;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;subplot(131)
scatter(phi_array,signal_power_array,'o','fill');
grid;axis equal; hold on;
xlabel('\phi (radians)','FontSize',14);ylabel('signal power','FontSize',14);
findpeaks(signal_power_array,phi_array);
[~,LOCS] = findpeaks(signal_power_array,phi_array);
maxPhi1 = LOCS;
legend('Signal Power','Line','Maximum Signal Power','FontSize',14);

subplot(132)
scatter(phi_array,noise_power_array,'o','fill');
grid on; axis equal;hold on;
xlabel('\phi (radians)','FontSize',14);ylabel('noise power','FontSize',14);
findpeaks(noise_power_array,phi_array);
[~,LOCS] = findpeaks(noise_power_array,phi_array);
maxPhi2 = LOCS;
legend('Noise Power','Line','Maximum Noise Power','FontSize',14);

subplot(133)
scatter(phi_array,snr_array,'o','fill');
grid on; axis equal;hold on;
xlabel('\phi (radians)','FontSize',14);ylabel('SNR','FontSize',14);
findpeaks(snr_array,phi_array);
[~,LOCS] = findpeaks(snr_array,phi_array);
maxPhi3 = LOCS;
legend('SNR','Line','Maximum SNR','FontSize',14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 0:pi/6:pi/3
    phi = i;
    signal_noise_snr(X, Y, phi, true);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];
theta = 1 * pi/6;
lambda1 = 1;
lambda2 = 0.25;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
% Compute the LDA solution by writing and invoking a function named LDA 

w_LDA = LDA(X,Y);
X1 = X(:,Y == 1);
X2 = X(:,Y == 2);
mu1x = mean(X1,2);
mu2x = mean(X2,2);
difference = mu2x-mu1x;
% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
quiver(w_LDA,difference,2,'LineWidth',2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create CCR vs b plot
n = n1+n2;
X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
scatter(b_array,ccr_array,'o','filled');
grid;hold on;
ax = gca;
ax.FontSize = 14;
xlabel('b','FontSize',14);
ylabel('Correct Classification Rate','FontSize',14);
[maxCCR,idx] = max(ccr_array);
b_max = b_array(idx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%
% Inputs:
%
% n1 = number of class 1 examples
% n2 = number of class 2 examples
% mu1 = 2 by 1 class 1 mean vector
% mu2 = 2 by 1 class 2 mean vector
% theta = orientation of eigenvectors of common 2 by 2 covariance matrix shared by both classes
% lambda1 = first eigenvalue of common 2 by 2 covariance matrix shared by both classes
% lambda2 = second eigenvalue of common 2 by 2 covariance matrix shared by both classes
% 
% Outputs:
%
% X = a 2 by (n1 + n2) matrix with first n1 columns containing class 1
% feature vectors and the last n2 columns containing class 2 feature
% vectors
%
% Y = a 1 by (n1 + n2) matrix with the first n1 values equal to 1 and the 
% last n2 values equal to 2
u1 = [cos(theta) sin(theta)]';
u2 = [sin(theta) -cos(theta)]';
U = [u1 u2];
Lambda = [lambda1 0;0 lambda2];
S = U * Lambda * U';

X = zeros(2,n1+n2);
X(:,1:n1) = mvnrnd(mu1,S,n1)';
X(:,n1+1:n1+n2) = mvnrnd(mu2,S,n2)';
Y = zeros(1,n1 + n2);
Y(1:n1) = 1;
Y(n1+1:n1+n2) = 2;

end

function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = length(X(:,1:50));
n2 = length(X(:,51:end));
PHI = [cos(phi) sin(phi)]';
mu1x = mean(X(:,1:50),2);
mu2x = mean(X(:,51:end),2);
X_projected_phi_class1 = (PHI'*mu1x);
X_projected_phi_class2 = (PHI'*mu2x);
signal = (X_projected_phi_class2 - X_projected_phi_class1)^2;
% ...
diffmatrix1 = 0;
for i = 1:length(X(:,1:50))
    diffmatrix1 = diffmatrix1 + ((X(:,i) - mu1x) * (X(:,i) - mu1x)');
end
Sx1 = diffmatrix1/n1;
diffmatrix2 = 0;
for i = 1:100
   diffmatrix2 = diffmatrix2 + ((X(:,50+i) - mu2x) * (X(:,50+i) - mu2x)'); 
end
Sx2 = diffmatrix2/n2;
Variance1 = PHI'*Sx1*PHI;
Variance2 = PHI'*Sx2*PHI;
p1 = n1/(n1+n2);
p2 = n2/(n1+n2);
%Sxavg = p1*Sx1 + p2*Sx2;
noise = p1*Variance1 + p2*Variance2;
snr = signal/noise;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity(X_projected_phi_class1);
    plot(pdf1,z1,'LineWidth',2)
    hold on;
    [pdf2,z2] = ksdensity(X_projected_phi_class2);
    plot(pdf2,z2,'LineWidth',2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title(['Estimated class density estimates of data projected along \phi = ' num2str(phi/(pi/6)) ' \times \pi/6. Ground-truth \phi = \pi/6'])
end

end

function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1 = X(:,Y == 1);
X2 = X(:,Y == 2);
n1 = length(X1);
n2 = length(X2);
n = n1 + n2;
mu1x = mean(X1,2);
mu2x = mean(X2,2);
diffmatrix1 = zeros(2,2);
diffmatrix2 = zeros(2,2);
for i = 1:n1
    diffmatrix1 = diffmatrix1 + ((X1(:,i) - mu1x) * (X1(:,i) - mu1x)');
end
Sx1 = diffmatrix1/n1;
for i = 1:n2
    diffmatrix2 = diffmatrix2 + ((X2(:,i) - mu1x) * (X2(:,i) - mu1x)');
end
Sx2 = diffmatrix2/n2;
p1 = n1/n;
p2 = n2/n;
Sxavg = p1*Sx1 + p2*Sx2;
w_LDA = Sxavg\(mu2x-mu1x);
end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = w_LDA' * X + b;
hwbx = zeros(1,length(result));
n = length(Y);
for i = 1:length(result)
    if(result(i) <=0)
    hwbx(i) = 1;
    else
    hwbx(i) = 2;
    end

end
conf_mat = confusionmat(Y,hwbx);
ccr = trace(conf_mat)/n;

end