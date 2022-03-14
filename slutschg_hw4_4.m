%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2022
% HW 4 Problem 4
% Stefan Lütschg (slutschg@bu.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup
clear;close all; clc;load("prostateStnd.mat");
%% 4.4(a)
feature_means = mean(Xtrain,1);
feature_variances = var(Xtrain,1);
label_mean = mean(ytrain);
label_variance = var(ytrain);

%Xtrain_norm = normalize(Xtrain)
Xtrain_norm = zeros(size(Xtrain,1),size(Xtrain,2));
ytrain_norm = zeros(1,length(ytrain));
for i = 1:length(Xtrain)
    for j = 1:length(feature_means)
        Xtrain_norm(i,j) = (Xtrain(i,j) - feature_means(j))/feature_variances(j);
    end
    ytrain_norm(i) = (ytrain(i) - label_mean)/label_variance;
end
Xtest_norm = zeros(size(Xtest,1),size(Xtest,2));
for i = 1:length(Xtest)
    for j = 1:length(feature_means)
        Xtest_norm(i,j) = (Xtest(i,j) - feature_means(j))/feature_variances(j);
    end
end

%% 4.4(b)
lambda = zeros(1,16);
for i = 1:length(lambda)
lambda(i) = exp(i-6);
end
w_ridge_array = zeros(length(lambda),size(Xtrain_norm,2));
for i = 1:length(w_ridge_array)
    w_ridge_array(i,:) = ridgeregression(Xtrain_norm,ytrain_norm,lambda(i),feature_means,label_mean);
end
%% 4.4(c)
% What the hell is the ridge regression coefficient

%% Functions
function w_ridge = ridgeregression(Xtrain_norm,Ytrain_norm,lambda,feature_means,label_mean)
    I = eye(length(feature_means));
    n = length(Ytrain_norm);
    Xprime = Xtrain_norm-feature_means.*ones(size(Xtrain_norm,1),size(Xtrain_norm,2));
    Sx = Xprime'*(Xprime)/n;
    Yprime = Ytrain_norm-(label_mean.*ones(1,length(Ytrain_norm)));
    Sxy = Xprime'*Yprime'/n;
    w_ridge = inv((lambda/n) * I + Sx)*Sxy;
end