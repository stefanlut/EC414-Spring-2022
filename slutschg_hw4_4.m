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
    ytest_norm(i) = (ytest(i) - label_mean)/label_variance;
end

%% 4.4(b)
lambda = zeros(1,16);
for i = 1:length(lambda)
lambda(i) = exp(i-6);
end
w_ridge_array = zeros(length(lambda),size(Xtrain_norm,2));
b_ridge_array = zeros(length(lambda),1);
for i = 1:length(w_ridge_array)
    [w_ridge_array(i,:), b_ridge_array(i)] = ridgeregression(Xtrain_norm,ytrain_norm,lambda(i),feature_means,label_mean);
end
%% 4.4(c)
% What the hell is the ridge regression coefficient
% Oh nvm
ln_array = log(lambda);
figure;
hold on;
grid on;
ax = gca;
ax.XTick = ln_array;
for i = 1:size(w_ridge_array,2)
   plot(ln_array,w_ridge_array(:,i),'LineWidth',2);
end
xlabel('ln(\lambda)','FontSize',14);
ylabel('Ridge Regression Coefficient','FontSize',14);
legend('lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45','FontSize',14);
%% 4.4(d)
%Plot MSE of Training & Test data as ln(lambda) increases
MSE_train_array = zeros(length(lambda),1);
n_train = length(Xtrain);
MSE_test_array = zeros(length(lambda),1);
n_test = length(Xtest);
for i = 1:length(lambda)
    MSE_train_array(i) = 1/n_train *sum((ytrain_norm - w_ridge_array(i,:)*Xtrain_norm' - b_ridge_array(i)).^2);
    MSE_test_array(i) = 1/n_test *sum((ytest_norm - w_ridge_array(i,:)*Xtest_norm' - b_ridge_array(i)).^2);
end
figure;
plot(ln_array,MSE_train_array,'LineWidth',2); hold on;
grid on;
plot(ln_array,MSE_test_array,'LineWidth',2);
xlabel('ln(\lambda)','FontSize',14);
ylabel('$$MSE(\textbf{w},b)$$','FontSize',14,'Interpreter','latex','FontSize',14);
ax = gca;
ax.XTick = ln_array;
legend('Training Data','Test Data','FontSize',14)
%% Functions
function [w_ridge , b_ridge] = ridgeregression(Xtrain_norm,Ytrain_norm,lambda,feature_means,label_mean)
    I = eye(length(feature_means));
    n = length(Ytrain_norm);
    Xprime = Xtrain_norm-feature_means.*ones(size(Xtrain_norm,1),size(Xtrain_norm,2));
    Sx = Xprime'*(Xprime)/n;
    Yprime = Ytrain_norm-(label_mean.*ones(1,length(Ytrain_norm)));
    Sxy = Xprime'*Yprime'/n;
    w_ridge = inv((lambda/n) * I + Sx)*Sxy;
    b_ridge = label_mean - (w_ridge)'*feature_means';
end