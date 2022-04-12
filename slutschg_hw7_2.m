%% Homework 7
% EC 414 Spring 2022
% Stefan Lutschg U27846111
clear,clc,close all;
load('iris.mat');
t_max = 2e5;
C = 1.2;
t = 1:1:t_max;
X = [X_data_train(:,2) X_data_train(:,4);
     X_data_test(:,2) X_data_test(:,4)];
Y = [Y_label_train; Y_label_test];
d = size(X,2);
%% Algorithm
X_train_ext = [X(1:105,:) ones(105,1)]';
theta = zeros(d+1,1);
for i = t
   j = randi(length(X_train_ext));
   yj = Y_label_train(j);
   xj = X_train_ext(:,j);
   v = [theta(1:d); 0] - length(X_train_ext)*C*yj*xj * (yj*theta'*xj < 1);
   st = 0.5/i;
   theta = theta - st*v;
end