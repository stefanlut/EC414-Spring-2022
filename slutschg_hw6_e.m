%% Homework 6
% Stefan Lutschg
% U27846111
clear,clc,close all

load('iris.mat');
t_max = 6000;
lambda = 0.1;
t = 1:1:t_max;
%% 6.3a)
figure(1);
Y = [Y_label_train;Y_label_test];
histogram(Y);
xticks(cell2mat(Label_legend(:,1)));
xticklabels(Label_legend(:,2));
X = [X_data_train; X_data_test];

%% 6.3b)
x_ext = [X_data_train ones(105,1)]';
Theta = zeros(5,3);
f_0 = lambda * sum(vecnorm(Theta,2));
gradient_0 = 2 * lambda * Theta;
probabilities = zeros(3,1);
for i = t
    j = randi(size(X_data_train,1));
    denominator = sum([exp(Theta(:,1)' * x_ext(:,j)) exp(Theta(:,2)' * x_ext(:,j)) exp(Theta(:,3)' * x_ext(:,j))]);
    for k = 1:3
       probabilities(k) = exp(Theta(:,k)' * x_ext(:,j))/denominator;
       v(k,:) = 2 * lambda * Theta(:,k) + 45*(probabilities(k) - (k == Y_label_train(j)))*x_ext(:,j);
    end
    Theta = Theta - (0.01/i)*(v');
end
fj = 0;
for i = 1:45
    
end