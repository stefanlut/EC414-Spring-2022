%% Intro
% Stefan Lutschg
% EC 414 Homework 6
% Problem 6.3
clear, clc, close all;
load("iris.mat");
%% 6.3a)
t_max = 6000;
lambda = 0.1;
t = 1:1:t_max;
histogram([Y_label_train;Y_label_test]);
xticks(cell2mat(Label_legend(:,1)));
xticklabels(Label_legend(:,2));
Sx = cov([X_data_train;X_data_test]);


%% 6.3b)
x_ext = [X_data_train ones(105,1)];
Theta = zeros(105,5);
sum_prob_y_x = 0;

for j = 1:size(X_data_train,1)
    numerator = exp();
    denominator = 0;
    exp(Theta(j,:) * x_ext(j,:)')
    for m = 1:size(Theta,1)

    end
end