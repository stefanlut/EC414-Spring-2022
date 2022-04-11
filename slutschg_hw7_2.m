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