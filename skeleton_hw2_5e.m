% EC 414 Introduction to Machine Learning
% Spring 2022
% Homework 2
% by (fill in name)
%
% Nearest Neighbor Classifier
%
% Problem 2.5e

clc, clear

fprintf("==== Loading data_mnist_train.mat\n");
load("data_mnist_train.mat");
fprintf("==== Loading data_mnist_test.mat\n");
load("data_mnist_test.mat");

% show test image
imshow(reshape(X_train(200,:), 28,28)')

% determine size of dataset
[Ntrain, dims] = size(X_train);
[Ntest, ~] = size(X_test);

% precompute components

% Note: To improve performance, we split our calculations into
% batches. A batch is defined as a set of operations to be computed
% at once. We split our data into batches to compute so that the 
% computer is not overloaded with a large matrix.
batch_size = 500;  % fit 4 GB of memory
num_batches = Ntest / batch_size;


% Using (x - y) * (x - y)' = x * x' + y * y' - 2 x * y'
for bn = 1:num_batches
  batch_start = 1 + (bn - 1) * batch_size;
  batch_stop = batch_start + batch_size - 1;
  
  % calculate cross term
  
  % compute euclidean distance
  
  fprintf("==== Doing 1-NN classification for batch %d\n", bn);
  % find minimum distance for k = 1
  
end

% compute confusion matrix
conf_mat = 
% compute CCR from confusion matrix
ccr = 
