% EC 414 Introduction to Machine Learning
% Spring 2022
% Homework 2
% by Stefan Lütschg
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
batch_size = 250;  % fit 4 GB of memory
num_batches = Ntest / batch_size;


%% Using (x - y) * (x - y)' = x * x' + y * y' - 2 x * y'
distances_k_1 = zeros(Ntest,1);

idx = zeros(Ntest,1);
ypred = zeros(Ntest,1);
for bn = 1:num_batches
  batch_start = 1 + (bn - 1) * batch_size;
  batch_stop = batch_start + batch_size - 1;
  
  % calculate cross term
 
  
  cross_term = 2*X_test(batch_start:batch_stop,:)*X_train' + sum(X_test(batch_start:batch_stop,:).^2,2) + sum(X_train.^2,2)';
  % compute euclidean distance
  batch_distances = sqrt(cross_term);
  
  fprintf("==== Doing 1-NN classification for batch %d\n", bn);
  % find minimum distance for k = 1
    for i = 0:batch_size-1
    [min_distancesI, idxI] = min(batch_distances(i+1,:));
    distances_k_1(i+batch_start) = min_distancesI;
    idx(i+batch_start) = idxI;
    ypred(i+batch_start) = Y_train(idx(i+batch_start));
    end
  %distances_k_1(batch_start:batch_stop,:) = min(batch_distances,[],2);
end
%%

% compute confusion matrix
conf_mat = confusionmat(Y_test,ypred);
% compute CCR from confusion matrix
ccr = trace(conf_mat)/dims;
