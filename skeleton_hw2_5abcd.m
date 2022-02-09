% EC 414 Introduction to Machine Learning
% Spring 2022
% Homework 2
% by (fill in name)
%
% Nearest Neighbor Classifier
%
% Problem 2.5 a, b, c, d


clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()

% label axis and include title
% xlabel('')
% ylabel('')
% title('')


%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);

% compute probabilities of being in class 2 for each point on grid
probabilities = 

% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
% xlabel('')
% ylabel('')
% title('')


% repeat steps above for class 3 below


%% c) Class label predictions
K = 1 ; % K = 1 case

% compute predictions 
ypred = 
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
% xlabel('')
% ylabel('')
% title('')

% repeat steps above for the K=5 case. Include code for this below.

%% d) LOOCV CCR computations

for k = 1:2:11
    % determine leave-one-out predictions for k
    ypred = 


    % compute confusion matrix
    conf_mat = confusionmat(Ygrid(:), ypred);
    % from confusion matrix, compute CCR
    CCR = 
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

% plot CCR values for k = 1,3,5,7,9,11
% label x/y axes and include title
