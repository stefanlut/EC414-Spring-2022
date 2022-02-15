% EC 414 Introduction to Machine Learning
% Spring 2022
% Homework 2
% by Stefan Lütschg
% U27846111
% Nearest Neighbor Classifier
%
% Problem 2.5 a, b, c, d


clc, clear, close all

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain);
ax = gca;
ax.FontSize = 20;
grid on;
% label axis and include title
xlabel('X_1','FontSize',14,'FontWeight','bold')
ylabel('X_2','FontSize',14,'FontWeight','bold')
title('Training Data','FontSize',14)


%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);
% compute probabilities of being in class 2 for each point on grid
distances10 = zeros(Ntest,Ntrain);
for i = 1:Ntest
    for j = 1:Ntrain
        distances10(i,j) = sqrt((Xtrain(j,1) - Xtest(i,1))^2 + (Xtrain(j,2) - Xtest(i,2))^2);
     
    end
    
end
dist_sorted = sort(distances10,2,'ascend');
closest_10 = dist_sorted(:,1:10); % Smallest 10 distances
count_of_twos = zeros(Ntest,1);
for i = 1:Ntest
    for j = 1:Ntrain
        for k = 1 : K
            if distances10(i,j) == closest_10(i,k) && ytrain(j) == 2
                count_of_twos(i) = count_of_twos(i) + 1;
            end
        end
    end
end
probabilities_2 = count_of_twos / K;
% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities_2,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('X_1','FontSize',20,'FontWeight','bold')
ylabel('X_2','FontSize',20,'FontWeight','bold')
title('$\hat{p}(y = 2 | \textbf{x})$','FontSize',20,'interpreter','latex')
ax = gca;
ax.FontSize = 20;


% repeat steps above for class 3 below
count_of_threes = zeros(Ntest,1);
for i = 1:Ntest
    for j = 1:Ntrain
        for k = 1 : K
            if distances10(i,j) == closest_10(i,k) && ytrain(j) == 3
                count_of_threes(i) = count_of_threes(i) + 1;
            end
        end
    end
end
probabilities3 = count_of_threes / K;
figure;
class3ProbonGrid = reshape(probabilities3,size(Xgrid));
contourf(Xgrid,Ygrid,class3ProbonGrid);
colorbar;
xlabel('X_1','FontSize',20,'FontWeight','bold')
ylabel('X_2','FontSize',20,'FontWeight','bold')
title('$\hat{p}(y = 3 | \textbf{x})$','FontSize',20,'interpreter','latex')
ax = gca;
ax.FontSize = 20;
%% c) Class label predictions
K = 1 ; % K = 1 case

% compute predictions 
min_distances = min(distances10);
%ypred = 
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
    %ypred = 


    % compute confusion matrix
    conf_mat = confusionmat(Ygrid(:), ypred);
    % from confusion matrix, compute CCR
    %CCR = 
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

% plot CCR values for k = 1,3,5,7,9,11
% label x/y axes and include title
