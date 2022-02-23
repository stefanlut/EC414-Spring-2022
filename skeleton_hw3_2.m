% EC 414 - HW 3 - Spring 2022
% K-Means starter code

clear, clc, close all;

%% Generate Gaussian data:
% Add code below:
n = 50;
mu1 = [2,2];
sig1 = 0.02*ones(2);
mu2 = [-2,2];
sig2 = 0.05*ones(2);
mu3 = [0,-3.25];
sig3 = 0.07*ones(2);
X1 = mvnrnd(mu1,sig1,n);
X2 = mvnrnd(mu2,sig2,n);
X3 = mvnrnd(mu3,sig3,n);
Y1 = ones(n,1);
Y2 = 2 * ones(n,1);
Y3 = 3 * ones(n,1);
figure;
gscatter([X1(:,1);X2(:,1);X3(:,1)],[X1(:,2);X2(:,2);X3(:,2)],[Y1;Y2;Y3]);
grid on;

%% Generate NBA data:
% Add code below:
A = readmatrix('NBA_stats_2018_2019.xlsx');
% HINT: readmatrix might be useful here

% Problem 3.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 3 folder on Blackboard.

%% K-Means implementation
% Add code below

K = 3;
mu1_init = [3 3]';
mu2_init = [-4 -1]';
mu3_init = [2 -4]';
MU_init = [mu1_init mu2_init mu3_init]; 

MU_previous = MU_init;
MU_current = MU_init;
BigX = [X1;X2;X3];
% initializations
labels = ones(length([X1(:,1);X2(:,1);X3(:,1)]),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;
min_distances = zeros(3,1);

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)
    x11 = 0;
    x12 = 0;
    x21 = 0;
    x22 = 0;
    x31 = 0;
    x32 = 0;
    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    for i = 1:length(labels)
       min_distances(1) = sqrt((BigX(i,1) - MU_current(1,1))^2 + (BigX(i,2) - MU_current(2,1))^2);
       min_distances(2) = sqrt((BigX(i,1) - MU_current(1,2))^2 + (BigX(i,2) - MU_current(2,2))^2);
       min_distances(3) = sqrt((BigX(i,1) - MU_current(1,3))^2 + (BigX(i,2) - MU_current(2,3))^2);
       [mind, idx] = min(min_distances);
       if idx == 1
           labels(i) = 1;
       elseif idx == 2
           labels(i) = 2;
       elseif idx == 3
           labels(i) = 3;
       end
    end
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    id1 = find(labels == 1);
    id2 = find(labels == 2);
    id3 = find(labels == 3);
    for i = 1:length(id1)
        x11 = x11 + BigX(id1(i),1);
        x12 = x12 + BigX(id1(i),2);
    end
    MU1current = [x11/length(id1) x12/length(id1)]';
    for i = 1:length(id2)
        x21 = x21 + BigX(id2(i),1);
        x22 = x22 + BigX(id2(i),2);
    end
    MU2current = [x21/length(id2) x22/length(id2)]';
    for i = 1:length(id3)
        x31 = x31 + BigX(id3(i),1);
        x32 = x32 + BigX(id3(i),2);
    end
    MU3current = [x31/length(id3) x32/length(id3)]';
    MU_previous = MU_current;
    MU_current = [MU1current MU2current MU3current];
    %% CODE 4 - Check for convergence 
    % Write code below here:
    distance = sqrt((MU_current(1,1) - MU_previous(1,1))^2 + (MU_current(2,1) - MU_previous(2,1))^2 + (MU_current(1,2) - MU_previous(1,2))^2 + (MU_current(2,2) - MU_previous(2,2))^2 + (MU_current(1,3) - MU_previous(1,3))^2 + (MU_current(2,3) - MU_previous(2,3))^2);
    if (distance < convergence_threshold)
        converged=1;
    end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure;
        gscatter(BigX(:,1),BigX(:,2),labels);
        
       
        
        %% If converged, get WCSS metric
        % Add code below
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end



