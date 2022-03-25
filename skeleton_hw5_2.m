function [lambda_top5, k_] = skeleton_hw5_2()
%% Q5.2
%% Load AT&T Face dataset
    img_size = [112,92];   % image size (rows,columns)
    % Load the AT&T Face data set using load_faces()
    %%%%% TODO
    faces = load_faces();
    k = 400;
    %% Compute mean face and the covariance matrix of faces
    % compute X_tilde
    %%%%% TODO
    mean_face = mean(faces);
    X_tilde = faces-mean_face.*ones(400,112*92);
    % Compute covariance matrix using X_tilde
    %%%%% TODO
    Sx = 1/(112 * 92) *( X_tilde'*X_tilde);
    %% Compute the eigenvalue decomposition of the covariance matrix
    %%%%% TODO
    [U,Lambda] = eig(Sx,'vector');
    
    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    %%%%% TODO
    [Lambda,idx] = sort(Lambda,'descend');
    U = U(:,idx);
    %% Compute the principal components: Y
    %%%%% TODO
    W_pca = U(:,1:400);
    Y = W_pca' * (X_tilde');
%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    %%%%% TODO
    imshow(uint8(reshape(faces(120,:),[112,92])));
    % Visualize the mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(uint8(reshape(mean(faces),[112,92])));
%% Q5.2 b) Analysing computed eigenvalues
    warning('off')
    
    % Report the top 5 eigenvalues
     lambda_top5 = Lambda(1:5); %%%%% TODO
    
    % Plot the eigenvalues in from largest to smallest
    k = 1:450;
    figure(2)
    sgtitle('Eigenvalues from largest to smallest')

    % Plot the eigenvalue number k against k
    subplot(1,2,1)
    %%%%% TODO
    scatter(k,Lambda(1:length(k)),'o','filled');
    grid on;
    xlabel('k');
    ylabel('$$\lambda_k$$','interpreter','latex','FontSize',14);
    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %%%%% TODO: Compute eigen fractions
    rho_k = zeros(1,length(k));
    subplot(1,2,2)
    lambda_sum = sum(Lambda);
    %%%%% TODO
    for i = 1:length(k)
       rho_k(i) = sum(Lambda(1:i))/lambda_sum;
    end
    rho_k = round(rho_k,2);
    scatter(k,rho_k,'o','filled');
    xlabel('k');
    ylabel('$$\rho_k$$','interpreter','latex','FontSize',14);
    grid on;
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    k_ = [find(rho_k == 0.51) find(rho_k == 0.75) find(rho_k == 0.9) find(rho_k == 0.95) find(rho_k == 0.99)];
    %%%%% TODO (Hint: ismember())
    % k_ = ?; %%%%% TODO
    [~,k_] = ismember(ef,rho_k);
%% Q5.2 c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = faces(test_img_idx,:);    
    % Compute eigenface coefficients
    %%%% TODO
    X_pca = (mean_face'.*ones(10304,400)) + W_pca*Y;
    K = [0,1,2,k_,400,d];
    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to the mean face

    % visulize and plot in a single figure using subplots the resulating image approximations obtained by adding eigen faces to the mean face.

    %%%% TODO 
    
    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    subplot(1,length(K),1)
    imshow(uint8(reshape(test_img,[112,92])));
    for i = 2:length(K)
       subplot(1,length(K),i)
       imshow(uint8(reshape(mean(faces),[112,92])));
    end
%% Q5.2 d) Principal components capture different image characteristics
%% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of percentile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,~] = size(X);
    %% Compute the mean face and the covariance matrix
    % compute X_tilde
    %%%%% TODO
    mean_image
    % Compute covariance using X_tilde
    %%%%% TODO
    
    %% Compute the eigenvalue decomposition
    %%%%% TODO
    
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    %%%%% TODO
    
    %% Compute principal components
    %%%%% TODO
    
    %% Computing the first 2 pricipal components
    %%%%% TODO

    % finding percentile points
    percentile_vals = [5, 25, 50, 75, 95];
    %%%%% TODO (Hint: Use the provided fucntion - percentile_points())
    
    % Finding the cartesian product of percentile points to find grid corners
    %%%%% TODO

    
    %% Find images whose PCA coordinates are closest to the grid coordinates 
    
    %%%%% TODO

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 100th image
    subplot(1,2,1)
    %%%%% TODO
    
    % Mean face image
    subplot(1,2,2)
    %%%%% TODO

    
    %% Image projections onto principal components and their corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the percentile points and highlight the image points that are closest to the 
    % percentile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)

    xlabel('Principal component 1')
    ylabel('Principal component 2')
    title('Image points closest to percentile grid corners')
    hold off
    
    figure(6)
    sgtitle('Images closest to percentile grid corners')
    hold on
    % Plot the images whose PCA coordinates are closest to the percentile grid 
    % corners. Use subplot to put all images in a single figure in a grid.
    
    %%%%% TODO
    
    hold off    
end