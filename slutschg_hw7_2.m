%% Homework 7
% EC 414 Spring 2022
% Stefan Lutschg U27846111
tic;
clear,clc,close all;
load('iris.mat');
t_max = 2e5;
C = 1.2;
t = 1:1:t_max;
X = [X_data_train(:,2) X_data_train(:,4);
     X_data_test(:,2) X_data_test(:,4)];
Y = [Y_label_train; Y_label_test];
d = size(X,2);
X_train_ext = [X(1:105,:) ones(105,1)]';
X_test_ext = [X(106:end,:) ones(45,1)]';
n = length(Y_label_train);

Y_train_pair1 = [Y_label_train(1:35); -1*ones(70,1)];
%Class 1(+1) & Class 2 (-1)
Y_test_pair1 = [Y_label_test(1:15); -1*ones(30,1)];

Y_train_pair2 = [-1*ones(70,1); (1/3)*Y_label_train(71:end)];
%Class 1 (-1) & Class 3 (+1)
Y_test_pair2 = [-1*Y_label_test(1:15); -1*ones(15,1); (1/3)*Y_label_test(31:end)];

Y_train_pair3 = [-1*ones(35,1); Y_label_train(36:70); -1*ones(35,1)];
%Class 2 (+1) & Class 3 (-1)
Y_test_pair3 = [-1*ones(15,1); Y_label_test(16:30); -1*ones(15,1)];

g_pair1 = 0;
g_pair2 = 0;
g_pair3 = 0;
t_plot = 1:1:200;
plot_g_pair1 = zeros(200,1);
plot_g_pair2 = zeros(200,1);
plot_g_pair3 = zeros(200,1);
y_pred_train1 = zeros(n,1);
y_pred_train2 = y_pred_train1;
y_pred_train3 = y_pred_train1;
plot_ccr_trainpair1 = zeros(200,1);
plot_ccr_trainpair2 = zeros(200,1);
plot_ccr_trainpair3 = zeros(200,1);
y_pred_test1 = zeros(45,1);
y_pred_test2 = y_pred_test1;
y_pred_test3 = y_pred_test1;
plot_ccr_testpair1 = zeros(200,1);
plot_ccr_testpair2 = zeros(200,1);
plot_ccr_testpair3 = zeros(200,1);
%% Algorithm - 1st classifier
theta1 = zeros(d+1,1);
% Pair 1
figure(1);
subplot(131);
sgtitle('$$\frac{1}{n}g(\bf{\theta})$$','Interpreter','latex');
for i = t
   j = randi(n);
   yj = Y_train_pair1(j);
   xj = X_train_ext(:,j);
   v = [theta1(1:d); 0] - (n*C*yj*xj * (yj*theta1'*xj < 1));
   st = 0.5/i;
   theta1 = theta1 - st*v;
   
   if(~mod(i,1000))
       f_0 = 0.5*vecnorm(theta1(1:d));
       g_pair1 = f_0;
       for m = 1:n
           g_pair1 = g_pair1 + C*max(0,1-Y_train_pair1(m)*theta1'*X_train_ext(:,m));
           y_pred_train1(m) = sign(theta1'*X_train_ext(:,m));
               
       end
       for k = 1:45
          y_pred_test1(k) =sign(theta1'*X_test_ext(:,k));

       end
       train_confusionmat = confusionmat(Y_train_pair1,y_pred_train1);
       test_confusionmat = confusionmat(Y_test_pair1,y_pred_test1);
       plot_g_pair1(i/1000) = g_pair1/n;
       plot_ccr_trainpair1(i/1000) = trace(train_confusionmat)/n;
       plot_ccr_testpair1(i/1000) = trace(test_confusionmat)/45;
   end
end
plot(t_plot,plot_g_pair1,'LineWidth',2);
title("Class 1 & Class 2");
ylabel('$$\frac{1}{n}g(\bf{\theta})$$','Interpreter','latex')
xlabel('$$t'' = \frac{t}{1000}$$','Interpreter','latex');
%% Algorithm - 2nd classifier
% Pair 2
theta2 = zeros(d+1,1);
subplot(132);
for i = t
   j = randi(n);
   yj = Y_train_pair2(j);
   xj = X_train_ext(:,j);
   v = [theta2(1:d); 0] - length(X_train_ext)*C*yj*xj * (yj*theta2'*xj < 1);
   st = 0.5/i;
   theta2 = theta2 - st*v;
   
   if(~mod(i,1000))
       f_0 = 0.5*vecnorm(theta2(1:d));
       g_pair2 = f_0;
       for j = 1:n
           g_pair2 = g_pair2 + C*max(0,1-Y_train_pair2(j)*theta2'*X_train_ext(:,j));
           y_pred_train2(j) = sign(theta2'*X_train_ext(:,j));

       end
       for k = 1:45
           y_pred_test2(k) = sign(theta2'*X_test_ext(:,k));
       end
       train_confusionmat = confusionmat(Y_train_pair2,y_pred_train2);
       test_confusionmat = confusionmat(Y_test_pair2,y_pred_test2);
       plot_g_pair2(i/1000) = g_pair2/n;
       plot_ccr_trainpair2(i/1000) = trace(train_confusionmat)/n;
       plot_ccr_testpair2(i/1000) = trace(test_confusionmat)/45;
   end
end
plot(t_plot,plot_g_pair2,'LineWidth',2);
title("Class 1 & Class 3");
ylabel('$$\frac{1}{n}g(\bf{\theta})$$','Interpreter','latex')
xlabel('$$t'' = \frac{t}{1000}$$','Interpreter','latex');
%% Algorithm - 3rd classifier
% Pair 3
theta3 = zeros(d+1,1);
subplot(133);
for i = t
   j = randi(n);
   yj = Y_train_pair3(j);
   xj = X_train_ext(:,j);
   v = [theta3(1:d); 0] - length(X_train_ext)*C*yj*xj * (yj*theta3'*xj < 1);
   st = 0.5/i;
   theta3 = theta3 - st*v;
   
   if(~mod(i,1000))
       f_0 = 0.5*vecnorm(theta3(1:d));
       g_pair3 = f_0;
       for j = 1:n
           g_pair3 = g_pair3 + C*max(0,1-Y_train_pair3(j)*theta3'*X_train_ext(:,j));
           y_pred_train3(j) = sign(theta3'*X_train_ext(:,j));
       end
       for k = 1:45
           y_pred_test3(k) = sign(theta3'*X_test_ext(:,k));
       end
       train_confusionmat = confusionmat(Y_train_pair3,y_pred_train3);
       test_confusionmat = confusionmat(Y_test_pair3,y_pred_test3);
       plot_g_pair3(i/1000) = g_pair3/n;
       plot_ccr_trainpair3(i/1000) = trace(train_confusionmat)/n;
       plot_ccr_testpair3(i/1000) = trace(test_confusionmat)/45;
   end
end
plot(t_plot,plot_g_pair3,'LineWidth',2);
title("Class 2 & Class 3");
ylabel('$$\frac{1}{n}g(\bf{\theta})$$','Interpreter','latex')
xlabel('$$t'' = \frac{t}{1000}$$','Interpreter','latex');
%% Algorithm - CCR
figure(2);
subplot(321)
sgtitle('CCR');
plot(t_plot,plot_ccr_trainpair1,'LineWidth',2);
title("Train: Class 1 & Class 2");
subplot(322)
plot(t_plot,plot_ccr_testpair1,'LineWidth',2);
title("Test: Class 1 & Class 2");
subplot(323)
plot(t_plot,plot_ccr_trainpair2,'LineWidth',2);
title("Train: Class 2 & Class 3");
subplot(324)
plot(t_plot,plot_ccr_testpair2,'LineWidth',2);
title("Test: Class 1 & Class 3");
subplot(325)
plot(t_plot,plot_ccr_trainpair3,'LineWidth',2);
title("Train: Class 2 & Class 3");
subplot(326)
plot(t_plot,plot_ccr_testpair3,'LineWidth',2);
title("Test: Class 2 & Class 3");
toc;