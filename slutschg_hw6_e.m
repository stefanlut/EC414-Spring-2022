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

%% 6.3b) through 6.3e)
x_ext_train = [X_data_train ones(105,1)]';
x_ext_test = [X_data_test ones(45,1)]';
Theta = zeros(5,3);
y_predtest = zeros(45,1);

probabilities = zeros(3,1);
t_prime = 1:1:300;
plot_g_theta = zeros(1,300);
g_theta_new_plot = zeros(t_max,1);
fj = zeros(45,1);
y_pred = zeros(105,1);
CCR_train = zeros(t_max,1);
CCR_test = CCR_train;
for i = t
    f_0 = lambda * sum(vecnorm(Theta,2));
    gradient_0 = 2 * lambda * Theta;
    j = randi(size(X_data_train,1));
    denominator = sum([exp(Theta(:,1)' * x_ext_train(:,j)) exp(Theta(:,2)' * x_ext_train(:,j)) exp(Theta(:,3)' * x_ext_train(:,j))]);
    for k = 1:3
       probabilities(k) = exp(Theta(:,k)' * x_ext_train(:,j))/denominator;
       if(probabilities(k) < 1e-10)
            probabilities(k) = 1e-10;
       end
       v(k,:) = 2 * lambda * Theta(:,k) + 105*(probabilities(k) - (k == Y_label_train(j)))*x_ext_train(:,j);
    end
    Theta = Theta - (0.01/i)*(v');
    for l = 1:105
        fj(l) = log(sum([exp(Theta(:,1)' * x_ext_train(:,l)) exp(Theta(:,2)' * x_ext_train(:,l)) exp(Theta(:,3)' * x_ext_train(:,l))])) - sum([(1 == Y_label_train(l))*Theta(:,1)'*x_ext_train(:,l) (2 == Y_label_train(l))*Theta(:,2)'*x_ext_train(:,l) (3 == Y_label_train(l))*Theta(:,3)'*x_ext_train(:,l)]);
    
    end
    g_theta = f_0 + sum(fj);
    g_theta_new_plot(i) = g_theta;
    
    for g = 1:105
        [~,yhat] = max([Theta(:,1)'*x_ext_train(:,g), Theta(:,2)'*x_ext_train(:,g), Theta(:,3)'*x_ext_train(:,g)]);
        ypred_train(g) = yhat;
    end
    for x = 1:45
        [~,yhat] = max([Theta(:,1)'*x_ext_test(:,x), Theta(:,2)'*x_ext_test(:,x), Theta(:,3)'*x_ext_test(:,x)]);
        ypred_test(x) = yhat;
    end
    confusionmatrix_train = confusionmat(Y_label_train,ypred_train);
    CCR_train(i) = trace(confusionmatrix_train)/105;
    confusionmatrix_test = confusionmat(Y_label_test,ypred_test);
    CCR_test(i) = trace(confusionmatrix_test)/45;
    if(mod(i,20) == 0 )
        plot_g_theta(i/20) = g_theta / 105;
        plot_ccr_train(i/20) = CCR_train(i);
        plot_ccr_test(i/20) = CCR_test(i);
    end
end
%% GRAPHS
figure;
plot(t_prime,plot_g_theta,'LineWidth',2);
xlabel('t''','FontSize',14);
ylabel('$$ \frac{1}{n}g(\Theta) $$','FontSize',20,'Interpreter','latex');
figure;
plot(t_prime,plot_ccr_train,'LineWidth',2);
ylabel('CCR','FontSize',14);
xlabel('t''','FontSize',14);
title('CCR of Training Set','FontSize',14);
figure;
plot(t_prime,plot_ccr_test,'LineWidth',2);
ylabel('CCR','FontSize',14);
xlabel('t''','FontSize',14);
title('CCR of Test Set','FontSize',14);