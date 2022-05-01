%% Homework 8 Problem 2
% Stefan Lutschg U27846111
clear, clc, close all;
load("kernel-svm-2rings.mat");
t_max = 1000;
t = 1:1:t_max;
n = length(x);
C = 256/n;
sigma = 0.5;
K = x'*x;
for i = 1:n
    for j = 1:n
    K_rbf(i,j) = exp(-(1/(2*sigma^2))*norm(x(:,i)-x(:,j))^2);
    end
end
%% Algorithm
psi = zeros(n + 1,1);
temp = [K_rbf zeros(200,1)];
temp = [temp;zeros(1,201)];
t_plot = 1:1:t_max/10;
g_plot = zeros(1,length(t_plot));
for i = t
    j = randi(n);
    Kj = [K_rbf(:,j); 1];
    yj = y(j);
    v = temp*psi - n*C*yj*Kj * (yj*psi'*Kj < 1);
    st = 0.256/i;
    psi = psi - st*v;

    if(~mod(i,10))
       f0 = 0.5*psi'*temp*psi;
       g = f0;
       for k = 1:n
           g = g + C*max(0, 1 - yj*psi'*[K_rbf(:,k); 1]);
           y_pred(k) = sign(psi'*[K_rbf(:,k); 1]);
       end
       g = g/n;
       g_plot(i/10) = g/n;
       confustionmatrix = confusionmat(y,y_pred);
       ccr = trace(confustionmatrix);
       ccr_plot(i/10) = ccr/n;

    end
end

%% Graphs
figure;
plot(t_plot,g_plot,'LineWidth',2);
title("$$\frac{1}{n}g(\Psi)$$",'Interpreter','latex','FontSize',20);
xlabel('t'' = 10t','FontSize',20);
figure;
plot(t_plot,ccr_plot,'LineWidth',2);
title("Correct Classification Rate",'FontSize',20);
xlabel('t'' = 10t','FontSize',20);
figure;
gscatter(x(1,:),x(2,:),y); hold on;
xlabel("X1",'FontSize',20);
ylabel("X2",'FontSize',20);
%contour(K_rbf)