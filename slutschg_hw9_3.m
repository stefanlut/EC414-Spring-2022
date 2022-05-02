%% Homework 9 Problem 3
clear,clc,close all;
load("nn-train.mat");
load("nn-test.mat");
%% 2 Neuron Classifier - Train
model = patternnet(2);
model.divideParam.trainRatio = 1;
model.divideParam.valRatio = 0;
model.divideParam.testRatio = 0;
model = train(model,trainData(:,1:2)',dummyvar(trainData(:,3))');
view(model);
y_train = model(trainData(:,1:2)');
[~,train_scores] = max(y_train);
train_confmat = confusionmat(trainData(:,3),train_scores);
train_ccr = trace(train_confmat)/length(trainData);
train_perf = perform(model,trainData(:,3)',y_train);
figure;
gscatter(trainData(:,1),trainData(:,2),train_scores); grid on;
legend("Predicted Training Class 1","Predicted Training Class 2");
xlabel('X1');
ylabel("X2");
title("Predicted Classes using Training Dataset")
%% 2 Neuron Classifier - Test
y_test = model(testData(:,1:2)');
[~,test_scores] = max(y_test);
test_confmat = confusionmat(testData(:,3),test_scores);
test_ccr = trace(test_confmat)/length(testData);
test_perf = perform(model,testData(:,3)',y_test);
figure;
gscatter(testData(:,1),testData(:,2),test_scores); grid on;
legend("Predicted Test Class 1","Predicted Test Class 2");
xlabel('X1');
ylabel("X2");
title("Predicted Classes using Test Dataset")
%% 10 Neuron Classifier - Train
clearvars -except testData trainData
clc,close all;
model = patternnet(10);
model.divideParam.trainRatio = 1;
model.divideParam.valRatio = 0;
model.divideParam.testRatio = 0;
model = train(model,trainData(:,1:2)',dummyvar(trainData(:,3))');
view(model);
y_train = model(trainData(:,1:2)');
[~,train_scores] = max(y_train);
train_confmat = confusionmat(trainData(:,3),train_scores);
train_ccr = trace(train_confmat)/length(trainData);
train_perf = perform(model,trainData(:,3)',y_train);
figure;
gscatter(trainData(:,1),trainData(:,2),train_scores); grid on;
legend("Predicted Training Class 1","Predicted Training Class 2");
xlabel('X1');
ylabel("X2");
title("Predicted Classes using Training Dataset")
%% 10 Neuro Classifier - Test
y_test = model(testData(:,1:2)');
[~,test_scores] = max(y_test);
test_confmat = confusionmat(testData(:,3),test_scores);
test_ccr = trace(test_confmat)/length(testData);
test_perf = perform(model,testData(:,3)',y_test);
figure;
gscatter(testData(:,1),testData(:,2),test_scores); grid on;
legend("Predicted Test Class 1","Predicted Test Class 2");
xlabel('X1');
ylabel("X2");
title("Predicted Classes using Test Dataset")
