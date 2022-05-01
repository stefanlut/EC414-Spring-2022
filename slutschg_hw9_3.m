%% Homework 9 Problem 3
clear,clc,close all;
load("nn-train.mat");
load("nn-test.mat");
% for i = 1:length(trainData)
%     if(trainData(i,3) == 2)
%        trainData(i,3) = -1;
%     end
% end
% for i = 1:length(testData)
%     if(testData(i,3) == 2)
%        testData(i,3) = -1;
%     end
% end
%% 2 Neuron Classifier - Train
model = patternnet(2);
model.divideParam.trainRatio = 1;
model.divideParam.valRatio = 0;
model.divideParam.testRatio = 0;
model = train(model,trainData(:,1:2)',trainData(:,3)');
view(model);
y_train = model(trainData(:,1:2)');

train_perf = perform(model,trainData(:,3)',y_train)
train_classes = vec2ind(y_train);
%% 2 Neuron Classifier - Test
y_test = model(testData(:,1:2)');
test_perf = perform(model,testData(:,3)',y_test)
test_classes = vec2ind(y_test);

%% 10 Neuron Classifier
clearvars -except testData trainData
clc,close all;
model = patternnet(10);
model.divideParam.trainRatio = 1;
model.divideParam.valRatio = 0;
model.divideParam.testRatio = 0;
model = train(model,trainData(:,1:2)',trainData(:,3)');
view(model);
y_train = model(trainData(:,1:2)');
y_test = model(testData(:,1:2)');
train_perf = perform(model,trainData(:,3)',y_train)
test_perf = perform(model,testData(:,3)',y_test)
train_classes = vec2ind(y_train);
test_classes = vec2ind(y_test);
