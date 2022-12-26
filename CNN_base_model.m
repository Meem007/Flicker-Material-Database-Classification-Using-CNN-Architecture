
%% Cleaning
 clear all;
 clc;
 warning('off');
 
%% CNN Deep Neural Network
% Load the deep sample data as an image datastore. 
imds = imageDatastore('image', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

pixelRange = [-60 60];
imageAugmenter = imageDataAugmenter("RandRotation", [-45  45],'RandScale',[0.5 1.5],...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain, 'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation,'ColorPreprocessing','gray2rgb');


% Define the convolutional neural network architecture.
layers = [
% Image Input Layer An imageInputLayer 
    imageInputLayer([224 224 3])
% Convolutional Layer 
convolution2dLayer(3,8,'Padding','same')
% Batch Normalization 
    batchNormalizationLayer
% ReLU Layer The batch
    reluLayer
% Max Pooling Layer  
    % More values means less weights
    maxPooling2dLayer(4,'Stride',4)
    %------------------------------
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
% Fully Connected Layer (Number of Classes) 
    fullyConnectedLayer(10)
% Softmax Layer 
    softmaxLayer
% Classification Layer The final layer 
    classificationLayer];
% Specify the training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
% Train the network 
[trainedNet, traininfo]= trainNetwork(augimdsTrain,layers,options);

%% Classify Validation Images
[YPred,probs] = classify(trainedNet,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
plotconfusion(imdsValidation.Labels,YPred)











