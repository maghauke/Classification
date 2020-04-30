% Sort training data into the 10 classes
trainvlab = zeros(60000,785);
for i = 1:60000
    trainvlab(i,1:784) = trainv(i,:);
    trainvlab(i,785) = trainlab(i);
end    
trainvsort = sortrows(trainvlab,785);

%%
% Count the amount trainingvectors of each class
counts = zeros(10,1);
for j = 1:60000
    for k = 1:10
        if trainlab(j) == (k-1)
            counts(k) = counts(k) + 1;
        end
    end
end

%%
% Repeating for every class by hand 0:9
M = 64;
[idx0, C0] = kmeans(trainvsort(1:5923,1:784), M,'Display','iter');

%%
% Make labels for the clustered training set [64 rows of 0; 64 rows of 1; ...; 64 rows of 9]
trainvcluster = [C0;C1;C2;C3;C4;C5;C6;C7;C8;C9];

trainvclusterlab = zeros(640,1);
for m = 0:9
    for n = 1:64
        trainvclusterlab(n+(m*64)) = m;
    end
end

%%
% Train clustered model and predict
tic
Knn_cluster = fitcknn(trainvcluster,trainvclusterlab,'NumNeighbors',7,'Distance','euclidean','DistanceWeight', 'inverse');
[predlab_cluster,score_cluster,cost_cluster] = predict(Knn_cluster,testv);
toc