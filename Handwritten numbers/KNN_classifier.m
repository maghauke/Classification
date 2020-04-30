% Design KNN classifier
X = trainv;
Y = trainlab;
K = 3;

tic
Knn = fitcknn(X,Y,'NumNeighbors',K,'Distance','euclidean','DistanceWeight', 'inverse');
[predlab,score,cost] = predict(Knn,testv);
toc
