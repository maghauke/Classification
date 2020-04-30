% Find best K-value

K_error = zeros(20,1);
K_confmat = [];

for k = 1:20
    Knn = fitcknn(X,Y,'NumNeighbors',k,'Distance','euclidean','DistanceWeight', 'inverse');
    [predlab,score,cost] = predict(Knn,testv);
    C_temp = confusionmat(testlab,predlab, 'Order', [0 1 2 3 4 5 6 7 8 9]);
    K_confmat = [K_confmat; C_temp];
    
    temp_err = 0;
    S = sum(C_temp,2);
    for i = 1:10
        temp_err = temp_err + (S(i)-C_temp(i,i))/S(i);
    end
    K_error(k) = temp_err;

end

%%
% Plot of accuracy vs K-values
figure()
hold on
plot(1:20,K_error)
plot(3,K_error(3),'r*')
grid on
title('Model accuracy with different K-values')
print()