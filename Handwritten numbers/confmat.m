% Make and plot confusion matrix
C = confusionmat(testlab,predlab, 'Order', [0 1 2 3 4 5 6 7 8 9]);

cm = confusionchart(C);
cm.Title = 'Handwritten number classification with an k-NN-based classifier (k=3) using euclidean distance';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
%print('cm','-depsc')

%%
%Clustering
C_cluster = confusionmat(testlab,predlab_cluster, 'Order', [0 1 2 3 4 5 6 7 8 9]);

cm_cluster = confusionchart(C_cluster);
cm_cluster.Title = 'Handwritten number classification with an k-NN-based classifier (k=3) using clustered templates';
cm_cluster.RowSummary = 'row-normalized';
cm_cluster.ColumnSummary = 'column-normalized';
%print('cm_cluster','-depsc')

%%
%Error Rate
tot_err = 0;
S = sum(C_cluster,2);
for i = 1:10
    tot_err = tot_err + (S(i)-C_cluster(i,i))/S(i);
end

error_rate_cluster = tot_err/10*100;
disp(error_rate_cluster);