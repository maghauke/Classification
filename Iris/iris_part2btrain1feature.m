close all
clear all
clc
%% The Iris task part 2b training

x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');

M = length(x1all);

N = 30;         % Training set size

% sep_len_index = 1;
% sep_wid_index = 2;
% pet_len_index = 3;
pet_wid_index = 4;      % can easily test for pet_len_index, by changing to 3.

x1_training = x1all(1:N,pet_wid_index);   % Training set for class 1   
x2_training = x2all(1:N,pet_wid_index);   % Training set for class 2
x3_training = x3all(1:N,pet_wid_index);   % Training set for class 3

x1_testing = x1all(N+1:end,pet_wid_index);   % Test set for class 1   
x2_testing = x2all(N+1:end,pet_wid_index);   % Test set for class 2
x3_testing = x3all(N+1:end,pet_wid_index);   % Test set for class 3


x_training = [x1_training; x2_training; x3_training]; % Total training set
x_testing = [x1_testing; x2_testing; x3_testing]; % Total testing set

C = 3;                              % Number of classes: Setosa, Versicolor, Virginica
D = size(x_training,2);             % Input feature dimension
W0 = eye(C,D);                      % Weighting matrix
w0 = zeros(1,C);                    % Initial offset
W0 = [W0 w0.'];                     % Weighting matrix with offset
X = [x_training.'; ones(1,N*C)];    % Input data for training
T = [kron(ones(1,N), [1 0 0].') ... % Targets
     kron(ones(1,N), [0 1 0].') ... 
     kron(ones(1,N), [0 0 1].')];
sigmoid = @(x) (1./(1+exp(-x)));     
gk = @(xk,W) sigmoid(W*xk);


% Training
alpha = 0.01;   % learning rate
gradient = @(W)MSE_grad(X,T,W,gk);      % MSE gradient
[W,n] = gradient_descent(gradient,W0,alpha);      % Trained  weighting  matrix W, n # iterations

% Testing
n_test = M-N;
Xtest = [x_testing'; ones(1, size(x_testing,1))];                   % Test cases
Ttest = [repelem(1,n_test), repelem(2,n_test), repelem(3,n_test)];  % True solution
[~,classified_classes] = max(W*Xtest);                              % Classes given after training

% Finding error rate and confusion matrix for testing data
error_test = classified_classes~=Ttest;
error_rate_test = sum(error_test)/n_test
figure(1)
subplot(2,1,1)
confusion_test = confusionchart(Ttest,classified_classes)
confusion_test.Title = 'Confusion matrix for test set, with one features';
confusion_test.RowSummary = 'row-normalized';
confusion_test.ColumnSummary = 'column-normalized';

% Finding error rate and confusion matrix for training data
Ttraining = [repelem(1,N), repelem(2,N), repelem(3,N)];
[~,classified_tclasses] = max(W*X);

error_training = classified_tclasses~=Ttraining;

error_rate_training = sum(error_training)/N
subplot(2,1,2)
confusion_training = confusionchart(Ttraining, classified_tclasses)
confusion_training.Title = 'Confusion matrix for training set, with one features';
confusion_training.RowSummary = 'row-normalized';
confusion_training.ColumnSummary = 'column-normalized';
% print('confusion_m_2b_1l','-depsc')

error_general = abs(error_rate_training-error_rate_test)