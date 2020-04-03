close all
clear all
clc
%% The Iris task part 2 a training

x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');

M = length(x1all);

N = 30;         % Training set size

sep_len_index = 1;
% sep_wid_index = 2;
pet_len_index = 3;
pet_wid_index = 4;

features = 3;


x1_training = x1all(1:N,[sep_len_index pet_len_index:pet_wid_index]);   % Training set for class 1   
x2_training = x2all(1:N,[sep_len_index pet_len_index:pet_wid_index]);   % Training set for class 2
x3_training = x3all(1:N,[sep_len_index pet_len_index:pet_wid_index]);   % Training set for class 3

x1_testing = x1all(N+1:end,[sep_len_index pet_len_index:pet_wid_index]);   % Test set for class 1   
x2_testing = x2all(N+1:end,[sep_len_index pet_len_index:pet_wid_index]);   % Test set for class 2
x3_testing = x3all(N+1:end,[sep_len_index pet_len_index:pet_wid_index]);   % Test set for class 3


x_training = [x1_training; x2_training; x3_training]; % Total training set
x_testing = [x1_testing; x2_testing; x3_testing]; % Total testing set

C = 3;                              % Number of classes: Setosa, Versicolor, Virginica
D = size(x_training,2);             % Input feature dimension
W0 = eye(C,D);                      % Weighting matrix
w0 = zeros(1,C);                    % Offset for w_i
W0 = [W0 w0.'];
X = [x_training.'; ones(1,N*C)];    % Input data for training
T = [kron(ones(1,N), [1 0 0].') ... % Targets
     kron(ones(1,N), [0 1 0].') ... 
     kron(ones(1,N), [0 0 1].')];
sigmoid = @(x) (1./(1+exp(-x)));     
gk = @(xk,W) sigmoid(W*xk);


% Training
alpha = 0.009; %% Try and fail
gradient = @(W)MSE_grad(X,T,W,gk);
[W,n] = gradient_descent(gradient,W0,alpha);

% Testing
n_test = M-N;
Xtest = [x_testing'; ones(1, size(x_testing,1))];                   % Test cases
Ttest = [repelem(1,n_test), repelem(2,n_test), repelem(3,n_test)];  % True solution
[~,classes] = max(W*Xtest);                                         % Classes given after training

error_test = classes~=Ttest;

error_rate_test = sum(error_test)/n_test
confusion_test = confusionmat(Ttest,classes)

% Finding error rate and confusion matrix for training data
Ttraining = [repelem(1,N), repelem(2,N), repelem(3,N)];
[~,training_classes] = max(W*X);

error_training = training_classes~=Ttraining;

error_rate_training = sum(error_training)/N
confusion_training = confusionmat(Ttraining, training_classes)



