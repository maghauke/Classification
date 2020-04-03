close all
clear all
clc
%% The Iris task part 1

x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');

M = length(x1all);

N = 30;         % Training set size
features = 4;   % sepal_length, sepal_width, petal_length, petal_width

x1_training = x1all(1:N,1:features);   % Training set for class 1   
x2_training = x2all(1:N,1:features);   % Training set for class 2
x3_training = x3all(1:N,1:features);   % Training set for class 3

x1_testing = x1all(N+1:end,1:features);   % Test set for class 1   
x2_testing = x2all(N+1:end,1:features);   % Test set for class 2
x3_testing = x3all(N+1:end,1:features);   % Test set for class 3


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
alpha = 0.0075; %% ??
gradient = @(W)MSE_grad(X,T,W,gk);
[W,n] = gradient_descent(gradient,W0,alpha);

% Testing
n_test = M-N;
Xtest = [x_testing'; ones(1, size(x_testing,1))];                   % Test cases
Ttest = [repelem(1,n_test), repelem(2,n_test), repelem(3,n_test)];  % True solution
[~,classes] = max(W*Xtest);                                         % Classes given after training

error_test = classes~=Ttest;

% Scatter
error_index = find(error_test);
x_testing_errors = x_testing(error_index,1:features);

error_rate_test = sum(error_test)/n_test
confusion_test = confusionmat(Ttest,classes)

% Finding error rate and confusion matrix for training data
Ttraining = [repelem(1,N), repelem(2,N), repelem(3,N)];
[~,training_classes] = max(W*X);

error_training = training_classes~=Ttraining;

error_rate_training = sum(error_training)/N
confusion_training = confusionmat(Ttraining, training_classes)


%% Scatter
sepal_length_index = 1;
sepal_width_index = 2;
petal_length_index = 3;
petal_width_index = 4;

sepal_length_class1 = x1_testing(:,sepal_length_index);
sepal_length_class2 = x2_testing(:,sepal_length_index);
sepal_length_class3 = x3_testing(:,sepal_length_index);

sepal_width_class1 = x1_testing(:,sepal_width_index);
sepal_width_class2 = x2_testing(:,sepal_width_index);
sepal_width_class3 = x3_testing(:,sepal_width_index);

petal_length_class1 = x1_testing(:,petal_length_index);
petal_length_class2 = x2_testing(:,petal_length_index);
petal_length_class3 = x3_testing(:,petal_length_index);

petal_width_class1 = x1_testing(:,petal_width_index);
petal_width_class2 = x2_testing(:,petal_width_index);
petal_width_class3 = x3_testing(:,petal_width_index);

% errors, we know :
sepal_length_error_class2 = x_testing_errors(:,sepal_length_index);
sepal_width_error_class2 = x_testing_errors(:,sepal_width_index);
petal_length_error_class2 = x_testing_errors(:,petal_length_index);
petal_width_error_class2 = x_testing_errors(:,petal_width_index);


% Sepal length and width feature
figure(1)
Ssl1 = scatter(sepal_length_class1,sepal_width_class1,'+');
hold on
Ssl2 = scatter(sepal_length_class2,sepal_width_class2,'x');
hold on
Ssl3 = scatter(sepal_length_class3,sepal_width_class3,'*');
hold on
Sse = scatter(sepal_length_error_class2,sepal_width_error_class2,'o');
legend('class1','class2','class3','misclassified')

title('Testing set: Class1 vs. Class2 vs. Class3')
xlabel('sepal length [cm]')
ylabel('sepal width [cm]')
hold off

% Petal length and width feature
figure(2)
Psl1 = scatter(petal_length_class1,petal_width_class1,'+');
hold on
Psl2 = scatter(petal_length_class2,petal_width_class2,'x');
hold on
Psl3 = scatter(petal_length_class3,petal_width_class3,'*');
Spe = scatter(petal_length_error_class2,petal_width_error_class2,'o1');
legend('class1','class2','class3','misclassified')

title('Testing set: Class1 vs. Class2 vs. Class3')
xlabel('petal length [cm]')
ylabel('petal width [cm]')
hold off