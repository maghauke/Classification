close all
clear all
clc
%% The Iris task part 2d?

x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');

sepal_length_index = 1;
sepal_width_index = 2;
petal_length_index = 3;
petal_width_index = 4;


sepal_length_class1 = x1all(:,sepal_length_index);
sepal_length_class2 = x2all(:,sepal_length_index);
sepal_length_class3 = x3all(:,sepal_length_index);

sepal_width_class1 = x1all(:,sepal_width_index);
sepal_width_class2 = x2all(:,sepal_width_index);
sepal_width_class3 = x3all(:,sepal_width_index);

petal_length_class1 = x1all(:,petal_length_index);
petal_length_class2 = x2all(:,petal_length_index);
petal_length_class3 = x3all(:,petal_length_index);

petal_width_class1 = x1all(:,petal_width_index);
petal_width_class2 = x2all(:,petal_width_index);
petal_width_class3 = x3all(:,petal_width_index);

%% Scatter

% Sepal length and width feature
figure(1)
subplot(2,1,1)
Ssl1 = scatter(sepal_length_class1,sepal_width_class1,'o');
hold on
Ssl2 = scatter(sepal_length_class2,sepal_width_class2,'x');
hold on
Ssl3 = scatter(sepal_length_class3,sepal_width_class3,'*');
legend('Setosa','Versicolor','Virginica')

title('Setosa vs. Versicolor vs. Virginica')
xlabel('sepal length [cm]')
ylabel('sepal width [cm]')
hold off

% Petal length and width feature
subplot(2,1,2)
Psl1 = scatter(petal_length_class1,petal_width_class1,'o');
hold on
Psl2 = scatter(petal_length_class2,petal_width_class2,'x');
hold on
Psl3 = scatter(petal_length_class3,petal_width_class3,'*');
legend('Setosa','Versicolor','Virginica')

title('Setosa vs. Versicolor vs. Virginica')
xlabel('petal length [cm]')
ylabel('petal width [cm]')
hold off

