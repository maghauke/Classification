close all
clear all
clc
%% The Iris task part 2a

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

%% Histograms
% Sepal length feature, the second most overlapping
figure(1)
hsl1 = histogram(sepal_length_class1);
hold on
hsl2 = histogram(sepal_length_class2);
hold on
hsl3 = histogram(sepal_length_class3);
xlabel('sepal length [cm]')
legend('class1','class2','class3')
title('Sepal length for the different classes')
hsl1.Normalization = 'probability';
hsl1.BinWidth = 0.25;
hsl2.Normalization = 'probability';
hsl2.BinWidth = 0.25;
hsl3.Normalization = 'probability';
hsl3.BinWidth = 0.25;
hold off


% Sepal width feature, the most overlapping
figure(2)
hsd1 = histogram(sepal_width_class1);
hold on
hsd2 = histogram(sepal_width_class2);
hold on
hsd3 = histogram(sepal_width_class3);
xlabel('sepal width [cm]')
legend('class1','class2','class3')
title('Sepal width for the different classes')
hsd1.Normalization = 'probability';
hsd1.BinWidth = 0.25;
hsd2.Normalization = 'probability';
hsd2.BinWidth = 0.25;
hsd3.Normalization = 'probability';
hsd3.BinWidth = 0.25;
hold off

% Petal length feature, least overlapping
figure(3)
hpl1 = histogram(petal_length_class1);
hold on
hpl2 = histogram(petal_length_class2);
hold on
hpl3 = histogram(petal_length_class3);
xlabel('Petal length [cm]')
legend('class1','class2','class3')
title('petal length for the different classes')
hpl1.Normalization = 'probability';
hpl1.BinWidth = 0.25;
hpl2.Normalization = 'probability';
hpl2.BinWidth = 0.25;
hpl3.Normalization = 'probability';
hpl3.BinWidth = 0.25;
hold off


% Petal width feature, least overlapping
figure(4)
hpd1 = histogram(petal_width_class1);
hold on
hpd2 = histogram(petal_width_class2);
hold on
hpd3 = histogram(petal_width_class3);
xlabel('Petal width [cm]')
legend('class1','class2','class3')
title('Petal width for the different classes')
hpd1.Normalization = 'probability';
hpd1.BinWidth = 0.25;
hpd2.Normalization = 'probability';
hpd2.BinWidth = 0.25;
hpd3.Normalization = 'probability';
hpd3.BinWidth = 0.25;
hold off

