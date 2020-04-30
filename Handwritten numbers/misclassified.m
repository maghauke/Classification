%Find all misclassified pictures

misclass = [];

for i = 1:10000
    for j = 1:10
        if score(i,j) == 1
            if j ~= (testlab(i)+1)
                misclass = [misclass; i];
            end
        end
    end
end

%Look at the 'misclass' matrix for misclassified pictures and plot them
figure()
miss = zeros(28,28);
miss(:)= testv(116,:);
I = image(miss');
print('I','-depsc');
        