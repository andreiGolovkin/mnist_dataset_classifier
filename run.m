training_x = loadMNISTImages('train-images-idx3-ubyte');
training_y = loadMNISTLabels('train-labels-idx1-ubyte');

testing_x = loadMNISTImages('t10k-images-idx3-ubyte');
testing_y = loadMNISTLabels('t10k-labels-idx1-ubyte');

results = [];

st = cputime;
start_time = cputime;

%for size = 10:10:200

m = MLP(784, 110, 10);
m.initWeight(1.0);

for epoch = 1:10

    step = 10;

    for n = 1:step:length(training_x(1, :))
        %disp(['size of hidden layer is ' num2str(size) ' attempt nr. ' num2str(n) ' of ' num2str(length(training_x))])
        if cputime - start_time > 5
            clc
            disp(['epoch ' num2str(epoch) ' attempt nr. ' num2str(n) ' of ' num2str(length(training_x))])
            disp(['overall progress    ' draw_loading(epoch, 10)])
            disp(['epoch progress      ' draw_loading(n, length(training_x))])
            
            start_time = cputime;
        end

        proper_label = zeros(10, step);
        for a = 1:step
            proper_label(training_y(n+a-1)+1,a) = 1.0;
        end

        m.adapt_to_target_batch(training_x(:, n:n+step-1), proper_label, 0.5);
    end
    
    overall = 0;

    for n = 1:length(testing_x(1, :))
       result = m.compute_output(testing_x(:, n));

       expected = testing_y(n);
       actual = get_val(result);

       overall = overall + (expected == actual);
    end

    results = [results overall/length(testing_x)];
    figure(2);
    plot(1:length(results), results);
    

end

clc
disp(['epoch ' num2str(10) ' attempt nr. ' num2str(length(training_x)) ' of ' num2str(length(training_x))])
disp(['overall progress    ' draw_loading(1, 1)])
disp(['epoch progress      ' draw_loading(1, 1)])
disp(['full run took ' num2str((cputime - st) / 60)])
%end


function ans = get_val(m)
    ans = 0;
    val = 0;
    for n = 1:length(m)
        if m(n) > val
            val = m(n);
            ans = n - 1;
        end
    end
end

function ans = draw_loading(current, max)
    ans = ['['];
    
    m = 20;
    n = m * (current / max);
    
    for a = 1:m
        if a <= n
            ans = [ans '#'];
        else
            ans = [ans ' '];
        end
    end
    
    ans = [ans ']'];
end