function [tests, training] = parse_data(test_size)
    data = 1:60000;
    
    tests = {};
    training = {};
   
    for start = 1:test_size:length(data)
       
        test_set = data(start:start+test_size-1);
       
        tests = [tests; test_set];
       
        training_set = [data(1:start-1) data(start+test_size:end)];
       
        training = [training; training_set];
    end
   
end
