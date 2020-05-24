classdef MLP < handle
    properties (SetAccess=private)
        inputDim
        hiddenDim
        outputDim
        
        hiddenWeights
        outputWeights
    end
    
    methods
        function obj=MLP(inputD,hiddenD,outputD)
            obj.inputDim=inputD;
            obj.hiddenDim=hiddenD;
            obj.outputDim=outputD;
            obj.hiddenWeights=zeros(hiddenD,inputD+1);
            obj.outputWeights=zeros(outputD,hiddenD+1);
            %disp(inputD);
        end
        
        function obj=initWeights_from_mat(obj, hidden, output)
            obj.hiddenWeights = hidden;
            obj.outputWeights = output;
        end
        
        function obj=initWeight(obj,variance)
            obj.hiddenWeights = 1 - 2 * rand(obj.hiddenDim, obj.inputDim + 1) * variance;
            obj.outputWeights = 1 - 2 * rand(obj.outputDim, obj.hiddenDim + 1) * variance;
        end
        
        function [hiddenNet,hidden,outputNet,output]=compute_net_activation(obj, input)
            input = [input; 1];
            hiddenNet = obj.hiddenWeights * input;
            hidden = obj.sigmoid(hiddenNet);
            input = [hidden; 1];
            outputNet = obj.outputWeights * input;
            output = obj.sigmoid(outputNet);
        end
        
        function output=compute_output(obj,input)
            [hN,h,oN,output] = obj.compute_net_activation(input);
        end
        
        function o=adapt_to_target(obj,input,target,rate)
            [hN,h,oN,o] = obj.compute_net_activation(input);
            
            error = o - target;
            do = error .* (o.*(1 - o));
            w_dif_o = do * transpose([h; 1]);
            
            dh = transpose(obj.outputWeights)*do;
            dh = dh(1:length(h)) .* (h.*(1 - h));
            
            w_dif_h = dh * transpose([input; 1]);
            
            obj.hiddenWeights = obj.hiddenWeights - rate*w_dif_h;
            obj.outputWeights = obj.outputWeights - rate*w_dif_o;
        end
        
        function outputs=adapt_to_target_batch(obj,inputs,targets,rate)
            h_delta = 0;
            o_delta = 0;
            
            for index = 1:length(inputs(1, :))
                input = inputs(:, index);
                target = targets(:, index);
                
                [hN,h,oN,o] = obj.compute_net_activation(input);
                
                error = o - target;
                do = error .* (o.*(1 - o));
                w_dif_o = do * transpose([h; 1]);

                dh = transpose(obj.outputWeights)*do;
                dh = dh(1:length(h)) .* (h.*(1 - h));

                w_dif_h = dh * transpose([input; 1]);

                h_delta = h_delta + w_dif_h;
                o_delta = o_delta + w_dif_o;
            end
            
            obj.hiddenWeights = obj.hiddenWeights - rate*(h_delta / length(inputs(1, :)));
            obj.outputWeights = obj.outputWeights - rate*(o_delta / length(inputs(1, :)));
        end
        
        function answer = sigmoid(obj, output)
            
            answer = [];
            for n = 1:length(output)
                answer = [answer; 1 / (1 + exp(-output(n)))];
            end
        end
        
        function answer = softmax(obj, output)
            
            answer = [];
            m = sum(exp(output));
            for n = 1:length(output)
                answer = [answer; exp(output(n)) / m];
            end
        end
    end
end
