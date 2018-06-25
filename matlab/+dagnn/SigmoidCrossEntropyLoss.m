classdef SigmoidCrossEntropyLoss < dagnn.ElementWise
    methods
        % inputs{1}:logits, inputs{2}:labels
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nn_sigmoid_cross_entropy_loss(inputs{1}, inputs{2}, inputs{3});
        end
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nn_sigmoid_cross_entropy_loss(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}) ;
            derInputs{2} = [] ;
            derInputs{3} = [];
            derParams = {} ;
        end
    end
end