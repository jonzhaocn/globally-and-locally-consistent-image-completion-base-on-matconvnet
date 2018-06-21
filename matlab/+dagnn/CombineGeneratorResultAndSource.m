classdef CombineGeneratorResultAndSource < dagnn.ElementWise
    methods
        % inputs{1}:generator result, inputs{2}:source, inputs{3}:mask
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nn_combine_generator_result_and_source(inputs{1}, inputs{2}, inputs{3});
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nn_combine_generator_result_and_source(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}) ;
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {} ;
        end
    end
end