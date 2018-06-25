classdef MaskImage < dagnn.ElementWise
    methods
        % inputs{1}: vl_nn_mask_image(source, mask, init_bias, dzdy)
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nn_mask_image(inputs{1}, inputs{2});
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nn_mask_image(inputs{1}, inputs{2}, derOutputs{1}) ;
            derInputs{2} = [];
            derParams = {} ;
        end
    end
end