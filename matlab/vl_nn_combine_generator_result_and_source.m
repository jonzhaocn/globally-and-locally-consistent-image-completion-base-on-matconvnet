% the miss area in the mask is masked as 1,the others area of mask is
% masked as 0, mask is a binary array and has the same size as generator_result
% and source
function out = vl_nn_combine_generator_result_and_source(generator_result, original_images, mask, dzdy)
    if nargin <= 3 || isempty(dzdy)
        out = generator_result .* mask + original_images .* (1-mask);
    else
        out = dzdy .* mask;
    end
end