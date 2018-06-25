function out = vl_nn_mask_image(original_images, mask, dzdy)
    init_bias = 0.437;
    if nargin <= 2 || isempty(dzdy)
        out = original_images .* (1-mask) + mask * init_bias;
    else
        out = dzdy .* (1-mask);
    end
end