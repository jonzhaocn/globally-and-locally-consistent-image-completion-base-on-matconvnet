function out = vl_nn_mask_image(original_images, mask, init_bias, dzdy)
    if nargin <= 3 || isempty(dzdy)
        out = original_images .* (1-mask) + mask * init_bias;
    else
        out = dzdy .* (1-mask);
    end
end