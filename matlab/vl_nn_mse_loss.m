% the miss area in the mask is masked as 1,the other is masked as 0, 
% mask is a binary array and has the same size as completed_images and original_images
function out = vl_nn_mse_loss(completed_images, original_images, mask, dzdy)
    missing_pixel_count = sum(mask(:))/3;
    if nargin <= 3 || isempty(dzdy)
        out = mask .* (completed_images-original_images);
        out = out .* out;
        % average
        out = sum(out(:)) / missing_pixel_count;
    else
        out = dzdy .* 2 * mask .* mask .* (completed_images - original_images) / missing_pixel_count;
    end
end