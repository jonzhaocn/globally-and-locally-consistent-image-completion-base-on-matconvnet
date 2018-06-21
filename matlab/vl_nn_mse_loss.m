% the miss area in the mask is masked as 1,the others area of mask is
% masked as 0, mask is a binary array and has the same size as completion
% and source
function out = vl_nn_mse_loss(completion, source, mask, dzdy)
    if nargin <= 3 || isempty(dzdy)
        out = mask .* (completion-source);
        out = out .* out;
        out = sum(out(:));
    else
        out = dzdy .* mask * 2;
    end
end