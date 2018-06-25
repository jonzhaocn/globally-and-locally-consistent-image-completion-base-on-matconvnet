function out = vl_nn_sigmoid_cross_entropy_loss(logits, labels, multiply_alpha, dzdy)
    if ~isequal(size(logits), size(labels))
        error("logits' size and labels' size should be equal");
    end
	if nargin <= 3 || isempty(dzdy)
	  out = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
      out = sum(out(:));
    else
        combination_loss_alpha = 0.0004;
		% --------
		temp1 = logits;
		temp1(logits>=0) = 1;
		temp1 = max(temp1,0);
		% --------
		temp2 = logits;
		temp2(temp2>=0) = -1;
		temp2(temp2<0) = 1;
		% --------
		temp3 = exp(-abs(logits));
		
		out = temp1 - labels + temp3./(1+temp3) .* temp2;
		out = dzdy .* out;
        if multiply_alpha
            out = out * combination_loss_alpha;
        end
	end
end