function out = vl_nn_sigmoid_cross_entropy_loss(logits, labels, dzdy)
    if ~isequal(size(logits), size(labels))
        error("logits' size and labels' size should be equal");
    end
	if nargin <= 2 || isempty(dzdy)
	  out = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
      out = sum(out(:));
	else
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
	end
end