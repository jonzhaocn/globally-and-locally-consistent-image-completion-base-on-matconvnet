function out = vl_nn_tanh(x,dzdy)
% VL_NNTANH CNN tanh unit.
%   Y = VL_NNTANH(X) computes the tanh of the data X. X can
%   have an arbitrary size.
%
%   DZDX = VL_NNTANH(X, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.


y = tanh(x);

if nargin <= 1 || isempty(dzdy)
  out = y;
else
  out = dzdy .* ( 1 - y.*y );
end
