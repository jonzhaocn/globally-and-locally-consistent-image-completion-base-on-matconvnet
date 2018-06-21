function [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, name, ksize, stride, dilate, depth, varargin)
    % Helper function to add a Convolutional + BatchNorm + ReLU
    % sequence to the network.
    args.relu = true ;
    args.bias = true ;
    args.bn = true;
    args = vl_argparse(args, varargin) ;
    if args.bias
        pars = {[name '_f'], [name '_b']};
    else
        pars = {[name '_f']};
    end
    % addLayer(name, block, inputs, outputs, params, varargin)
    if mod(ksize, 2)~=1
        error('ksize should be a odd');
    end
    if dilate == 1
        pad = (ksize - 1) / 2;
    elseif dilate > 1
        if stride > 1
            error('stride shoule be 1 when dilate > 1');
        end
        pad = (ksize-1)*dilate/2;
    else
        error('dilate should >= 1');
    end
    net.addLayer([name  '_conv'], ...
        dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
        'stride', stride, ...
        'dilate', dilate, ...
        'pad', pad, ...
        'hasBias', args.bias, ...
        'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
        lastAdded.var, ...
        [name '_conv'], ...
        pars) ;
    lastAdded.var = [name '_conv'];
    lastAdded.depth = depth ;
    
    if args.bn
        net.addLayer([name '_bn'], ...
            dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
            lastAdded.var, ...
            [name '_bn'], ...
            {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
        lastAdded.var = [name '_bn'] ;
    end
    
    if args.relu
        net.addLayer([name '_relu'] , ...
            dagnn.ReLU(), ...
            lastAdded.var, ...
            [name '_relu']) ;
        lastAdded.var = [name '_relu'] ;
    end
end