function [net, lastAdded] = glcic_add_conv_transpose_block(net, opts, lastAdded, name, ksize, upsample, depth, varargin)
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
    if mod(ksize, 2)~=1
        error('ksize should be a odd')
    end
    if ksize < upsample
        error('ksize should >=  upsample')
    end
    crop_h = ksize - upsample;
    crop_w = ksize - upsample;
    crop_top = floor(crop_h/2);
    crop_bottom = crop_h - crop_top;
    crop_left = floor(crop_w/2);
    crop_right = crop_w - crop_left;
    % addLayer(name, block, inputs, outputs, params, varargin)
    net.addLayer([name  '_conv_transpose'], ...
        dagnn.ConvTranspose('size', [ksize ksize depth lastAdded.depth], ...
        'upsample', upsample, ...
        'crop', [crop_top, crop_bottom, crop_left, crop_right],  ...
        'hasBias', args.bias, ...
        'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
        lastAdded.var, ...
        [name '_conv_transpose'], ...
        pars) ;
    lastAdded.var = [name '_conv_transpose'];
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
            dagnn.ReLU('leak', 0.2), ...
            lastAdded.var, ...
            [name '_relu']) ;
        lastAdded.var = [name '_relu'] ;
    end
end