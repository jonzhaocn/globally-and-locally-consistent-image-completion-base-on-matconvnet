% init the completion network
function net = glcic_gen_init(varargin)
    opts.cudnnWorkspaceLimit = 1024*1024*1024;
    opts = vl_argparse(opts, varargin) ;

    net = dagnn.DagNN() ;
    lastAdded.var = 'original_images';
    
    net.addLayer(['masked_images'] , ...
        dagnn.MaskImage(), ...
        {'original_images', 'mask'}, ...
        'masked_images') ;
    
    lastAdded.var = 'masked_images';
    lastAdded.depth = 3;
    
    % glcic_conv(net, opts, lastAdded, name, ksize, stride, dilate, depth, varargin)
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_1', 5, 1, 1, 64);

    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_2', 3, 2, 1, 128);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_3', 3, 1, 1, 128);

    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_4', 3, 2, 1, 256);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_5', 3, 1, 1, 256);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_6', 3, 1, 1, 256);

    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_7', 3, 1, 2, 256);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_8', 3, 1, 4, 256);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_9', 3, 1, 8, 256);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_10', 3, 1, 16, 256);

    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_11', 3, 1, 1, 256);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_12', 3, 1, 1, 256);

    % glcic_conv_transpose(net, opts, lastAdded, name, ksize, upsample, depth, varargin)
    [net, lastAdded] = glcic_add_conv_transpose_block(net, opts, lastAdded, 'gen_layer_13', 5, 2, 128);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_14', 3, 1, 1, 128);

    [net, lastAdded] = glcic_add_conv_transpose_block(net, opts, lastAdded, 'gen_layer_15', 5, 2, 64);
    [net, lastAdded] = glcic_add_conv_block(net, opts, lastAdded, 'gen_layer_16', 3, 1, 1, 32);

    name = 'gen_layer_17';
    net.addLayer([name  '_conv'], ...
        dagnn.Conv('size', [3 3 lastAdded.depth 3], ...
        'stride', 1, ...
        'dilate', 1, ...
        'pad', (3 - 1) / 2, ...
        'hasBias', true, ...
        'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
        lastAdded.var, ...
        [name '_conv'], ...
        {[name '_f'], [name '_b']}) ;
    lastAdded.var = [name '_conv'];
    
    net.addLayer([name '_tanh'] , ...
        dagnn.Tanh(), ...
        lastAdded.var, ...
        [name '_tanh']) ;
    lastAdded.var = [name '_tanh'];
    
    % combine the convolutional output and the masked image to complete
    % the image
    net.addLayer('combine_generator_result_and_source', ...
        dagnn.CombineGeneratorResultAndSource(), ...
        {lastAdded.var, 'original_images', 'mask'}, ...
        'completed_images');
    net.vars(net.getVarIndex('completed_images')).precious = 1;
    % add a mse loss layer
    net.addLayer('mse_loss', ...
        dagnn.MSELoss(), ...
        {'completed_images', 'original_images', 'mask'}, ...
        'mse_loss');
    net.vars(net.getVarIndex('mse_loss')).precious = 1;
    
    % -------------
    net.initParams() ;
    net.meta.inputSize = [1 1 100];
end