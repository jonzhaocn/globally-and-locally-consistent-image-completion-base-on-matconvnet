function [netG, netD, info] = glcic_train(varargin)
% GLCIC_TRAIN demonstrates training a global and local consistent image completion gan on CelebA 
% global and local consistent image completion

% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

    run(fullfile(fileparts(mfilename('fullpath')), ...
      '..','..', 'matlab', 'vl_setupnn.m')) ;

    opts.dataDir = fullfile(vl_rootnn, 'data','celeba') ;
    [opts, varargin] = vl_argparse(opts, varargin) ;

    opts.expDir = fullfile(vl_rootnn, 'exp', 'celeba-glcic') ;
    [opts, varargin] = vl_argparse(opts, varargin) ;

    opts.numFetchThreads = 12 ;
    % opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
    opts.train = struct('gpus', [1]) ;
    opts = vl_argparse(opts, varargin) ;

    if ~isfield(opts.train, 'gpus')
        opts.train.gpus = [];
    end

    % -------------------------------------------------------------------------
    %                                                              Prepare data
    % -------------------------------------------------------------------------
    save_dir_name = 'cropped_for_glcic';
    
    if ~exist(sprintf('%s/%s',opts.dataDir, save_dir_name), 'dir')
      preprocess_celeba(opts.dataDir, save_dir_name);
    end
    d = dir(sprintf('%s/%s/*.jpg',opts.dataDir, save_dir_name));
    imdb.images.name = cell(numel(d),1);
    imdb.imageDir = fullfile(opts.dataDir, save_dir_name);
    for i=1:numel(d)
      imdb.images.name{i} = d(i).name;
    end
    imdb.images.set = ones(numel(d),1);

    % -------------------------------------------------------------------------
    %                                                             Prepare model
    % -------------------------------------------------------------------------
    % generative model
    netG = glcic_gen_init;
    % discriminator model
    netD = glcic_disc_init;
    
    % Meta parameters
    meta.inputSize = [96 96 3] ;
    meta.augmentation.jitterLocation = false ;
    meta.augmentation.jitterFlip = false ;
    meta.augmentation.jitterBrightness = 0 ;
    meta.augmentation.jitterAspect = 0 ;
    % train options 
    lr = logspace(-3, -6, 20);
    meta.trainOpts.learningRate =  lr;
    meta.trainOpts.numEpochs = numel(lr) ;
    meta.trainOpts.batchSize = 64 ;
    meta.trainOpts.weightDecay = 0.0005 ;
    meta.trainOpts.sample_save_per_batch_count = 100;
    meta.trainOpts.mask_range = [32, 32];
    meta.trainOpts.local_area_size = [64, 64];
    % 
    meta.normalization.averageImage = [];
    meta.normalization.imageSize = [128 128 3];
    meta.normalization.cropSize = 128/178;
    % -------------------------------------------------------------------------
    %                                                                     Learn
    % -------------------------------------------------------------------------

    [netG, netD, info] = glcic_train_dagnn(netG, netD, imdb, ... 
                          getBatchFn(opts, meta), ...
                          'expDir', opts.expDir, ...
                          meta.trainOpts, ...
                          opts.train) ;

end
% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

    useGpu = numel(opts.train.gpus) > 0 ;

    bopts.test = struct(...
      'useGpu', useGpu, ...
      'numThreads', opts.numFetchThreads, ...
      'imageSize',  meta.normalization.imageSize(1:2), ...
      'cropSize', meta.normalization.cropSize, ...
      'subtractAverage', []) ;

    % Copy the parameters for data augmentation
    bopts.train = bopts.test ;
    for f = fieldnames(meta.augmentation)'
      f = char(f) ;
      bopts.train.(f) = meta.augmentation.(f) ;
    end

    fn = @(x,y) getBatch(bopts,x,y) ;
end
% -------------------------------------------------------------------------
function varargout = getBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    if ~isempty(batch) && imdb.images.set(batch(1)) == 1
      phase = 'train' ;
    else
      phase = 'test' ;
    end

    data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
    if nargout > 0
        % scale down to [-1,1]
        data = (data-128)/128;
        varargout{1} = {'input',  data};
    end
end
