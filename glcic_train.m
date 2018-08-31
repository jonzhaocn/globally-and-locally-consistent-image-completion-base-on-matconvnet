% entry program of the training process

% glcic_train demonstrates training a global and local consistent image completion gan on CelebA 
% global and local consistent image completion

function [net, info] = glcic_train(varargin)

    run(fullfile(fileparts(mfilename('fullpath')), ...
      '..','..', 'matlab', 'vl_setupnn.m')) ;

    opts.dataDir = fullfile(vl_rootnn, 'data','celeba') ;
    [opts, varargin] = vl_argparse(opts, varargin) ;

    opts.expDir = fullfile(vl_rootnn, 'exp', 'celeba-glcic') ;
    [opts, varargin] = vl_argparse(opts, varargin) ;

    opts.numFetchThreads = 12 ;
    % opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
    gpus = [1,2];
    opts.train = struct('gpus', gpus) ;
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
    d = dir(sprintf('%s/%s/*.jpg', opts.dataDir, save_dir_name));
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
    net = [netG, netD];
    
    % the process is cropping firstly and then resize the image according to the imageSize
    
    % if the jitterLocation is equal to 1, the crop location will be random,
    % if no the result will crop from the center in image
    meta.augmentation.jitterLocation = 1 ;
    % if jitterFilp equals 1, randomly flips a crop horizontally with 50% probability.
    meta.augmentation.jitterFlip = 1 ;
    % jitterAspect = Wcrop/Hcrop, a value of [0 0] or 0 stretches the crop to fit the input image. 
    meta.augmentation.jitterAspect = 0 ;
    meta.augmentation.jitterScale = 1 ;
    meta.augmentation.jitterBrightness = 0 ;
    % train options 
    lr = logspace(-3, -5, 30);
    meta.trainOpts.learningRate =  lr;
    meta.trainOpts.numEpochs = 21 ;
    meta.trainOpts.batchSize = 64 ;
    meta.trainOpts.weightDecay = 0.0005 ;
    % save a sample image per x batchs
    meta.trainOpts.sample_save_per_batch_count = 100;
    % the range of each side of the mask is [32,64]
    meta.trainOpts.mask_range = [32, 64];
    % the input size of local discriminator is 64*64
    meta.trainOpts.local_area_size = [64, 64];
    % 
    meta.normalization.averageImage = [];
    % imageSize is the final output size
    meta.normalization.imageSize = [128 128 3];
    % the cropSize in vl_imreadjpeg is equal to meta.normalization.cropSize*jitterScale
    meta.normalization.cropSize = 128/178;
    % -------------------------------------------------------------------------
    %                                                                     Learn
    % -------------------------------------------------------------------------

    [net, info] = glcic_train_dagnn(net, imdb, ... 
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
