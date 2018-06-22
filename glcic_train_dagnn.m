function [netG,netD,stats] = glcic_train_dagnn(netG, netD, imdb, getBatch, varargin)
% GLCIC_TRAIN_DAGNN demonstrates training a global and local consistent image completion GAN using the DagNN wrapper

% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

    opts.expDir = fullfile('data','exp') ;
    opts.continue = true ;
    opts.batchSize = 256 ;
    opts.numSubBatches = 1 ;
    opts.train = [] ;
    opts.val = [] ;
    opts.gpus = [] ;
    opts.prefetch = false ;
    opts.numEpochs = 10 ;
    opts.learningRate = 0.002 ;
    opts.weightDecay = 0.0005 ;
    opts.labelSmoothing = false ;

    opts.solver = @solver.adam ; 
    opts.solverOpts.beta1 = 0.5 ;
    opts.sample_save_per_batch_count = 100;
    opts.mask_range = [32, 64];
    opts.local_area_size = [64, 64];
    opts.miss_area_init_bias = 0.5;
    
    [opts, varargin] = vl_argparse(opts, varargin) ;
    if ~isempty(opts.solver)
        assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
            'Invalid solver; expected a function handle with two outputs.') ;
        % Call without input arguments, to get default options
        opts.solverOpts = opts.solver() ;
    end

    opts.momentum = 0.9 ;
    opts.saveSolverState = true ;
    opts.nesterovUpdate = false ;
    opts.randomSeed = 0 ;
    opts.profile = false ;
    opts.parameterServer.method = 'mmap' ;
    opts.parameterServer.prefix = 'mcn' ;

    opts.derOutputs = {'objective', 1} ;
    opts.extractStatsFn = @extractStats ;
    opts.plotStatistics = true;
    opts.postEpochFn = [] ;  % postEpochFn(net,params,state) called after each epoch; can return a new learning rate, 0 to stop, [] for no change
    opts = vl_argparse(opts, varargin) ;

    if ~exist(opts.expDir, 'dir')
        mkdir(opts.expDir);
    end
    if isempty(opts.train)
        opts.train = find(imdb.images.set==1);
    end
    if isempty(opts.val)
        opts.val = find(imdb.images.set==2);
    end
    if isnan(opts.train)
        opts.train = [];
    end
    if isnan(opts.val)
        opts.val = [];
    end

    % -------------------------------------------------------------------------
    %                                                            Initialization
    % -------------------------------------------------------------------------

    evaluateMode = isempty(opts.train) ;
    if ~evaluateMode
        if isempty(opts.derOutputs)
            error('DEROUTPUTS must be specified when training.\n') ;
        end
    end

    % -------------------------------------------------------------------------
    %                                                        Train and validate
    % -------------------------------------------------------------------------

    modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
    modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

    start = opts.continue * findLastCheckpoint(opts.expDir) ;
    if start >= 1
        fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
        [netG, netD, state, stats] = loadState(modelPath(start)) ;
    else
        state = [] ;
    end

    for epoch=start+1:opts.numEpochs

        % Set the random seed based on the epoch and opts.randomSeed.
        % This is important for reproducibility, including when training
        % is restarted from a checkpoint.

        rng(epoch + opts.randomSeed) ;
        prepareGPUs(opts, epoch == start+1) ;

        % Train for one epoch.
        params = opts ;
        params.epoch = epoch ;
        params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
        params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
        params.val = opts.val(randperm(numel(opts.val))) ;
        params.imdb = imdb ;
        params.getBatch = getBatch ;
        params.sample_save_per_batch_count = opts.sample_save_per_batch_count;
        if epoch == 1
            trainingObject = 'generator';
        elseif epoch == 2
            trainingObject = 'discriminator';
        else
            trainingObject = 'combination';
        end
        if numel(opts.gpus) <= 1
            [netG, netD, state] = process_epoch(netG, netD, state, params, 'train', trainingObject) ;
            if ~evaluateMode
                saveState(modelPath(epoch), netG, netD, state) ;
            end
            lastStats = state.stats ;
        else
            error('Multi-gpu support is not implemented!\n');
        end

        stats.train(epoch) = lastStats.train ;
        stats.val(epoch) = struct() ;
        clear lastStats ;
        saveStats(modelPath(epoch), stats) ;

        if opts.plotStatistics
            switchFigure(1) ; clf ;
            plots = setdiff(...
                cat(2,...
                fieldnames(stats.train)', ...
                fieldnames(stats.val)'), {'num', 'time'}) ;
            for p = plots
                p = char(p) ;
                values = zeros(0, epoch) ;
                leg = {} ;
                for f = {'train', 'val'}
                    f = char(f) ;
                    if isfield(stats.(f), p)
                        tmp = [stats.(f).(p)] ;
                        values(end+1,:) = tmp(1,:)' ;
                        leg{end+1} = f ;
                    end
                end
                subplot(1,numel(plots),find(strcmp(p,plots))) ;
                plot(1:epoch, values','o-') ;
                xlabel('epoch') ;
                title(p) ;
                legend(leg{:}) ;
                grid on ;
            end
            drawnow ;
            print(1, modelFigPath, '-dpdf') ;
        end

        if ~isempty(opts.postEpochFn)
            if nargout(opts.postEpochFn) == 0
                opts.postEpochFn(net, params, state) ;
            else
                lr = opts.postEpochFn(net, params, state) ;
                if ~isempty(lr), opts.learningRate = lr; end
                if opts.learningRate == 0, break; end
            end
        end
    end

    % With multiple GPUs, return one copy
    if isa(net, 'Composite')
        net = net{1}; 
    end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

    for s = {'train', 'val'}
        s = char(s) ;
        total = 0 ;

        % initialize stats stucture with same fields and same order as
        % stats_{1}
        stats__ = stats_{1} ;
        names = fieldnames(stats__.(s))' ;
        values = zeros(1, numel(names)) ;
        fields = cat(1, names, num2cell(values)) ;
        stats.(s) = struct(fields{:}) ;

        for g = 1:numel(stats_)
            stats__ = stats_{g} ;
            num__ = stats__.(s).num ;
            total = total + num__ ;

            for f = setdiff(fieldnames(stats__.(s))', 'num')
                f = char(f) ;
                stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

                if g == numel(stats_)
                    stats.(s).(f) = stats.(s).(f) / total ;
                end
            end
        end
        stats.(s).num = total ;
    end
end
% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
    sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
    for i = 1:numel(sel)
        stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
    end
end
% -------------------------------------------------------------------------
function saveState(fileName, netG_, netD_, state)
% -------------------------------------------------------------------------
    netG = netG_.saveobj() ;
    netD = netD_.saveobj() ;
    save(fileName, 'netG', 'netD', 'state') ;
end
% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
    if exist(fileName)
        save(fileName, 'stats', '-append') ;
    else
        save(fileName, 'stats') ;
    end
end
% -------------------------------------------------------------------------
function [netG, netD, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
    load(fileName, 'netG', 'netD', 'state', 'stats') ;
    netG = dagnn.DagNN.loadobj(netG) ;
    netD = dagnn.DagNN.loadobj(netD) ;
    if isempty(whos('stats'))
        error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
            fileName) ;
    end
end
% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
    list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
    tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
    epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
    epoch = max([epoch 0]) ;
end
% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
    if get(0,'CurrentFigure') ~= n
        try
            set(0,'CurrentFigure',n) ;
        catch
            figure(n) ;
        end
    end
end
% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
    clear vl_tmove vl_imreadjpeg ;
end
% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
    numGpus = numel(opts.gpus) ;
    if numGpus > 1
        % check parallel pool integrity as it could have timed out
        pool = gcp('nocreate') ;
        if ~isempty(pool) && pool.NumWorkers ~= numGpus
            delete(pool) ;
        end
        pool = gcp('nocreate') ;
        if isempty(pool)
            parpool('local', numGpus) ;
            cold = true ;
        end

    end
    if numGpus >= 1 && cold
        fprintf('%s: resetting GPU\n', mfilename)
        clearMex() ;
        if numGpus == 1
            gpuDevice(opts.gpus)
        else
            spmd
                clearMex() ;
                gpuDevice(opts.gpus(labindex))
            end
        end
    end
end
