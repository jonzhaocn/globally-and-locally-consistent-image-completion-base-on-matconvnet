% process a epoch when training the networks
% input:
%   netG: a instance, the completed network
%   netD: a instance, the local and global discriminator
%   state: a struct to store vars for update network
%   parmas: setting
%   mode: 'train' or 'val'
function [net, state] = process_epoch(net, state, params, mode)
    netG = net(1);
    netD = net(2);
    netG.mode = 'normal';
    netD.mode = 'normal';
    alpha = 0.25;
    % initialize with momentum 0
    if isempty(state)
        stateG.solverState = cell(1, numel(netG.params)) ;
        stateG.solverState(:) = {0} ;
        stateD.solverState = cell(1, numel(netD.params)) ;
        stateD.solverState(:) = {0} ;
        state = [stateG, stateD];
    end
    
    stateG = state(1);
    stateD = state(2);
    
    % move CNN  to GPU as needed
    numGpus = numel(params.gpus) ;
    if numGpus >= 1
        netG.move('gpu') ;
        netD.move('gpu') ;
        
        for i = 1:numel(stateG.solverState)
            s = stateG.solverState{i} ;
            if isnumeric(s)
                stateG.solverState{i} = gpuArray(s) ;
            elseif isstruct(s)
                stateG.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
            end
        end
        for i = 1:numel(stateD.solverState)
            s = stateD.solverState{i} ;
            if isnumeric(s)
                stateD.solverState{i} = gpuArray(s) ;
            elseif isstruct(s)
                stateD.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
            end
        end
    end
    if numGpus > 1
          parservG = ParameterServer(params.parameterServer) ;
          netG.setParameterServer(parservG) ;
          
          parservD = ParameterServer(params.parameterServer) ;
          netD.setParameterServer(parservD) ;
          
    else
        parservG = [] ;
        parservD = [];
    end
    
    % profile
    if params.profile
        if numGpus <= 1
            profile clear ;
            profile on ;
        else
            mpiprofile reset ;
            mpiprofile on ;
        end
    end
    
    % --------------
    num = 0 ;
    epoch = params.epoch;
    subset = params.(mode);
    adjustTime = 0 ;
    stats.num = 0 ; % return something even if subset = []
    stats.time = 0 ;
    start = tic ;
    stats.GLoss = 0;
    stats.DLoss = 0;
    % don't have to complete the whole epoch as need
    if params.epochPercentage > 1
        params.epochPercentage =1;
    elseif params.epochPercentage <= 0
        params.epochPercentage = 0.1;
    end
    subset = subset(1: round(numel(subset)*params.epochPercentage) );
    
    % --------------
    % get batch and train
    % --------------
    for t=1:params.batchSize:numel(subset)
        batchIndex = fix((t-1)/params.batchSize)+1;
        totalBatch = ceil(numel(subset)/params.batchSize);
        fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
            batchIndex, totalBatch) ;
        batchSize = min(params.batchSize, numel(subset) - t + 1) ;
        %
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) ;
        batchEnd = min(t+params.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        % get images as input
        inputs = params.getBatch(params.imdb, batch) ;
        
        if params.prefetch
            batchStart = t + (labindex-1) + params.batchSize ;
            batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
            nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
            params.getBatch(params.imdb, nextBatch) ;
        end
        % create random mask
        maskC = create_random_mask(size(inputs{2}), params.local_area_size, params.mask_range);
        maskD = create_random_mask(size(inputs{2}), params.local_area_size, params.mask_range);
        
        original_images = inputs{2};
        labelFake = zeros(1, 1, 1, numel(batch), 'single');
        labelReal = ones(1, 1, 1, numel(batch), 'single');
        
        % if using gpus, convert the array to gpuArray
        if numGpus>0
            maskC = structfun(@gpuArray, maskC, 'UniformOutput', false) ;
            maskD = structfun(@gpuArray, maskD, 'UniformOutput', false) ;
            labelFake = gpuArray(labelFake);
            labelReal = gpuArray(labelReal);
        end
        
        if strcmp(mode, 'train')
            % -------------------
            % training object
            % --------------------
            if strcmp(params.trainingObject, "generator")
                % if the accumulateParamDers is equal to 0, the derivative will
                % be recalculated in a bp
                netG.accumulateParamDers = 0;
                % eval({input},{ders}) will complete a forward propagation and
                % a backward propagation
                netG.eval({'original_images', original_images, 'mask', maskC.mask_array}, ...
                    {'mse_loss', 1});
                GLoss = netG.getVar('mse_loss');
                GLoss = gather(GLoss.value);
                % mseLoss should be a scalar
                % update netG
                if ~isempty(parservG)
                    parservG.sync(); 
                end
                stateG = accumulateGradients(netG, stateG, params, parservG);
                
            elseif strcmp(params.trainingObject, "discriminator") || strcmp(params.trainingObject, "combination")
                % netG.eval({input}) complete a forward propagation without a
                % backward propagation
                netG.eval({'original_images', original_images, 'mask', maskC.mask_array});
                % get the completed images
                completedImages = netG.getVar('completed_images');
                completedImages = completedImages.value;
                
                % train discriminator with fake and real data
                netD.accumulateParamDers = 0 ;
                local_images_area_fake = get_local_area(completedImages, maskC);
                local_images_area_real = get_local_area(original_images, maskD);
                netD.eval({'local_disc_input', cat(4, local_images_area_fake, local_images_area_real),...
                    'global_disc_input',cat(4, completedImages, original_images) , ...
                    'labels',cat(4, labelFake, labelReal)}, ...
                    {'sigmoid_cross_entropy_loss',1}, 'holdOn', 0) ;
                DLoss = netD.getVar('sigmoid_cross_entropy_loss');
                DLoss = gather(DLoss.value);
                % update netD
                if ~isempty(parservD)
                    parservD.sync(); 
                end
                stateD = accumulateGradients(netD, stateD, params, parservD);
                
                % --------------------------------
                if strcmp(params.trainingObject, "combination")
                    % calculate the gan loss of generator
                    netD.accumulateParamDers = 0 ;
                    netD.eval({'local_disc_input', local_images_area_fake, 'global_disc_input',completedImages , 'labels',labelReal}, ...
                        {'sigmoid_cross_entropy_loss', alpha}, 'holdOn', 0);
                    GLoss = netD.getVar('sigmoid_cross_entropy_loss');
                    GLoss = gather(GLoss.value) * alpha;
                    % get the derivative from local dicriminator and the global
                    % dicriminator for generator's backwarking
                    df_dg = get_der_from_discriminator(netD, maskC);
                    
                    % eval generator
                    netG.accumulateParamDers = 0;
                    % netG can use backward propagation from the completed_images layer instead of the loss layer
                    netG.eval({'original_images', original_images, 'mask', maskC.mask_array}, ...
                        {'completed_images', df_dg}, 'holdOn', 1);
                    % calculate the mse loss of the generator
                    netG.accumulateParamDers = 1;
                    netG.eval({'original_images', original_images, 'mask', maskC.mask_array}, ...
                        {'mse_loss', 1}, 'holdOn', 0);
                    
                    mseLoss = netG.getVar('mse_loss');
                    mseLoss = gather(mseLoss.value);
                    GLoss = GLoss + mseLoss;
                    % update netG
                    if ~isempty(parservG)
                        parservG.sync();
                    end
                    stateG = accumulateGradients(netG, stateG, params, parservG);
                end
            else
                error('wrong params.trainingObject:%s', params.trainingObject);
            end
            % test mode
        else
            netG.forward({'original_images', original_images, 'mask', maskC.mask_array})
            completedImages = netG.getVar('completed_images');
            completedImages = completedImages.value;
            
            local_images_area_fake = get_local_area(completedImages, maskC);
            local_images_area_real = get_local_area(original_images, maskD);
            netD.eval({'local_disc_input', cat(4, local_images_area_fake, local_images_area_real),...
                'global_disc_input',cat(4, completedImages, original_images) , ...
                'labels',cat(4, labelFake, labelReal)}) ;
        end

        % Get statistics.
        time = toc(start) + adjustTime ;
        batchTime = time - stats.time ;
        stats.num = num ;
        stats.time = time ;
        
        currentSpeed = batchSize / batchTime ;
        averageSpeed = (t + batchSize - 1) / time ;
        
        stats = params.extractStatsFn(stats,netD);
        stats = params.extractStatsFn(stats,netG);
        
        if t == 3*params.batchSize + 1
            % compensate for the first three iterations, which are outliers
            adjustTime = 4*batchTime - time ;
            stats.time = time + adjustTime ;
        end
        fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
        
        if strcmp(mode, 'train')
            switch params.trainingObject
                case 'generator'
                    stats.GLoss = stats.GLoss + GLoss;
                    fprintf('\t GLoss: %.3f\n', GLoss) ;
                case 'discriminator'
                    stats.DLoss = stats.DLoss + DLoss;
                    fprintf('\t DLoss: %.3f\n', DLoss);
                case 'combination'
                    stats.GLoss = stats.GLoss + GLoss;
                    stats.DLoss = stats.DLoss + DLoss;
                    fprintf('\t GLoss: %.3f DLoss: %.3f\n', GLoss, DLoss) ;
                otherwise
                    error('wrong training object')
            end
            % ----------
            % save sample images
            % ----------
            if mod(batchIndex, params.sample_save_per_batch_count)==0 && labindex==1
                path = sprintf('./pics/epoch_%d_%d.png', params.epoch, batchIndex);
                completedImages = netG.getVar('completed_images');
                completedImages = completedImages.value;
                save_sample_images(completedImages, [4, 4], path);
                fprintf('save sample images as %s\n.', path);
            end
        end
        
    end
    
    if strcmp(mode, 'train')
        stats.GLoss = stats.GLoss / numel(subset) * numlabs ;
        stats.DLoss = stats.DLoss / numel(subset) * numlabs;
    end
    
    % Save back to state.
    stateG.stats.(mode) = stats ;
    stateD.stats.(mode) = stats;
    
    if params.profile
        if numGpus <= 1
            stateG.prof.(mode) = profile('info') ;
            stateD.prof.(mode) = profile('info') ;
            profile off ;
        else
            stateG.prof.(mode) = mpiprofile('info');
            stateD.prof.(mode) = mpiprofile('info');
            mpiprofile off ;
        end
    end
    if ~params.saveSolverState
        stateG.solverState = [] ;
        stateD.solverState = [] ;
    else
        for i = 1:numel(stateG.solverState)
            s = stateG.solverState{i} ;
            if isnumeric(s)
                stateG.solverState{i} = gather(s) ;
            elseif isstruct(s)
                stateG.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
            end
        end
        for i = 1:numel(stateD.solverState)
            s = stateD.solverState{i} ;
            if isnumeric(s)
                stateD.solverState{i} = gather(s) ;
            elseif isstruct(s)
                stateD.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
            end
        end
    end
    
    netG.reset() ;
    netG.move('cpu') ;
    netD.reset() ;
    netD.move('cpu') ;
    
    state = [stateG, stateD];
    net = [netG netD];
end
% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, parserv)
% -------------------------------------------------------------------------
    numGpus = numel(params.gpus) ;
    otherGpus = setdiff(1:numGpus, labindex) ;
    numWorkers = max(1,numGpus) * params.numSubBatches ;

    for p=1:numel(net.params)

        if ~isempty(parserv)
            parDer = parserv.pullWithIndex(p) ;
        else
            parDer = net.params(p).der ;
        end

        switch net.params(p).trainMethod

            case 'average' % mainly for batch normalization
                thisLR = net.params(p).learningRate ;
                net.params(p).value = vl_taccum(...
                    1 - thisLR, net.params(p).value, ...
                    (thisLR/numWorkers/net.params(p).fanout),  parDer) ;

            case 'gradient'
                thisDecay = params.weightDecay * net.params(p).weightDecay ;
                thisLR = params.learningRate * net.params(p).learningRate ;

                if thisLR>0 || thisDecay>0
                    % Normalize gradient and incorporate weight decay.
                    parDer = vl_taccum(1/numWorkers, parDer, ...
                        thisDecay, net.params(p).value) ;

                    if isempty(params.solver)
                        % Default solver is the optimised SGD.
                        % Update momentum.
                        state.solverState{p} = vl_taccum(...
                            params.momentum, state.solverState{p}, ...
                            -1, parDer) ;

                        % Nesterov update (aka one step ahead).
                        if params.nesterovUpdate
                            delta = params.momentum * state.solverState{p} - parDer ;
                        else
                            delta = state.solverState{p} ;
                        end

                        % Update parameters.
                        net.params(p).value = vl_taccum(...
                            1,  net.params(p).value, thisLR, delta) ;

                    else
                        % call solver function to update weights
                        [net.params(p).value, state.solverState{p}] = ...
                            params.solver(net.params(p).value, state.solverState{p}, ...
                            parDer, params.solverOpts, thisLR) ;
                    end
                end
            otherwise
                error('Unknown training method ''%s'' for parameter ''%s''.', ...
                    net.params(p).trainMethod, ...
                    net.params(p).name) ;
        end
    end
end
% get local area
function local_area = get_local_area(batch_images, Mask)
    la_h_s = Mask.local_area_top_left_point(1);
    la_w_s = Mask.local_area_top_left_point(2);
    la_size_h = Mask.local_area_size(1);
    la_size_w = Mask.local_area_size(2);
    local_area = batch_images(la_h_s:la_h_s+la_size_h-1, la_w_s:la_w_s+la_size_w-1, :, :);
end
% get the derivative from local dicriminator and the global
% dicriminator for generator's backwarking
function der = get_der_from_discriminator(netD, Mask)
    netD_local_input_der = netD.getVar('local_disc_input');
    netD_local_input_der = netD_local_input_der.der;
    netD_global_input_der = netD.getVar('global_disc_input');
    netD_global_input_der = netD_global_input_der.der;
    der = netD_global_input_der;
    la_h_s = Mask.local_area_top_left_point(1);
    la_w_s = Mask.local_area_top_left_point(2);
    la_size_h = Mask.local_area_size(1);
    la_size_w = Mask.local_area_size(2);
    der(la_h_s:la_h_s+la_size_h-1, la_w_s:la_w_s+la_size_w-1, :, :) = ...
        der(la_h_s:la_h_s+la_size_h-1, la_w_s:la_w_s+la_size_w-1, :, :) + netD_local_input_der;
end
% save a sample
function save_sample_images(images, arrangement, path)
    if isa(images, 'gpuArray')
        images = gather(images);
    end
    % show generated images
    sz = size(images) ;
    row = arrangement(1);
    col = arrangement(2);
    im = zeros(row*sz(1), col*sz(2),3, 'uint8');
    for ii=1:row
        for jj=1:col
            idx = col*(ii-1)+jj ;
            if idx<=sz(4)
                im((ii-1)*sz(1)+1:ii*sz(1),(jj-1)*sz(2)+1:jj*sz(2),:) = imsingle2uint8(images(:,:,:,idx)) ;
            end
        end
    end
    imwrite(im, path);
end

% -------------------------------------------------------------------------
function imo = imsingle2uint8(im)
% -------------------------------------------------------------------------
    mini = min(im(:));
    im = im - mini;
    maxi = max(im(:));
    if maxi<=0
        maxi = 1;
    end
    imo = uint8(255 * im ./ maxi);
end