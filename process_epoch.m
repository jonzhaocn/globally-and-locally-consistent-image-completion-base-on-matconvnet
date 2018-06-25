function [netG, netD, state] = process_epoch(netG, netD, state, params, mode)
    % initialize with momentum 0
    if isempty(state) || isempty(state.solverStateG) || isempty(state.solverStateD)
        state.solverStateG = cell(1, numel(netG.params)) ;
        state.solverStateG(:) = {0} ;
        state.solverStateD = cell(1, numel(netD.params)) ;
        state.solverStateD(:) = {0} ;
    end
    
    % move CNN  to GPU as needed
    numGpus = numel(params.gpus) ;
    if numGpus >= 1
        netG.move('gpu') ;
        netD.move('gpu') ;
        
        for i = 1:numel(state.solverStateG)
            s = state.solverStateG{i} ;
            if isnumeric(s)
                state.solverStateG{i} = gpuArray(s) ;
            elseif isstruct(s)
                state.solverStateG{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
            end
        end
        for i = 1:numel(state.solverStateD)
            s = state.solverStateD{i} ;
            if isnumeric(s)
                state.solverStateD{i} = gpuArray(s) ;
            elseif isstruct(s)
                state.solverStateD{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
            end
        end
    end
    if numGpus > 1
        error('Multi-gpu is not supported!');
        %   parserv = ParameterServer(params.parameterServer) ;
        %   net.setParameterServer(parserv) ;
    else
        parserv = [] ;
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
    n = 0;
    stats.errorG = 0;
    stats.errorD = 0;
    
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
        batchCount = fix((t-1)/params.batchSize)+1;
        fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
            batchCount, ceil(numel(subset)/params.batchSize)) ;
        batchSize = min(params.batchSize, numel(subset) - t + 1) ;

        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) ;
        batchEnd = min(t+params.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end

        inputs = params.getBatch(params.imdb, batch) ;

        if params.prefetch
            if s == params.numSubBatches
                batchStart = t + (labindex-1) + params.batchSize ;
                batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
            params.getBatch(params.imdb, nextBatch) ;
        end
        
        MaskC = createRandomMask(size(inputs{2}), params.local_area_size, params.mask_range);
        MaskD = createRandomMask(size(inputs{2}), params.local_area_size, params.mask_range);
        original_images = inputs{2};
        labelFake = zeros(1, 1, 1, numel(batch), 'single');
        labelReal = ones(1, 1, 1, numel(batch), 'single');
        if numGpus>0
            MaskC = structfun(@gpuArray, MaskC, 'UniformOutput', false) ;
            MaskD = structfun(@gpuArray, MaskD, 'UniformOutput', false) ;
            labelFake = gpuArray(labelFake);
            labelReal = gpuArray(labelReal);
        end
        
        % -------------------
        % training object
        % --------------------
        switch params.trainingObject
            case 'generator'
                netG.mode = 'normal';
                netG.accumulateParamDers = 0;
                netG.eval({'original_images', original_images, 'mask', MaskC.mask_array}, ...
                    {'mse_loss', 1});
                mseLoss = netG.getVar('mse_loss');
                mseLoss = gather(mseLoss.value);
                % mseLoss should be a scalar
                errorG = mseLoss;
                % update netG
                state.solverState = state.solverStateG;
                state = accumulateGradients(netG, state, params, batchSize, parserv);
                state.solverStateG = state.solverState;
            case 'discriminator'
                netG.mode = 'normal';
                netG.eval({'original_images', original_images, 'mask', MaskC.mask_array});
                completedImages = netG.getVar('completed_images');
                completedImages = completedImages.value;

                % train discriminator with fake and real data
                netD.mode = 'normal' ;
                netD.accumulateParamDers = 0 ;

                local_images_area = crop_local_area(completedImages, MaskC);
                netD.eval({'local_disc_input',local_images_area, 'global_disc_input',completedImages , 'labels',labelFake, ...
                    'multiply_alpha', false}, {'sigmoid_cross_entropy_loss',1}, 'holdOn', 1) ;
                errorFake = netD.getVar('sigmoid_cross_entropy_loss');
                errorFake = gather(errorFake.value);
                netD.accumulateParamDers = 1 ;

                local_images_area = crop_local_area(original_images, MaskD);
                netD.eval({'local_disc_input',local_images_area, 'global_disc_input', original_images, 'labels',labelReal, ...
                    'multiply_alpha', false}, {'sigmoid_cross_entropy_loss',1}, 'holdOn', 0) ;

                errorReal = netD.getVar('sigmoid_cross_entropy_loss');
                errorReal = gather(errorReal.value);
                errorD = errorFake + errorReal;
                % update netD
                state.solverState = state.solverStateD;
                state = accumulateGradients(netD, state, params, 2 * batchSize, parserv);
                state.solverStateD = state.solverState;
            case 'combination'
                netG.mode = 'normal';
                netG.eval({'original_images', original_images, 'mask', MaskC.mask_array});
                completedImages = netG.getVar('completed_images');
                completedImages = completedImages.value;

                % train discriminator with fake and real data
                netD.mode = 'normal' ;
                netD.accumulateParamDers = 0 ;
                local_images_area = crop_local_area(completedImages, MaskC);
                netD.eval({'local_disc_input',local_images_area, 'global_disc_input',completedImages , 'labels',labelFake, ...
                    'multiply_alpha', true}, {'sigmoid_cross_entropy_loss',1}, 'holdOn', 1) ;
                errorFake = netD.getVar('sigmoid_cross_entropy_loss');
                errorFake = gather(errorFake.value);
                % -----
                netD.accumulateParamDers = 1 ;
                local_images_area = crop_local_area(original_images, MaskD);
                netD.eval({'local_disc_input',local_images_area, 'global_disc_input', original_images, 'labels',labelReal, ...
                    'multiply_alpha', false}, {'sigmoid_cross_entropy_loss',1}, 'holdOn', 0) ;

                errorReal = netD.getVar('sigmoid_cross_entropy_loss');
                errorReal = gather(errorReal.value);
                errorD = errorFake + errorReal;
                % update netD
                state.solverState = state.solverStateD;
                state = accumulateGradients(netD, state, params, 2 * batchSize, parserv);
                state.solverStateD = state.solverState;
                % --------------------------------
                % calculate the gan loss of generator
                netD.accumulateParamDers = 0 ;
                local_images_area = crop_local_area(completedImages, MaskC);
                netD.eval({'local_disc_input', local_images_area, 'global_disc_input',completedImages , 'labels',labelReal, ...
                    'multiply_alpha', true}, {'sigmoid_cross_entropy_loss', 1}, 'holdOn', 0);
                errorG = netD.getVar('sigmoid_cross_entropy_loss');
                errorG = gather(errorG.value);
                df_dg = get_der_from_discriminator(netD, MaskC);
                % cleanup der
                for p=1:numel(netD.params)
                    netD.params(p).der = [];
                end
                for v=1:numel(netD.vars)
                    netD.vars(v).der = [];
                end
                % eval generator
                netG.mode = 'normal';
                netG.accumulateParamDers = 0;
                % netG can use backward propagation from the completed
                % images layer instead of the loss layer
                netG.eval({'original_images', original_images, 'mask', MaskC.mask_array}, ...
                    {'completed_images', df_dg}, 'holdOn', 1);
                % calculate the mse loss of the generator
                netG.accumulateParamDers = 1;
                netG.eval({'original_images', original_images, 'mask', MaskC.mask_array}, ...
                    {'mse_loss', 1}, 'holdOn', 0);
                % update netG
                state.solverState = state.solverStateG;
                state = accumulateGradients(netG, state, params, 2 * batchSize, parserv);
                state.solverStateG = state.solverState;
            otherwise
                error('wrong params.trainingObject:%s', params.trainingObject);
        end
        % Get statistics.
        time = toc(start) + adjustTime ;
        batchTime = time - stats.time ;
        stats.num = num ;
        stats.time = time ;
        currentSpeed = batchSize / batchTime ;
        averageSpeed = (t + batchSize - 1) / time ;
        if t == 3*params.batchSize + 1
            % compensate for the first three iterations, which are outliers
            adjustTime = 4*batchTime - time ;
            stats.time = time + adjustTime ;
        end
        % loss may get inf values
        switch params.trainingObject
            case 'generator'
                stats.errorG = stats.errorG + errorG;
                fprintf(' errorG: %.3f', errorG/batchSize) ;
            case 'discriminator'
                stats.errorD = stats.errorD + errorD;
                fprintf(' errorD: %.3f', errorD/(batchSize * 2));
            case 'combination'
                stats.errorG = stats.errorG + errorG;
                stats.errorD = stats.errorD + errorD;
                fprintf(' errorG: %.3f errorD: %.3f', errorG/batchSize, errorD/(batchSize * 2)) ;
            otherwise
                error('wrong training object')
        end
        fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
        fprintf('\n') ;
        % ----------
        % save sample images
        % ----------
        if mod(batchCount, params.sample_save_per_batch_count)==0
           path = sprintf('./pics/epoch_%d_%d.png', params.epoch, batchCount);
           completedImages = netG.getVar('completed_images');
           completedImages = completedImages.value;
           save_sample_images(completedImages, [4, 4], path);
           fprintf('save sample images as %s\n.', path);
        end
    end
    
    stats.errorG = stats.errorG / numel(subset) ;
    stats.errorD = stats.errorD / numel(subset) ;
    % Save back to state.
    state.stats.(mode) = stats ;
    if params.profile
        if numGpus <= 1
            state.prof.(mode) = profile('info') ;
            profile off ;
        else
            state.prof.(mode) = mpiprofile('info');
            mpiprofile off ;
        end
    end
    if ~params.saveSolverState
        state.solverStateG = [] ;
        state.solverStateD = [] ;
    else
        for i = 1:numel(state.solverStateG)
            s = state.solverStateG{i} ;
            if isnumeric(s)
                state.solverStateG{i} = gather(s) ;
            elseif isstruct(s)
                state.solverStateG{i} = structfun(@gather, s, 'UniformOutput', false) ;
            end
        end
        for i = 1:numel(state.solverStateD)
            s = state.solverStateD{i} ;
            if isnumeric(s)
                state.solverStateD{i} = gather(s) ;
            elseif isstruct(s)
                state.solverStateD{i} = structfun(@gather, s, 'UniformOutput', false) ;
            end
        end
    end
    netG.reset() ;
    netG.move('cpu') ;
    netD.reset() ;
    netD.move('cpu') ;
end
% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
    numGpus = numel(params.gpus) ;
    otherGpus = setdiff(1:numGpus, labindex) ;
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
                    (thisLR/batchSize/net.params(p).fanout),  parDer) ;
                
            case 'gradient'
                thisDecay = params.weightDecay * net.params(p).weightDecay ;
                thisLR = params.learningRate * net.params(p).learningRate ;
                if thisLR>0 || thisDecay>0
                    % Normalize gradient and incorporate weight decay.
                    parDer = vl_taccum(1/batchSize, parDer, ...
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
% local area and mask array
function Mask = createRandomMask(batch_images_size, local_area_size, mask_range)
    images_h = batch_images_size(1);
    images_w = batch_images_size(2);
    
    local_area_h = local_area_size(1);
    local_area_w = local_area_size(2);
    
    if images_h < local_area_h || images_w < local_area_w
        error('images size should not smaller than local size');
    end
    if mask_range(1) > mask_range(2)
        error('wrong mask range');
    end
    if mask_range(2) > local_area_h || mask_range(2) > local_area_w
        error('max mask size should not bigger than local area size');
    end
    
    local_area_h_start = randi( [1, images_h-local_area_h+1], 1, 1);
    local_area_w_start = randi( [1, images_w-local_area_w+1], 1, 1);
    
    mask_size = randi(mask_range, 1, 2);
    mask_h = mask_size(1);
    mask_w = mask_size(2);
    
    mask_h_start = randi( [1, local_area_h-mask_h+1], 1, 1) + local_area_h_start-1;
    mask_w_start = randi( [1, local_area_w-mask_w+1], 1, 1) + local_area_w_start-1;
    
    mask_array = zeros(batch_images_size);
    mask_array(mask_h_start:mask_h_start+mask_h-1, mask_w_start:mask_w_start+mask_w-1, :, :) = 1;
    
    Mask.local_area_size = local_area_size;
    Mask.local_area_left_top_point = [local_area_h_start, local_area_w_start];
    Mask.mask_array = mask_array;
end
 
function local_area = crop_local_area(batch_images, Mask)
    la_h_s = Mask.local_area_left_top_point(1);
    la_w_s = Mask.local_area_left_top_point(2);
    la_size_h = Mask.local_area_size(1);
    la_size_w = Mask.local_area_size(2);
    local_area = batch_images(la_h_s:la_h_s+la_size_h-1, la_w_s:la_w_s+la_size_w-1, :, :);
end
function der = get_der_from_discriminator(netD, Mask)
    netD_local_input_der = netD.getVar('local_disc_input');
    netD_local_input_der = netD_local_input_der.der;
    netD_global_input_der = netD.getVar('global_disc_input');
    netD_global_input_der = netD_global_input_der.der;
    der = netD_global_input_der;
    la_h_s = Mask.local_area_left_top_point(1);
    la_w_s = Mask.local_area_left_top_point(2);
    la_size_h = Mask.local_area_size(1);
    la_size_w = Mask.local_area_size(2);
    der(la_h_s:la_h_s+la_size_h-1, la_w_s:la_w_s+la_size_w-1, :, :) = ...
        der(la_h_s:la_h_s+la_size_h-1, la_w_s:la_w_s+la_size_w-1, :, :) + netD_local_input_der;
end
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