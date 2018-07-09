% --------
run(fullfile(fileparts(mfilename('fullpath')), ...
    '..','..', 'matlab', 'vl_setupnn.m')) ;

opts.gpus = [1];
opts.expDir = fullfile(vl_rootnn, 'exp', 'celeba-glcic') ;
opts.dataDir = fullfile(vl_rootnn, 'data','celeba') ;
opts.testDir = fullfile(opts.dataDir, 'cropped_for_glcic_test');
opts.saveDir = './results';
opts.mask_range = [32, 32];
opts.local_area_size = [64, 64];
opts.batch_size = 64;

% ------------
read_images_opts.useGpu = numel(opts.gpus) > 0;
read_images_opts.numThreads = 12;
read_images_opts.imageSize = [128 128 3];
read_images_opts.cropSize = 128/178;
read_images_opts.subtractAverage = [];

read_images_opts.jitterLocation = false ;
read_images_opts.jitterFlip = false ;
read_images_opts.jitterBrightness = 0 ;
read_images_opts.jitterAspect = 0 ;

if ~isempty(opts.gpus) && ~exist('gpus', 'var')
    clear vl_tmove vl_imreadjpeg ;
    gpus = gpuDevice(opts.gpus);
end

complete_images(opts, read_images_opts);

% ----------
function complete_images(opts, read_images_opts)
    % GLCIC_COMPLETION: image completion function 
    % complete images via a trained model
    
    % load model
    modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
    model_index = findLastCheckpoint(opts.expDir) ;
    load(modelPath(model_index), 'netG') ;
    fprintf('load model:net-epoch-%d.mat\n', model_index);
    netG = dagnn.DagNN.loadobj(netG) ;
    
    % --
    if ~isempty(opts.gpus)
        netG.move('gpu');
    end
    netG.mode = 'normal';
    netG.vars(netG.getVarIndex('completed_images')).precious = 1;
    netG.vars(netG.getVarIndex('masked_images')).precious = 1;
    % ----------------------
    % 
    % ----------------------
    % gather test images from dir
    total_images = dir(sprintf('%s/*.jpg', opts.testDir));
    total_images_path = cell(size(total_images));
    total_images_names = cell(size(total_images));
    for i=1:size(total_images_path)
        total_images_path(i)= cellstr(fullfile(total_images(i).folder, total_images(i).name));
        total_images_names(i) = cellstr(total_images(i).name);
    end
    images_count = size(total_images_path, 1);

    for i= 1:ceil(images_count/opts.batch_size)
        % split images
        if i<ceil(images_count/opts.batch_size)
            images_path_batch = total_images_path( (i-1) * opts.batch_size+1: i * opts.batch_size, :);
        else
            images_path_batch = total_images_path((i-1) * opts.batch_size+1:end, :);
        end
        images_batch = getImageBatch(images_path_batch, read_images_opts);
        images_batch = (images_batch - 128) / 128;
        mask = create_random_mask(size(images_batch), opts.local_area_size, opts.mask_range);
        
        if ~isempty(opts.gpus)
            mask = structfun(@gpuArray, mask, 'UniformOutput', false) ;
        end
        
        % ------------
        % complete images
        % ------------
        netG.eval({'original_images', images_batch, 'mask', mask.mask_array});
        completed_images = netG.getVar('completed_images');
        completed_images = gather(completed_images.value);
        masked_images = netG.getVar('masked_images');
        masked_images = gather(masked_images.value);
        images_batch = gather(images_batch);
        % save images
        for j=1:size(images_batch, 4)
            [~, images_name, ~] = fileparts(char(images_path_batch(j)));
            name = sprintf('%s_original.png', images_name);
            imwrite(images_batch(:,:,:,j), fullfile(opts.saveDir, name));
            
            name = sprintf('%s_masked.png', images_name);
            imwrite(masked_images(:,:,:,j), fullfile(opts.saveDir, name));
            
            name = sprintf('%s_completed.png', images_name);
            imwrite(completed_images(:,:,:,j), fullfile(opts.saveDir, name));
        end
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
