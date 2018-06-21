function preprocess_celeba(data_dir, save_dir_name)
    % PREPROCESS_CELEBA prepares celeba data for training

    % This file is part of the VLFeat library and is made available under
    % the terms of the BSD license (see the COPYING file).
    
    % opeartion order: (1)crop  (2)resize
    crop_size = [178, 178];
    resize_size = [178, 178];

    d = dir(sprintf('%s/img_align_celeba/*.jpg', data_dir));
    if isempty(d), error('No images found in %s', data_dir); end
    in_dir = sprintf('%s/img_align_celeba/', data_dir);
    out_dir = sprintf('%s/%s/',data_dir, save_dir_name);

    mkdir(out_dir);

    for i=1:numel(d)
        if mod(i,100)==1
            fprintf('%d // %d\n',i,numel(d));
        end
        im = imread(fullfile(in_dir,d(i).name));
        %   figure(1),imshow(im);

        [h,w,~] = size(im);
        crop_h_start = floor( (h-crop_size(1))/2 );
        crop_w_start = floor( (w-crop_size(2))/2 );
        im = im(crop_h_start+1:crop_h_start+crop_size(1), crop_w_start+1:crop_w_start+crop_size(2), : );

        if ~isequal(crop_size, resize_size)
            im = imresize(im, resize_size);
        end

        imwrite(im,fullfile(out_dir, d(i).name));
        %   figure(2),imshow(imc);
        %   pause();
    end
end
