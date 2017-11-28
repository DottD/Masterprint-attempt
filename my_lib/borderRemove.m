% Remove the border of the input image, using the following procedure:
% - for each pixel compute the stddev in a circle neighborhood of radius r
% - if it is less than p-percentile consider it backgroud
% - compute the smallest enclosing rectangle of the foreground

function imseg = borderRemove(img, p, range)
    %% Check input
    if isempty(img)
        error('Input image cannot be empty');
    end
    if p > 1 || p < 0
        error('p must be in [0,1]');
    end
    if all(mod(range,1)~=0) || numel(range)~=2
        error('range must be a pair of integers (left, right) extrema');
    end
    
    %% Estimate ridge frequency
    % Make a copy of the initial image
    orig = img;
    % Apply a gaussian window to reduce the effects of the border on the
    % fft
    img = img .* hann2D(size(img,1), size(img,2));
    % Normalize the input image to have 0 mean and 1 stddev
    img = (img-mean(img(:))) ./ std(img(:));
    % Compute the spectrum
    cartfs = fftshift(abs(fft2(img)));
    % Crop it to a square with side equal to the shortest dimension
    cartfs = cropToSquare(cartfs);
    % Convert in polar coordinate and sum along different angles
    circles_energy = sum(cart2pol(cartfs, 16),2);
    % Find peaks and remove the ones too low
    span = range(1):range(2);
    [~, locs, w, prom] = findpeaks(log(circles_energy(span)), span);
    [~,idx] = sort(w(:).*prom(:), 'descend');
    % Compute the frequency
    r = size(cartfs,1)/locs(idx(1));
    
    %% Segmentation
    % Take r as half the frequency
    r = ceil(r/2);
    % Create a circular kernel
    [x,y] = meshgrid(-r:r);
    kernel = double(x.^2+y.^2 <= r^2);
    kernel = kernel ./ sum(kernel(:));
    
    % Compute the mean in each neighborhood
    m = flexConv2(orig, kernel, 'mirror');
    s = sqrt(flexConv2((orig-m).^2, kernel, 'mirror'));
    
    % Compute the threshold for s as the p-percentile
    sorted_s_values = sort(s(:),'ascend');
    thres = sorted_s_values( round( p*(numel(s)-1) )+1 );
    
    % Segment image according to s
    foreground = s >= thres;
    
    % Compute smallest bounding box
    [row, col] = find(foreground);
    l = min(col(:));
    r = max(col(:));
    u = min(row(:));
    d = max(row(:));
    
    % Cut to that box
    imseg = orig(u:d, l:r);
end