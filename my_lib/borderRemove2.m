% Remove the border of the input image, using the following procedure:
% - segment foreground using Peter Kovesi algorithm
% (http://www.peterkovesi.com/matlabfns/)
% - compute the smallest enclosing rectangle of the foreground

function imseg = borderRemove2(img)
    %% Check input
    if isempty(img)
        error('Input image cannot be empty');
    end
    
    %% Estimate ridge frequency
    % Apply a gaussian window to reduce the effects of the border on the
    % fft
    tmp = img .* hann2D(size(img,1), size(img,2));
    % Normalize the input image to have 0 mean and 1 stddev
    tmp = (tmp-mean(tmp(:))) ./ std(tmp(:));
    % Compute the spectrum
    cartfs = fftshift(abs(fft2(tmp)));
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
    % Use the algorithm provided by Peter Kovesi
    [~, foreground, ~] = ridgesegment(img, 8, 0.2);
    
    % Compute smallest bounding box
    [row, col] = find(foreground);
    l = min(col(:));
    r = max(col(:));
    u = min(row(:));
    d = max(row(:));
    
    % Cut to that box
    imseg = img(u:d, l:r);
end