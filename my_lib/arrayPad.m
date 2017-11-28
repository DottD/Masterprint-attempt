% Performs padding using the following extrapolation:
% - constant, CCC|acbdefgh|CCC (C is the last argument)
% - mirror, cba|abcdefgh|hgf
% - mirror101, dcb|abcdefgh|gfe
% - replicate, aaa|abcdefgh|hhh

function out = arrayPad(im, bd, extrapolation, varargin)
    % Check that r and im are compatible
    if ~isscalar(bd) && ndims(im) ~= length(bd)
        error('r must be a scalar or have an element for each dimension of im');
    end
    if isscalar(bd)
       bd = repmat(bd, 1, ndims(im)); 
    end
    if bd >= size(im)
        error('r must be less than the number of elements of im');
    end
    % Padding
    switch extrapolation
        case 'constant'
            % 0 in the border
            if nargin == 4 && isscalar(varargin{1})
               out = ones( size(im)+2*bd ) .* varargin{1};
               out( bd(1)+(1:size(im,1)), bd(2)+(1:size(im,2)) ) = im;
            else
                error('You must specify a constant after extrapolation');
            end
        case 'mirror'
            out = vertcat( flipud(im(1:bd(1),:)) , im , flipud(im(end-bd(1)+1:end,:)));
            outl = vertcat( rot90(im(1:bd(1),1:bd(2)),2), fliplr(im(:,1:bd(2))), rot90(im(end-bd(1)+1:end,1:bd(2)),2) );
            outr = vertcat( rot90(im(1:bd(1),end-bd(2)+1:end),2), fliplr(im(:,end-bd(2)+1:end)), rot90(im(end-bd(1)+1:end,end-bd(2)+1:end),2) );
            out = horzcat( outl, out, outr );
        case 'mirror101'
            out = vertcat( flipud(im(2:bd(1)+1,:)) , im , flipud(im(end-bd(1):end-1,:)));
            outl = vertcat( rot90(im(2:bd(1)+1,2:bd(2)+1),2), fliplr(im(:,2:bd(2)+1)), rot90(im(end-bd(1):end-1,2:bd(2)+1),2) );
            outr = vertcat( rot90(im(2:bd(1)+1,end-bd(2):end-1),2), fliplr(im(:,end-bd(2):end-1)), rot90(im(end-bd(1):end-1,end-bd(2):end-1),2) );
            out = horzcat( outl, out, outr );
        case 'replicate'
            out = vertcat( repmat(im(1,:), bd(1), 1) , im , repmat(im(end,:), bd(1), 1) );
            outl = vertcat( ones(bd)*im(1,1), repmat(im(:,1), 1, bd(2)), ones(bd)*im(end,1) );
            outr = vertcat( ones(bd)*im(1,end), repmat(im(:,end), 1, bd(2)), ones(bd)*im(end,end) );
            out = horzcat( outl, out, outr );
        otherwise
            error('Extrapolation type not allowed');
    end
end