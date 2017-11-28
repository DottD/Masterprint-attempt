function varargout = imcmp(varargin)
%% Check input
if nargin <= 1
    error('At least two images expected');
end
rows = floor(sqrt(nargin));
cols = ceil(sqrt(nargin));
%% Create the window
h = figure('Name','Figure Comparison');
for k = 1:nargin
    subplot(rows, cols, k);
    imagesc(varargin{k});
    axis('image');
end
%% Check output
if nargout == 1
    varargout{1} = h;
elseif nargout >= 2
    error('At most one output argument');
end