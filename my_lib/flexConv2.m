% This function allows to compute the bidimensional convolution
% deciding the type of extrapolation among:
% - constant, the same as conv2
% - mirror, cba|abcdefgh|hgf
% - mirror101, dcb|abcdefgh|gfe
% - replicate, aaa|abcdefgh|hhh
% As for the other arguments, see the conv2 documentation

function out = flexConv2(A, B, extrapolation)
    % If extrapolation is costant, then use conv2
    if strcmp( extrapolation, 'constant' )
        out = conv2(A, B, 'same');
        return
    end
    % Pad the initial array
    cpos = ceil( (size(B)+1)/2 ); % position of the center of B
    AA = arrayPad(A, cpos-1, extrapolation);
    % Perform the convolution
    out = conv2(AA, B, 'same');
    % Remove the padded part
    out = out( cpos(1)+(0:size(A,1)-1), cpos(2)+(0:size(A,2)-1) );
end