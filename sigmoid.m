function [ gz ] = sigmoid( z )
%sigmoid Summary of this function goes here
%   Detailed explanation goes here
    gz = 1 ./ ( 1 + exp(-1*z));
end

