function [ J Grad ] = compute_cost( X, Y, initial_theta )
%compute_cost Summary of this function goes here
    J = 1;
    [m n] = size(X);
    Grad = 1;
    z = X*initial_theta;
    h_theta = sigmoid(z);
    J = -1*(sum((Y .* log(h_theta)) + (  (1 - Y).*log(1 - h_theta)))) / m;
%     size(repmat((h_theta - Y), 1, size(X,2)))
    Grad = (1 / m) * sum( X .* repmat((h_theta - Y), 1, size(X,2)) );
    Grad = transpose(Grad);
end

