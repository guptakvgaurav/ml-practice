function [] = Main()
%    Initialization
clear;
close all;
clc;

% load data
data = load('bank copy.csv');

% prepare data
alpha = 0.05;   % learning rate...
scaled_data = (data - min(data)) ./ (max(data) - min(data));
[ rows, columns ]= size(scaled_data);
X = [ ones(rows, 1) scaled_data(:, 1:1:columns-1)];
Y = scaled_data(:,columns);
initial_theta = rand(columns, 1);

% visualize raw data
figure;
hold on;
pos = find(Y == 1);
neg = find(Y == 0);
plot(X(pos, 7), X(pos, 12), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 7), X(neg, 12), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold off;


% determine theta using fminunc
resultTable = [];
for i = 1: 400
    % compute cost and gradient
    [J Grad] = compute_cost(X, Y, initial_theta);
    initial_theta = initial_theta - (alpha .* Grad);
%     fprintf('Iteration %d has cost %d\n', i, J)
    resultTable(i, :) =  [i J transpose(Grad)];
end

final_theta = initial_theta;  % just for sake of understanding.

% visualize cost
figure;
hold on;
plot(resultTable(:,1), resultTable(:, 2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(resultTable(:,4), resultTable(:, 6), 'g+', 'MarkerSize', 5, 'LineWidth', 1);
xlabel('iteration');
ylabel('Cost');
hold off;
csvwrite('result.csv', resultTable);

% predict data
prediction = sigmoid(X*final_theta);
comparision = [prediction Y]
% compute accuracy of prediction
end