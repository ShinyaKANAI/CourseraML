function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];
% C_list = [0.01 0.1 1];
% sigma_list = [0.01 0.1 1];

err_list = zeros(size(C_list,2), size(sigma_list,2));

i_C = 1;
for C = C_list
    i_s = 1;
    for sigma = sigma_list
        fprintf('now : ');
        [n_C, n_s] = size(err_list);
        disp((i_C - 1) * n_s + i_s);
        fprintf('end : ');
        disp(n_C * n_s);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        err_list(i_C, i_s) = err;
        % fprintf('err_list here')
        % disp(err_list);
        i_s ++;
    end
    i_C ++;
end

[Mval, Mind] = min(err_list(:));
[minC, mins] = ind2sub(size(err_list), Mind);
C = C_list(minC);
sigma = sigma_list(mins);
fprintf('Selected C : ');
disp(C)
fprintf('Selected sigma : ');
disp(sigma);

% =========================================================================

end
