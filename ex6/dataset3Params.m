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




range = [0.01 0.03 0.1 0.3 1 3 10 30];




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
for a = 1:length(range)
    for b = 1:length(range)
	C = range(a);
        sigma = range(b);      
	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));      
	predictions = svmPredict(model, Xval);

        %we can use 2 methods to select a and b to pass on as C(a) and sigma(b)
	%First -  we can compare predictions and yval and sum all the values in comparison vector
	%we can then select value of a and b index where the sum_compare matrix gives maxiumum value
	%(max value because for all rightly predicted output we are assigning 1 hence larger the value, better the params C and sigma)
        
        %sum_compare(a,b) = sum(predictions == yval);

        %Second -  we can compare predictions and yval and find mean error for all values that we predicted wrong
        %we can then select value of a and b index where the mean_error matrix gives minimum value

        mean_error(a,b) = mean(double(predictions ~= yval));

    endfor
endfor

%following gives indexes a and b for: max value in sum_compare method matrix 
%[maxValForSumCompare, a] = max(max(sum_compare,[],2));    
%[maxValForSumCompare, b] = max(max(sum_compare,[],1));
    
%following gives indexes a and b for: min value in mean_error method matrix 
[minValForMeanError, a] = min(min(mean_error,[],2));    
[minValForMeanError, b] = min(min(mean_error,[],1));

C = range(a);
sigma = range(b);

% =========================================================================

end
