## Description

This is an octave prototype to train a SVM algorithm for given dataset. In the first half of this exercise, we will be using support vector machines (SVMs) with various example 2D datasets.
Then we will implement a SVM algorithm to train and test a spam email classifier.

## Sample Invocation

After running Octave cli, in the project directory input the following to execute the algorithm
```
ex6
```
For running the spam email classifier algorithm invoke ex6_spam:
```
ex6_spam
```

## General Procedure

**1.**  It is better to visualise the data first before execution so a plotting function is used. 
![Visualising data](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/1Visualise.png)

For C=1 the decision boundary is:
![Cequals1](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/2Cequals1.png)

For C=100 the decision boundary is:
![Cequals100](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/3Cequals100.png)

**2.** Next we implement SVM with a gaussian kernel as per the following formulation:
```
sim = exp(-sum((x1-x2).^2)/(2 * sigma^2));
```
![SampleKernelValue](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/4SampleGaussianKernelEvaluation.png)

**3.** We then load a new dataset as visualised below. By using SVM with gaussian kernel we can get a non-linear decision boundary that fits well for the given dataset.
![Dataset2](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/5Dataset2Visualisation.png) 

![Dataset2Boundary](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/6Dataset2DecisionBoundry.png) 

**4.** For dataset 3 we experiment with training and cross validation sets to automatically find optimum values of S and sigma. 
The optimisation method can be found in file [dataset3Params.m](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/ex6/dataset3Params.m) 
![Dataset3Visualise](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/7Dataset3Visualise.png) 

![Cequals1SigmaEquals0.3](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/8Cequals1SigmaEquals0.3.png) 

![AutoselectedSigmaC](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/SVM%20outputs/9GaussianSVMWithAutoselectedC%26Sigma.png) 

---
**1.** Using same principles, we can use SVM to build a spam email classifier. First step is to preprocess the emails for normalising different formats. Refer to file [processEmail.m](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/ex6/processEmail.m)
 ```matlab
    %Avoiding for loops method to go for a vectorised implementation:
    comp = strcmp(str, vocabList);
    idx = find(comp);
    word_indices = [word_indices; idx];

    %For loops method:
    %for i = 1 : length(vocabList)
    %    if strcmp(str, vocabList{i})
    %        word_indices = [word_indices; i];
    %        break;
    %    end
    %end

```
Preprocessed email is as follows:

![Preprocessed](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/spamEmailOutputs/PreProcessedEmail.png) 

And the word indices corresponding to the processed email are:
![WordIndices](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/spamEmailOutputs/WordIndices.png) 

**2.** Now implement the feature extraction that converts each email into a vector.  The feature vector has length 1899.
```octave
%Avoiding for loops
x(word_indices) = 1;
```

Proceed to train a SVM to classify between spam (y = 1) and non-spam (y = 0) emails.
![spamEmailOutputs/Accuracy](https://github.com/kushalchaudhari21/SVMandSpamEmailClassification/blob/master/spamEmailOutputs/Accuracy.png)

## Important Insights

* C plays a role equivalent to (1/lambda).
* Large *C*(Small *λ*): Low Bias, High Variance. 
* Small *C*(Large *λ*): High Bias, Low Variance.
* Large *(σ^2)*: High Bias, Low Variance - Feature f1 varies more smoothly. 
* Small *(σ^2)*: Lower Bias, Higher Variance - Feature f1 varies less smoothly. 
* Let n = number of features and m = number of training examples:
```
- If n≥m(Eg. n=10000,m=10-1000): Use Logistic Regression or SVM without kernel. 
- If n is small and m is intermediate(Eg. n=1-1000,m=10-10000): Use SVM with Gaussian kernel. 
- If n is small and m is large(Eg. n=1-1000,m=50000+): SVM will be slow/struggling. Create/Add more features then use Logistic regression or SVM without a kernel.
```
