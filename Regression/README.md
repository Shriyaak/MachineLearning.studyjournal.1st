## SIMPLE LINEAR REGRESSION: 
In this project, I begin by exploring Simple Linear Regression, which is a foundational regression technique used to model the relationship between a single independent variable (X1) and a dependent variable (y). The goal of this model is to find the best-fitting line that predicts y based on the value of X1. <br/>
y = β0 +β1 x1 + ϵ <br/> 
where,  <br/> 
y = dependent variable( target/ output)  <br/> 
β0 = intercept (the value of y when all xi = 0 ) <br/> 
x1, x2,... xp = independent variables( features/predictors)  <br/> 
β1, β2.. βp = coefficients of the independent variables (weights showing the impact of each predictor on y)  <br/> 
ϵ = Error term (captures the variablitiy in y not explained by the predictors)  <br/> 
<pre>
|                
|            + 
|        +
|    +
|+_______________ 
</pre>
Objective:
The primary goal of Simple Linear Regression is to find the best values for the coefficients β0 and β1 that minimize the sum of squared errors (SSE), which represents the difference between the actual values yi and the predicted values y^i <br/>
​SSE = i=1∑n { (yi −y^ i ) }^ 2 <br/>
 
where,  <br/>
yi = actual value of the response variables ; <br/>
y^ i = Predicted values (y^ I = β0 + β1xi1 + … βpxip)  <br/>
The best fit line is such that the sum of the squares of the residuals is minimised <br/>

Data Used: <br/>
YearsExperience (X1): The independent variable representing years of experience. <br/>
Salary (y): The dependent variable representing the salary of an individual. <br/>
<br/>
<br/>
## MULTIPLE LINEAR REGRESSION: 
<pre>
|                
|            + 
|        +
|    +
|+_______________ 
</pre>
Next I explored Multiple Linear Regression to predict company profit based on various independent variables, such as R&D Spend, Administration Spend, Marketing Spend, and State. The goal is to analyze which factors contribute most to profit and determine which companies are performing better, not just based on maximum profit, but also by considering how each company allocates resources. <br/>
The general equation for Multiple Linear Regression is: <br/>
y = β0 +β1 x1 +β2 x2 + ⋯ +βp xp + ϵ <br/>

To find the best model, we use Backward Elimination:   <br/>
2. Backward Elimination <--  you need all-in for this  <br/>
step 1: select a significance level to stay in the model, α = 0.05 <br/>
step 2: fit the full model with all possible predictors <br/>
step 3: Consider the predictor with the highest p-value if P > α, go to step 4 or go to FINISH <br/>
step 4: Remove that predictor  <br/>
step 5: Fit the model without this variable  <br/>
step 6: Go back to step 3, repeat until P < α if so FINISH  <br/>
<br/>
Data Used:  <br/>
The dataset includes: <br/>
Independent Variables (X1, X2, X3, X4): <br/>
Dependent Variable (y): Profit: The profit of the company, which we aim to predict based on the independent variables. <br/>
<br/>
<br/>
## POLYNOMIAL LINEAR REGRESSION:
Polynomial Linear Regression is an extension of simple linear regression that allows us to model non-linear relationships between the dependent variable (y) and the independent variables (x). In simple linear regression, the relationship between the independent variable and the dependent variable is represented by a straight line. However, when the relationship between the variables is more complex and not linear, we use polynomial regression to fit a curved line to the data. <BR/>
y = b0 + b1x1 + b2(x1)^2 + ...bn(x1)^n <BR/> 
<Pre>
Y 
|             +        
|            +          
|           +           
|         +            
|      +                
|__+_______________ x 
</Pre>
<br/>
<br/>

## SUPPORT VECTOR REGRESSION: (linear-SVR) 
In Support Vector Regression (SVR), much like Simple Linear Regression (SLR), we consider a regression line. However, instead of just a single line, SVR introduces a "tube" around the line, referred to as the ε-insensitive tube (where ε = epsilon). This tube has a width of epsilon, measured vertically along the axis (not perpendicularly). The purpose of this tube is to define a margin of error within which we disregard any deviations or errors from the regression line. <BR/> 
<BR/>
Key Concepts: <BR/>
ε-insensitive Tube: <BR/>
- Think of this tube as a margin of tolerance for the model. Any data points that fall inside the tube are considered "close enough" to the regression line, and their errors are ignored. <BR/>
- Only points that fall outside the tube are treated as contributing to the error, and the magnitude of this error is calculated as the distance between the point and the nearest boundary of the tube (not the regression line itself). <BR/>
<BR/>
*Slack Variables (ξᵢ and ξᵢ)**: <BR/>
For points outside the tube, errors are represented by slack variables: <BR/>
ξᵢ* for points below the tube. <BR/>
ξᵢ for points above the tube. <BR/>
<BR/>
Objective Function: <BR/>
The goal of SVR is to minimize the following function: <BR/>
1/2 * (||w||)^2 + C ∑(ξi + ξi*) -> min    <BR/>
(implicit relationship involves transforming the data and modeling complex, non-linear dependencies without an explicit formula for the output.)  <BR/>
where, <BR/>
(||w||)^2 : Regulation term to control models complexity  <BR/>
C: A constant that balances the trade-off between fitting the data and maintaining a simpler model. <BR/>
ξi, ξi*: Slack variables representing errors for points outside the ε-insensitive tube. <BR/>
<BR/>
<BR/>

## DECISION TREE REGRESSION: <BR/>

<PRE> 
 Regression Tree 
    X2
    |      +  +
    | +  +   +  +
    | + + + + 
    | +  +     +
    |_____________ X1  
   /
  /
 
Y 
</PRE>

- we have got two independent variables X1 and X2 <BR/>
- and what we are predicting is the third variable a dependent variable ( Y is the third dimension)  <BR/>

so once you run the decision tree algorithm in the regression sense of it, your scatter plot will be split into segments(each one of these splits is called a leaf) <BR/>
- the final leafs are called the terminal leafs: <BR/>
<PRE> 
 SCATTER PLOT OF THE DATA: 
     X2 split1  split3
    | +  |    +|   +
    |  + |  +  |+ +
    | +  |+   +|
    | + +| + + |
170 |    |-----|----- split2 
    | +  |+     +
    |____|________ X1 
         20    50
</PRE>
- The algorithm decides where to split the data by measuring something called information entropy, a mathematical concept that checks whether a split actually improves how well the data is grouped. If a split helps make the groups more meaningful, it's considered useful.<BR/>
- The splitting process continues until the improvement (information gain) becomes too small. When the additional value from a split drops below a certain threshold, the algorithm stops making further splits <BR/>
<PRE> 
 Decision Tree: 

so our 1st split happend at 20 so: 

                 X1 < 20 
             yes  /    \  no 
                 /      \

split 2 happens at 170 & only happens for points greater than 20 so: 

                  X1 < 20 
             yes  /    \  no 
                 /      \ 
         300.05       X2 < 170
                      yes /  \ no 
                         /    \
                    65.7    x3 < 50 
                         yes /  \ no 
                            /    \
                        -64.01   0.7 } average y-values 
</PRE>
- We check which group (leaf) the new point belongs to, based on its features. <BR/>
- Each group already has some known Y-values. <BR/>
- We find the average of those Y-values. <BR/>
- The new point is given this average as its predicted Y-value <BR/>
<br/>
<br/>

## RANDOM FOREST REGRESSION:  <br/>
step1: Pick at Random K data points from the training set <br/>
step2: Build the decision tree associated to these k data points <br/>
step3: Choose the number Ntrees f trees you want to build and repeat steps 1 & 2 <br/>
step4: For a new data point, make each one of your Ntree trees predict the value of Y. for the data point in question, and assign the new data point the average across all of the predicted Y values. <br/>
