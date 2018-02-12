#开始学习sklearn

#第一天
#1.1.1. Ordinary Least Squares

#LinearRegression fits a linear model with coefficients w = (w_1, ..., w_p) to minimize the residual
# sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.
# Mathematically it solves a problem of the form:
# \underset{w}{min\,} {|| X w - y||_2}^2

from sklearn import linear_model
reg = linear_model.LinearRegression()
xoy = [[0,0],[1,1],[2,2]]
#X, y, sample_weight=None
sample_weight = [0,1,2]
reg.fit(xoy,sample_weight)
LinearRegression()