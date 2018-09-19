import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, lasso_path
from sklearn import cross_validation
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from itertools import cycle
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Hitters.csv').dropna()
y = data.Salary
X = data.drop(columns=['Name', 'League', 'Division', 'NewLeague', 'Salary'])

alphas = np.logspace(9,-3,100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Ridge
coeffs = []

for a in alphas:
    ridge = Ridge(alpha = a, normalize=True, fit_intercept=False)
    ridge.fit(X, y)
    coeffs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coeffs)
ax.set_xscale('log')
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.legend(list(X))
plt.axis('tight')
plt.show()

ridgecv = RidgeCV(alphas = alphas, normalize = True, cv = 10)
ridgecv.fit(X, y)
print("ridge optimal alpha: ", ridgecv.alpha_)

print("Mean Squared Error(Ridge):",mean_squared_error(y, ridge.predict(X)))

# LASSO
eps = 10e-8
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps)

ax = plt.gca()
neg_log_alphas_lasso = -np.log10(alphas_lasso)
colors = cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
for coef, c in zip(coefs_lasso, colors):
    plt.plot(neg_log_alphas_lasso, coef, c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.legend(list(X))
plt.axis('tight')
plt.show()


lassocv = LassoCV(alphas = None, max_iter = 100000, normalize = True, cv = 10)
lassocv.fit(X, y)
print("lasso optimal alpha", lassocv.alpha_)

lasso = Lasso(max_iter = 100000, normalize = True)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X, y)
count = 0
for coef in lasso.coef_:
    if coef != 0:
        count = count + 1
print("The number of predictors left in the model is:", count)

print("Mean Squared Error(Lasso):", mean_squared_error(y, lasso.predict(X)))

