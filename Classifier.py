#sklearn pipeline
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('alipay_huabei_GS1.csv', header=None)
#print(data.shape)
x, y = data.values[1:30000,0:23],data.values[1:30000,24]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2,random_state= 666)
#Use pipeline mechanism
from sklearn.preprocessing import StandardScaler #Normalize so that the mean of each feature is 1 and the variance is 0
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=666))]) #Set a random seed to reproduce the test results
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',GaussianNB())]) #Priority is Naive Bayes of Gaussian distribution, the distribution of sample features is mostly continuous values
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',BernoulliNB())]) #Naive Bayesian distribution of Bernoulli, most of the distribution of sample features are multivariate discrete values
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',DecisionTreeClassifier())]) #Decision tree
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',svm.SVC())]) #C-support vector classification
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',svm.SVR())]) #Epsilon-support vector regression
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',KNeighborsClassifier())]) #KNN classifier
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',RandomForestClassifier())]) #Random Forest
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',ExtraTreesClassifier())]) #Extremely Randomized Trees
# pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',AdaBoostClassifier())]) #Adaboost



#test accuracy 0.795,0.795,0.779,0.714,0.800,0.080,0.781,0.777,0.772,0.799
# By compare the test accuracy, C-support vector classification and Adaboost classifier performs better.


k_range = range(1, 31)  # Optimize the value range of parameter k
weight_options = ['uniform', 'distance']  # Value range of parameter weights. uniform is the uniform weighting value, distance means the reciprocal of the distance
param_grid = {'n_neighbors':k_range,'weights':weight_options}  # Define the optimization parameter dictionary, the key value in the dictionary must be the parameter name of the function of the classification algorithm
pipe = Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=2)),('clf',GridSearchCV(estimator =KNeighborsClassifier(n_neighbors=5), param_grid = param_grid, cv=10, scoring='accuracy'))]) #GridSearchCV 
#After using the GridSearchCV method based on KNN classifier, the accuracy rate imporve 2.3%



pipe.fit(x_train, y_train)
print('Test accuracy is %.3f' % pipe.score(x_test, y_test))
