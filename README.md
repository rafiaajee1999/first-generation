# first-generation
import sys
print('Python:{}'.format(sys.version))
import scipy
print('Scipy:{}'.format(scipy._version_))
import numpy
print(Numpy:{}'.format(numpy._version_))
import matplotlib
print('Matplotlib:{}'.format(matplotlib._version_))
import pandas
print('Pandas:{}'.format(pandas._version_))
import sklearn
print('Sklearn:{}'.format(sklearn._version_))
python:3.6.5|Anaconda,Inc.|(default,Apr29.2018,16:14)
[GCC 7.2.0]
scipy:1.1.0
numpy:1.14.5
matplotlib:2.2.2
pandas:0.23.4
sklearn:0.19.1
import pandas
fromat pandas import read_CSV
from pandas.plotting import scattermatrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import stratifiedkFlod
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Logistic Regression
from sklearn.tree import DecisionTreeclassifier
from sklearn.neighbors import kNeighborsclassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_boys import GaussianNB
from sklearn.SVM import SVC
from sklearn import model_selection
from sklearn.ensemble import voting classifier
#loading the data
url=
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_CSV(url,names=names)
#dimension of the dataset
print(dataset.shape)
(150,5)
#take a peak at the data
print(dataset.head(20))
#statistical summary
print(dataset.describe())
#class distribution
print(dataset.groupy('class').size())
#univariate plots_box and whisker plots
dataset.plot(kind='box', subplots=True, layout(2,2), sharen=Flase, sharey=Flase)
pyplot.show()
#histogram of the variable
dataset.hist()
pyplot.show()
#multivariate plots
scatter_matrix(dataset)
pyplot.show()
#creating a validation dataset
#splitting dataset
array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train, x_validation, y_train, y_va;idation: tarin_test_split(x,y, test_size+0.2, random_state=1)
#Logistic Regression
#Linear Discriminant Analysis
#K_Nearest neighbors
#Classification and Regression Tree
#Gaussian Naive Bayes
#Support Vector Machines
#building models
model=[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='over')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsclassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', (gamma='auto')))
#evalute the created models
result=[]
names=[]
for name, model in models:
kflod=stratified kflod(n_splits=10, random_state=1)
CV_result=cross_val_score(model, x_train, y_train, CV=kflod.scoring='accuracy')
results.append(CV_results)
name.append(name)
print('%s:%f(%f)'%(name, CV_results.mean(), CV_results.std()))
#Compare our models
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
#make prediction on SVM
model=SVC(gamma='auto')
model.fit(x_train, y_train)
predictions=model.predict(x_validation)
#evaluate our predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
