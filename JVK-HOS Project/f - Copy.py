from datetime import datetime, date
import CleanDefs
import matplotlib.pyplot as plt
import pandas
import numpy as np



"""****Split Report into lists by user****"""
fileName = "..\JVK-HOS Project\VK Report.txt"
def create_large_List(fileName):
    with open(fileName) as f:
       content = f.readlines()
    newContent = str(content).replace('\\n',"").split("USER") 
    largeList = []
    for i in newContent[1:]:
        i =i.split(" ")
        temp = filter(None, i)
        largeList += [temp]
    return largeList


def create_values_dict(largeList,valuesDict):
    
    for smallList in largeList:
        """"***Determine Index Numbers***"""
        try:
            InsurIndexNum = smallList.index("ICD-10")
        except:
            continue
        try:    
            indexNum =smallList.index("F")
        except:
            try:
                indexNum = smallList.index("M")
            except:
                pass
        """***Get personal Info***"""
        valuesDict = CleanDefs.get_personal_info(indexNum, valuesDict, smallList )
        
        
        """***Clean Procedure names***"""
        x =next(i for i in smallList[indexNum+2:] if CleanDefs.hasNumbers(i))
        procedureIndexEnd = smallList.index(x)
        countIndex = indexNum+2
        proString = ""
        while countIndex < procedureIndexEnd:
            proString += smallList[countIndex] + " "
            countIndex +=1
           
        valuesDict['Procedure1'] += [proString]
        proString = CleanDefs.modifyProcedures(proString.rstrip(" "))
        valuesDict['Procedure2'] += [proString]
        proString = CleanDefs.modifyProcedures2(proString)
        valuesDict['Procedure3'] += [proString]
        proString =CleanDefs.modifyProcedures3(proString)
        valuesDict['Procedure4'] += [proString]
        
        """***Get Procedure Times***"""
        valuesDict = CleanDefs.get_procedure_times(valuesDict,smallList, countIndex)
        
        """***Get Codes and History Strings***"""
        valuesDict = CleanDefs.getCodesandSurgeries(valuesDict, smallList,InsurIndexNum)
        
        
    return valuesDict
    


"""*****Run main Processess*****"""  
largeList = create_large_List(fileName)
orgValuesDict = CleanDefs.org_values_dict()
valuesDict = create_values_dict(largeList, orgValuesDict)

"""*****Print length of each row in valuesDict*****"""  
for i in valuesDict:
     print i + " : " + str(len(valuesDict[i]))



"""*****Write Procedures to text file to check modifications*****"""  
df3 = pandas.DataFrame.from_dict(valuesDict)
df3 = df3[['Procedure1','Procedure2','Procedure3', 'ID']]
df3.drop_duplicates('Procedure1').to_csv('C:\Users\James\Documents\GitHub\sfdat28\JVK-HOS Project\Procedures3.csv')
text_file = open("C:\Users\James\Documents\Output.txt", "w")
#for i in zip(valuesDict['Procedure1'],valuesDict['Procedure2'],valuesDict['Procedure3'],valuesDict['ID']):
    
    #text_file.write(str(i).replace("(","").replace(")","") +'\n')    




"""*****Final dataframe formatting******"""
df = pandas.DataFrame.from_dict(valuesDict)
dfNotNull = df[df['ProStartTime'].notnull()]
dfNotNull['Age'] = dfNotNull['Age'].astype("int")
dfNotNull['HealthCodesLen'] = dfNotNull['HealthCodesLen'].astype("int")
dfNotNull['EnterTime'] = pandas.to_datetime(dfNotNull['EnterTime'], format='%H%M')
dfNotNull['ProStartTime'] = pandas.to_datetime(dfNotNull['ProStartTime'], format='%H%M')        
dfNotNull['ProEndTime'] = pandas.to_datetime(dfNotNull['ProEndTime'], format='%H%M')         
dfNotNull['ExitTime'] = pandas.to_datetime(dfNotNull['ExitTime'], format="%H%M")  
dfNotNull['TotalTime']= dfNotNull['ExitTime']- dfNotNull['EnterTime']
dfNotNull['TotalTimeMin']= dfNotNull['TotalTime'].astype('timedelta64[m]')
dfNotNull['BMI'] = dfNotNull['BMI'].apply(lambda x: float(x.replace("\n","")))
dfNotNull['Gender'] = dfNotNull['Gender'].apply(lambda x: CleanDefs.genderSwitch(x))
dfNotNull['Day'] = pandas.to_datetime(dfNotNull['Date']).dt.dayofweek

dfNotNull['TotalLen'] = 0



for i in filter(lambda x: "History" in x, valuesDict.keys()):
    dfNotNull[i+"Len"] = dfNotNull[i].apply(lambda x: len(x))
for i in filter(lambda x: "History" in x, valuesDict.keys()):
    dfNotNull['TotalLen'] += dfNotNull[i+"Len"]
    
    
dumms = pandas.get_dummies(dfNotNull.Procedure3)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)

dumms = pandas.get_dummies(dfNotNull.Procedure4)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)

dumms=pandas.get_dummies(dfNotNull.Day)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)

dumms=pandas.get_dummies(dfNotNull.InsuranceName)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)


        
print dfNotNull['Procedure3'].value_counts()


"""Create 10 minute classifications"""

dfNotNull['MinClass'] =  dfNotNull['TotalTimeMin'].apply(lambda x: CleanDefs.minClass(x))

print"""
Run KNN Classification ****
"""
#shortDF = dfNotNull[(dfNotNull['MinClass'] == "29 Max Class") | (dfNotNull['MinClass'] == "39 Max Class") | (dfNotNull['MinClass'] == "49 Max Class") | (dfNotNull['MinClass'] == "19 Max Class") |(dfNotNull['MinClass'] == "59 Max Class" )|(dfNotNull['MinClass'] == "69 Max Class")]
#feature_cols = [3,'Lumbar Radiofrequency','Nerve Block','Cervical/Thoracic Radiofrequency', 'Facet Joint Injection','Cervical/Thoracic ESI', 'Cervical/Thoracic Medial Branch Block','Spinal Cord Stimulator']
feature_cols = ['Lumbar Radiofrequency', 'Facet Joint Injection', 'Cervical/Thoracic ESI', 'Spinal Cord Stimulator']
X = dfNotNull[feature_cols]
y = dfNotNull['MinClass']
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
for i in range (1,25):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X, y, cv=3, scoring='accuracy')
    print i
    print scores.mean()



print"""
Run Logistic Reggression Classification Cross Val ****
"""    
    
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
logreg = LogisticRegression()
print cross_val_score(logreg, X, y, cv=5, scoring='accuracy').mean()


print"""
Run Logistic Reggression Classification Predictiona ****
""" 

from sklearn.cross_validation import train_test_split
  
features_train, features_test, response_train, response_test \
= train_test_split(X, y, random_state=4)
# instantiate and fit
logreg = LogisticRegression()
logreg.fit(features_train, response_train)

predictionDF = pandas.DataFrame()
predictionDF['Predictions'] = logreg.predict(features_test)
predictionDF['Actual'] = response_test.values
predictionDF =pandas.concat([predictionDF,features_train],axis = 1)

print predictionDF.head(15)




print"""
Run Decison-Tree Classification ****
"""    

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
for i in range (0,5):
    clf = DecisionTreeClassifier(random_state=i)
    scores = cross_val_score(clf, X, y, cv=10)
    print scores.mean()
    

    
    
    
print """
Run Linear Regression Train-Test Split ****
"""
X = dfNotNull[feature_cols]
y = dfNotNull['TotalTimeMin']


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

features_train, features_test, response_train, response_test \
= train_test_split(X, y, random_state=4)
# instantiate and fit


linreg = LinearRegression()
linreg.fit(features_train, response_train)

predictionDF = pandas.DataFrame()
predictionDF['Predictions'] = linreg.predict(features_test)
predictionDF['Actual'] = response_test.values
f = lambda x:int(round(x/15.)*15)
print predictionDF.head(15)
print zip(feature_cols,linreg.coef_) 


    
print """
Run Linear Regression Cross Val ****
"""

from sklearn.neighbors import KNeighborsRegressor
def cross_val_r2(X, y,n):
    linreg = LinearRegression()
    neigh = KNeighborsRegressor(n_neighbors=n)
    scores = cross_val_score(neigh, X, y, cv=3, scoring='r2')
    mean2escores = cross_val_score(neigh, X, y, cv=3, scoring='mean_squared_error')
    meanAbsScores= cross_val_score(neigh, X, y, cv=3, scoring='mean_absolute_error')
    return (scores,mean2escores,meanAbsScores) # return average R2
cvScores= cross_val_r2(X, y,23)
r2Scores = cvScores[0]
mean2escores = cvScores[1]
meanAbsScores= cvScores[2]
print "root mean squared error"
print np.sqrt(abs(mean2escores)).mean()
print "r2 Scores"
print r2Scores
print r2Scores.mean()
print "mean Abs Scores"
print meanAbsScores
print meanAbsScores.mean()


print """
Run Linear Regression Cross Val All Feature Combos **** temporarily changing to KNN!!
"""

import itertools
r2 = (.5, 4)
feature_cols = [ 3,'Gender','Lumbar Radiofrequency','Nerve Block','Cervical/Thoracic Radiofrequency', 'Facet Joint Injection','Cervical/Thoracic ESI', 'Cervical/Thoracic Medial Branch Block','Spinal Cord Stimulator']

def iter_r2(r2,dfNotNull,feature_cols):
    for L in range(2, len(feature_cols)+1):
      print L
      for subset in itertools.combinations(feature_cols, L):
        Subset = list(subset)
        X = dfNotNull[Subset]
        for i in range(15,25):
            rTest = cross_val_r2(X, y, i)[0].mean()
            if rTest.mean() > r2[0]:
                r2 = (rTest, Subset, i)
    return r2
#print iter_r2(r2,dfNotNull,feature_cols)  
print """
Run Linear Regression on Scaled Features with Pipeline ****
"""
"""
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

feature_cols = ['BMI', 'Age','SurgicalHistoryLen','Gender']
X = dfNotNull[feature_cols]

pipe = make_pipeline(StandardScaler(), LinearRegression())

#grid = GridSearchCV(pipe, cv=5, scoring='r2')
#do a feature union with the non-scalable data around here!!!!!-jvk


scores = cross_val_score(pipe, X, y, cv=3, scoring='r2')
print scores.mean()
"""

print """
Run KNN Regression Cross Val All Feature Combos ****
"""
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
"Scale values in the DF"   
from sklearn.base import TransformerMixin
from sklearn import preprocessing
import itertools
class FeatureScalarTransformer(TransformerMixin):
        def transform(self,X,  **transform_params):
            if 'BMI' in X:
                X['BMI'] = preprocessing.scale(X['BMI'])
            if 'Age' in X:
                X['Age']= preprocessing.scale(X['Age'].astype('float'))
            if 'SurgicalHistoryLen' in X:
                X['SurgicalHistoryLen'] = preprocessing.scale(X['SurgicalHistoryLen'].astype('float'))
            
            
            return X
           

        def fit(self, X, y=None, **fit_params):
            return self

"Use KNN to find scores"
def cross_val_r2_knn(X, y, neighs):
    neigh = KNeighborsRegressor(n_neighbors=neighs)
    
    h = FeatureScalarTransformer()
    X = h.fit_transform(X)
    pipe = Pipeline([('Tranform_Features', FeatureScalarTransformer()), ('KNNeighbors', neigh)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    
    return scores
    


r2 = (.5, 4)
pipe = make_pipeline(StandardScaler(), LinearRegression())


feature_cols = [ 'BMI', 'Age','SurgicalHistoryLen',3,'Gender','Lumbar Radiofrequency','Nerve Block','Cervical/Thoracic Radiofrequency', 'Facet Joint Injection','Cervical/Thoracic ESI', 'Cervical/Thoracic Medial Branch Block','Spinal Cord Stimulator']
feature_cols = [ 'Injection',3,'Gender','Lumbar Radiofrequency','Nerve Block','Nerve Block 2','Cervical/Thoracic Radiofrequency', 'Facet Joint Injection','Cervical/Thoracic ESI', 'Cervical/Thoracic Medial Branch Block','Spinal Cord Stimulator']

def iter_r2_knn(r2,dfNotNull,feature_cols):
    for L in range(2, len(feature_cols)+1):
      print L
      for subset in itertools.combinations(feature_cols, L):
        Subset = list(subset)
        X = dfNotNull[Subset]
        for i in range(10,25):
            rTest = cross_val_r2_knn(X, y, i).mean()
            if rTest > r2[0]:
                print rTest
                r2 = (rTest, Subset, i)
    return r2
       
r2score= iter_r2_knn(r2,dfNotNull,feature_cols) 
print r2score









"""*****Sample plots and analysis*****"""
plt.scatter(dfNotNull['SurgicalHistoryLen'], dfNotNull['TotalTimeMin'])
dfNotNull[['Age','TotalTimeMin']].corr()
plt.show()
dfNotNull[dfNotNull['SurgicalHistoryLen'] <5]['TotalTimeMin'].mean()



dfNotNull.to_csv("C:\Users\James\Documents\GitHub\sfdat28\JVK-HOS Project\Long.csv", ",")