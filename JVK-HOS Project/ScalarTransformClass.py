# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 17:48:54 2016

@author: James
"""
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
from sklearn.base import TransformerMixin
from sklearn import preprocessing

class FeatureScalarTransformer(TransformerMixin):
        def transform(self,X,  **transform_params):
            X['BMT'] = preprocessing.scale(X['BMI'])
            X['Age2']= preprocessing.scale(X['Age'].astype('float'))
            return X
           

        def fit(self, X, y=None, **fit_params):
            return self



        
feature_cols = [ 'BMI', 'Age','SurgicalHistoryLen',3,'Gender','Lumbar Radiofrequency','Nerve Block','Cervical/Thoracic Radiofrequency', 'Facet Joint Injection','Cervical/Thoracic ESI', 'Cervical/Thoracic Medial Branch Block','Spinal Cord Stimulator']

X = dfNotNull[feature_cols]
h = FeatureScalarTransformer()
transformed = h.fit_transform(X)
