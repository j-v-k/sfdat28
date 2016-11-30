from datetime import datetime, date
import CleanDefs
import matplotlib.pyplot as plt
import pandas
import numpy as np



"""****Split Report into lists by user****"""
fileName = "..\JVK-HOS Project\VK Report.txt"
csvFileName ="C:\Users\James\Documents\GitHub\sfdat28\JVK-HOS Project\Long.csv"
#csvFileName ="C:\Users\James\Documents\GitHub\sfdat28\JVK-HOS Project\OutOS.csv"
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




#for i in zip(valuesDict['Procedure1'],valuesDict['Procedure2'],valuesDict['Procedure3'],valuesDict['ID']):
    
    #text_file.write(str(i).replace("(","").replace(")","") +'\n')    




"""*****Final dataframe formatting******"""
df = pandas.DataFrame.from_dict(valuesDict)
dfNotNull = df[df['ProStartTime'].notnull()]
dfNotNull['Age'] = dfNotNull['Age'].astype("int")
dfNotNull['HealthCodesLen'] = dfNotNull['HealthCodesLen'].astype("int")
dfNotNull['SurgicalHistoryLen'] = dfNotNull['SurgicalHistory'].apply(lambda x: len(x))
dfNotNull['EnterTime'] = pandas.to_datetime(dfNotNull['EnterTime'], format='%H%M')
dfNotNull['ProStartTime'] = pandas.to_datetime(dfNotNull['ProStartTime'], format='%H%M')        
dfNotNull['ProEndTime'] = pandas.to_datetime(dfNotNull['ProEndTime'], format='%H%M')         
dfNotNull['ExitTime'] = pandas.to_datetime(dfNotNull['ExitTime'], format="%H%M")  
dfNotNull['TotalTime']= dfNotNull['ExitTime']- dfNotNull['EnterTime']
dfNotNull['TotalTimeMin']= dfNotNull['TotalTime'].astype('timedelta64[m]')
dfNotNull['BMI'] = dfNotNull['BMI'].apply(lambda x: float(x.replace("\n","")))
dfNotNull['Gender'] = dfNotNull['Gender'].apply(lambda x: CleanDefs.genderSwitch(x))
dfNotNull['Day'] = pandas.to_datetime(dfNotNull['Date']).dt.dayofweek
dfNotNull['Month'] = pandas.to_datetime(dfNotNull['Date']).dt.month
dfNotNull['1r'] = pandas.to_datetime(dfNotNull['ProStartTime'].dt.time.astype('str') + " " + dfNotNull['Date'])
dfNotNull['1r'] = dfNotNull['1r'].apply(lambda x: CleanDefs.create1r(x))

dfNotNull['AverageProcTime']=dfNotNull['Procedure3'].apply(lambda x: CleanDefs.createProcAvgs(x,dfNotNull) )
dfNotNull['TotalLen'] = 0


for i in filter(lambda x: "History" in x, valuesDict.keys()):
    
    dfNotNull[i+"Len"] = dfNotNull[i].apply(lambda x: len(x))
for i in filter(lambda x: "History" in x, valuesDict.keys()):
    dfNotNull['TotalLen'] += dfNotNull[i+"Len"]

# Health Codes1   
HealthCodesList = CleanDefs.getHealthCodes(dfNotNull)
for i in HealthCodesList:
    dfNotNull[i] = dfNotNull['HealthCodes'].apply(lambda x: CleanDefs.CodeMatch(x,i))
    HealthCodesList = CleanDefs.getHealthCodes(dfNotNull)

#Health Codes 2
tt=[]
for i in HealthCodesList:
    tt += [i[0]]
    dfNotNull[i[0]] = dfNotNull['HealthCodes'].apply(lambda x: CleanDefs.CodeMatch(x,i[0]))

tt=[]
CPTCodeList = CleanDefs.getCPTCodes(dfNotNull)
for i in CPTCodeList:
    print i
    tt += [i]
    dfNotNull[i] = dfNotNull['CPTCode'].apply(lambda x: CleanDefs.CodeMatch(x,i))

   
""""Dummy Creation"""    
dumms = pandas.get_dummies(dfNotNull.Procedure3)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)

dumms = pandas.get_dummies(dfNotNull.Procedure4)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)

dumms=pandas.get_dummies(dfNotNull.Day)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)

dumms=pandas.get_dummies(dfNotNull.InsuranceName)
dfNotNull = pandas.concat([dfNotNull,dumms],axis = 1)




"""*****Scaling******"""

from sklearn.preprocessing import StandardScaler
scal = StandardScaler().fit(dfNotNull['BMI'])
dfNotNull['BMIScaled'] = scal.transform(dfNotNull['BMI'])
scal = StandardScaler().fit(dfNotNull['Age'])
dfNotNull['AgeScaled'] = scal.transform(dfNotNull['Age'])

scal = StandardScaler().fit(dfNotNull['Age'])
dfNotNull['HealthCodesLenScaled'] = scal.transform(dfNotNull['Age'])

scal = StandardScaler().fit(dfNotNull['SurgicalHistoryLen'])
dfNotNull['SurgicalHistoryLenScaled'] = scal.transform(dfNotNull['SurgicalHistoryLen'])

        
print dfNotNull['Procedure3'].value_counts()


"""Create 10 minute classifications"""

dfNotNull['MinClass'] =  dfNotNull['TotalTimeMin'].apply(lambda x: CleanDefs.minClass(x))

dfNotNull.to_csv(csvFileName, ",")















"""*****Sample plots and analysis*****"""
plt.scatter(dfNotNull['SurgicalHistoryLen'], dfNotNull['TotalTimeMin'])
dfNotNull[['Age','TotalTimeMin']].corr()
plt.show()
dfNotNull[dfNotNull['SurgicalHistoryLen'] <5]['TotalTimeMin'].mean()


