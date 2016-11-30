# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:51:45 2016

@author: James
"""
from datetime import datetime, date
import numpy as np
import re

def InsuranceSort(x):
    if "MEDICARE" in x:
        return "MEDICARE"
    elif "BLUE CROSS" in x:
        return "BLUE CROSS"
    elif "BLUE SHIELD" in x:
        return "BLUE SHIELD"
    else:
        return "other"

def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)
def hasLetters(inputString):
     return any(char.isalpha() for char in inputString)
     
def get_personal_info(indexNum, valuesDict, smallList):
    
    valuesDict['ID'] += [smallList[indexNum-2]]
    valuesDict['Age'] += [smallList[indexNum-1]]
    valuesDict['Gender'] += [smallList[indexNum]]
    valuesDict['BMI'] += [smallList[indexNum+1].replace("\n", "")]
    valuesDict['Date'] += [smallList[indexNum-3].replace("'","")]
    return valuesDict
     
def get_procedure_times(valuesDict,smallList, countIndex):
       timeList =[]
       while hasNumbers(smallList[countIndex]):
           timeList += [smallList[countIndex]]
           countIndex += 1
       
       
       if len(timeList) == 2:
           valuesDict['EnterTime'] += [np.nan]
           valuesDict['ProStartTime'] += [np.nan]
           valuesDict['ProEndTime'] += [timeList[0]]
           valuesDict['ExitTime'] += [timeList[1].replace("',","")]
       elif len(timeList) ==5:
           valuesDict['EnterTime'] += [timeList[0]]
           valuesDict['ProStartTime'] += [timeList[1]]
           valuesDict['ProEndTime'] += [timeList[2]]
           valuesDict['ExitTime'] += [timeList[4].replace("',","")]
       elif len(timeList) ==1:
           valuesDict['EnterTime'] += [np.nan]
           valuesDict['ProStartTime'] += [np.nan]
           valuesDict['ProEndTime'] += [np.nan]
           valuesDict['ExitTime'] += [np.nan]
                
       else:
           valuesDict['EnterTime'] += [np.nan]
           valuesDict['ProStartTime'] += [np.nan]
           valuesDict['ProEndTime'] += [np.nan]
           valuesDict['ExitTime'] += [np.nan]
                #print smallList[indexNum-2]+ "ssss"
       return valuesDict
       
       
       
def getHealthCodes(dfNotNull):
    healthcodesBIGlist = []
    for i in dfNotNull['HealthCodes']:
        healthcodesBIGlist += i
    smallList =list(set(healthcodesBIGlist))
    iterList = []
    for i in smallList:
        iterList += [i[0:i.find('.')]]
    smallList = list(set(iterList))
    return smallList
    
def getCPTCodes(dfNotNull):
    CPTCodeBIGlist = []
    for i in dfNotNull['CPTCode']:
        CPTCodeBIGlist += i
    smallList =list(set(CPTCodeBIGlist))
    iterList = []
    for i in smallList:
        iterList += [i]
    smallList = list(set(iterList))
    return smallList
        
    
    



def modifyProcedures(procedure):
    
    procedure = procedure.replace("CERIVICAL","CERVICAL")
    procedure = procedure.replace("LUMBER","LUMBAR")
    procedure = procedure.replace("MEDIAN","MEDIAL")
    procedure = procedure.replace("LUMABAR","LUMBAR")
    procedure = procedure.replace("BRANH","BRANCH")
    procedure = procedure.replace("SACRO ILIAC","SACROILIAC")
    procedure = procedure.replace("CERVICAL FACET JOINT STEROID INJECTIO..","CERVICAL FACET JOINT STEROID INJECTION..") 
    procedure = procedure.replace("CERVICAL SPINAL CORD STIMULATOR..","CERVICAL SPINAL CORD STIMULATOR") 
    procedure = procedure.replace("LUMBAR SPINAL CORD STIMULATOR PERMANE..","LUMBAR SPINAL CORD STIMULATOR PERMANEN..")     
    procedure = procedure.replace("THORACIC MEDIAL BRANCH RADIOFREQUENCY,..", "THORACIC MEDIAL BRANCH RADIOFREQUENCY..")         
    procedure = procedure.replace("CERVICAL MEDIAL BRANCH BLOCK, LEVELS T..", "CERVICAL MEDIAL BRANCH BLOCK LEVELS TO..")    
    procedure = procedure.replace("CERVICAL MEDIAL BRANCH RADIOFREQUENCY,..", "CERVICAL MEDIAL BRANCH RADIOFREQUENCY..")     
    procedure = procedure.replace("LUMBAR FACET JOINT STEROID INJECTIONS..", "LUMBAR FACET JOINT STEROID INJECTION..")    
    procedure = procedure.replace("LUMBAR FACET JOINT STEROID INJECTION,..", "LUMBAR FACET JOINT STEROID INJECTION..")    
    procedure = procedure.replace("LUMBAR FACET JOINT STEROID INJECTION:..", "LUMBAR FACET JOINT STEROID INJECTION..")     
    procedure = procedure.replace("LUMBAR SPINAL CORD STIMULATOR TRIAL.." , "LUMBAR SPINAL CORD STIMULATOR TRIAL")    
    procedure = procedure.replace("RIG.." , "RIGHT")
    procedure = procedure.replace("RIGHTH" , "RIGHT")
    procedure = procedure.replace("RIGH..", "RIGHT..")
    if "LUMBAR MEDIAL BRANCH BLOCK" in procedure:
        procedure = "LUMBAR MEDIAL BRANCH BLOCK"
        
    elif "CERVICAL MEDIAL BRANCH BLOCK" in procedure:
        procedure = "CERVICAL MEDIAL BRANCH BLOCK"
    elif "CERVICAL MEDIAL MEDIAL" in procedure:
        procedure = "CERVICAL MEDIAL BRANCH BLOCK"
    elif "HIP INJECTION" in procedure or "HIP JOINT INJECTION" in procedure:
        procedure = "HIP JOINT INJECTION"
    elif "LUMBAR MEDIAL BRANCH RADIOFREQUENCY" in procedure or "LUMBAR MEDIAL RADIOFREQUENCY" in procedure or "LUMBAR RADIOFREQUENCY MEDIAL BRANCH" in procedure:
        procedure = "LUMBAR MEDIAL BRANCH RADIOFREQUENCY"
    elif "LUMBAR SELECTIVE NERVE ROOT BLOCK" in procedure:
        procedure = "LUMBAR SELECTIVE NERVE ROOT BLOCK"
    elif "THORACIC MEDIAL BRANCH BLOCK" in procedure:
        procedure = "THORACIC MEDIAL BRANCH BLOCK"
    elif "LUMBAR MEDIAL BRANCH RADIOFREQUENCY" in procedure:
        procedure = "LUMBAR MEDIAL BRANCH RADIOFREQUENCY"
    elif "CERVICAL FACET JOINT STEROID" in procedure:
        procedure = "CERVICAL FACET JOINT STEROID INJECTION"
    elif "SACROILIAC JOINT" in procedure:
        procedure = "SACROILIAC JOINT STEROID INJECTION"
    elif "CERVICAL SPINAL CORD STIMULATOR" in procedure:
        procedure = "CERVICAL SPINAL CORD STIMULATOR"
    elif "LUMBAR SPINAL CORD STIMULATOR" in procedure:
        procedure = "LUMBAR SPINAL CORD STIMULATOR"
    elif "MEDIAL BRANCH RADIOFREQUENCY" in procedure and "THORACIC" not in procedure:
        procedure = "MEDIAL BRANCH RADIOFREQUENCY"
    elif "RIGHT LUMBAR MEDIAL BRANCH.." in procedure:
        procedure = "LUMBAR MEDIAL BRANCH.."
    return procedure

def modifyProcedures2(procedure):
     
    a = ["LUMBAR MEDIAL BRANCH RADIOFREQUENCY", "MEDIAL BRANCH RADIOFREQUENCY LEVELS TO..", "MEDIAL BRANCH RADIOFREQUENCY LEFT.. ", "MEDIAL BRANCH RADIOFREQUENCY", "LUMBAR MEDIAL BRANCH LEFT" ]
    b = ["LUMBAR EPIDURAL STEROID INJECTION..", "LUMBAR EPIDURAL TRANSFORAMINAL STEROID.." ]
    c = ["LUMBAR MEDIAL BRANCH BLOCK","LUMBAR MEDIAL BRANCH..", "RIGHT LUMBAR MEDIAL BRANCH.." ]
    d = ["LUMBAR FACET JOINT STEROID INJECTION..", "THORACIC FACET JOINT STEROID INJECTION..", "CERVICAL FACET JOINT STEROID INJECTION"]
    e = ["CERVICAL MEDIAL BRANCH RADIOFREQUENCY.." , "THORACIC MEDIAL BRANCH RADIOFREQUENCY..", "CERVICAL BRANCH RADIOFREQUENCY.."]
    f = ["CERVICAL MEDIAL BRANCH BLOCK", "THORACIC MEDIAL BRANCH BLOCK", "CERVICAL/THORACIC MEDIAL BRANCH..", "CERVICAL MEDIAL MEDIAL BRANCH.."]    
    g = ["CERVICAL EPIDURAL STEROID INJECTION.." , "THORACIC EPIDURAL STEROID INJECTION.."]
    h = ["LUMBAR SPINAL CORD STIMULATOR" , "CERVICAL SPINAL CORD STIMULATOR"]
    i = ["LUMBAR SELECTIVE NERVE ROOT BLOCK", "LATERAL FEMORAL CUTANEOUS NERVE BLOCK..","CERVICAL SELECTIVE NERVE ROOT BLOCK.." ]
    j = ["SACROILIAC JOINT STEROID INJECTION" , "HIP JOINT INJECTION" , "SACROCOCCYGEAL JOINT INJECTION UNDER.."]
    k = ["Removal of Spinal Cord Stimulator"]
    
    
    procedureName = "Not Assigned"    
    if any(procedure in x for x in a):
        procedureName = "Lumbar Radiofrequency"
    
    elif any(procedure in x for x in b):
        procedureName = "Lumbar ESI"
    elif any(procedure in x for x in c):
        procedureName = "Lumbar Medial Branch Block"
    elif any(procedure in x for x in d):
        procedureName = "Facet Joint Injection"
    elif any(procedure in x for x in e):
        procedureName = "Cervical/Thoracic Radiofrequency"
    elif any(procedure in x for x in f):
        procedureName = "Cervical/Thoracic Medial Branch Block"
    elif any(procedure in x for x in g):
        procedureName = "Cervical/Thoracic ESI"
    elif any(procedure in x for x in h):
        procedureName = "Spinal Cord Stimulator"
    elif any(procedure in x for x in i):
        procedureName = "Nerve Block"
    elif any(procedure in x for x in j):
        procedureName = "Joint Injection, other"
    elif any(procedure in x for x in k):
        procedureName = "Implant Removal"
    else:
        print procedure
        pass
    return procedureName
    
    
def modifyProcedures3(procedure):
     
   
    
    if "ESI" in procedure:
        procedureName = "ESI"
    elif "Medial Branch Block" in procedure:
        procedureName = "Medial Branch Block"
    elif "Radiofrequency" in procedure:
        procedureName = "Radiofrequency"
    elif "Injection" in procedure:
        procedureName = "Injection"
    elif "Nerve Block" in procedure:
        procedureName = 'Nerve Block 2'
    elif "Stimulator" in procedure:
        procedureName = 'Spinal Cord Stimulator 2'
    else:
        procedureName = procedure
    return procedureName
    
    
def getSurgeValues(smallList2,SurgicalHistIndex, valuesDict):
         smallString = " ".join(smallList2[SurgicalHistIndex:])
         pattern = "\w+ History"
         HistoryList =  re.findall(pattern, smallString)
         
         #print HistoryList
         for num,i in enumerate(HistoryList):
             try:
                 nextIt =HistoryList[num +1]
             except:
                 nextIt = HistoryList[num]
                
             sIndex =smallString.find(i)
             eIndex =smallString.find(nextIt)
             tinyString= smallString[sIndex:eIndex]
             
             pattern = "' .+-.+ '"
             TempList =  re.findall(pattern, tinyString)
             try:
                 
                 JoinList = TempList[0].replace(" ', '', ", "").split(",")
                 
             except:
                 JoinList = TempList
                    
             valuesDict[i.replace(" ","")] += [JoinList]
         
         for i in filter(lambda x: "History" in x, valuesDict.keys()):
             if i in str(HistoryList).replace(" ",""):
                 pass
             else:
                
                valuesDict[i] += [[]]
                 
         return valuesDict
    
    
def getCodesandSurgeries(valuesDict, smallList,InsurIndexNum):
        ICDCodeList = []
        InsuranceName = ""
        CPTCode = []
        while 'Surgical' not in smallList[InsurIndexNum]:
            if hasNumbers(smallList[InsurIndexNum]) and hasLetters(smallList[InsurIndexNum]) and "ICD-10" not in str(smallList[InsurIndexNum]):
                ICDCodeList += [smallList[InsurIndexNum]]
            elif hasLetters(smallList[InsurIndexNum]) and "ICD-10" not in str(smallList[InsurIndexNum]) and "Code" not in str(smallList[InsurIndexNum]) and "Insurance" not in str(smallList[InsurIndexNum]) :
                InsuranceName += str(smallList[InsurIndexNum]) + " "
            elif hasNumbers(smallList[InsurIndexNum]) and "." not in smallList[InsurIndexNum] and "-" not in smallList[InsurIndexNum] and "+" not in smallList[InsurIndexNum]:
                print smallList[InsurIndexNum]
                CPTCode += [str(smallList[InsurIndexNum])]
                    
            InsurIndexNum += 1
        
        if CPTCode == []:
            CPTCode = ['noCPT']
        valuesDict['HealthCodes'] += [ICDCodeList]
        valuesDict['HealthCodesLen'] += [len(ICDCodeList)]
        valuesDict['InsuranceName'] += [InsuranceSort(InsuranceName.rstrip(" "))]
        valuesDict['CPTCode'] += [CPTCode]
        valuesDict['CPTCodeLen'] += [len(CPTCode)]
        
        smallList2 = smallList[InsurIndexNum:300]
        SurgicalHistIndex = smallList2.index("Surgical")
        valuesDict = getSurgeValues(smallList2,SurgicalHistIndex, valuesDict)
        return valuesDict
        
def remove(inputList):
     inputList =  filter(lambda a: a != '', inputList)
     inputList =  filter(lambda a: a != ' ', inputList) 
     return inputList

def minClass(x):
    for i in range (0,200,15):
        if int(x) >= i - 7.5 and int(x) < i +7.5 :
            return str(i) + " Max Class"
def genderSwitch(x):
    if "F" in str(x):
        return 0
    elif "M" in str(x):
        return 1
    else:
        return "LOOOK HERE!!!"

def CodeMatch(x, i):
    
    if any(i in j for j in x):
        return 1
    else:
        return 0
    
def create1r(date2):
    
    if date2.weekday() == 3 and date2 < datetime(date2.year, date2.month, date2.day, 14,30,0):
        return 0
    else:
        return 1
        
def createProcAvgs(x, dfNotNull):
    
    return dfNotNull[dfNotNull['Procedure4'] == x]['TotalTimeMin'].mean()
    

def org_values_dict():
    valuesDict = {}
    valuesDict['ID'] = []
    valuesDict['Age'] = []
    valuesDict['Gender'] =[]
    valuesDict['BMI'] = []
    valuesDict['Date'] = []
    valuesDict['EnterTime'] = []
    valuesDict['ExitTime'] = []
    valuesDict['ProStartTime'] = []
    valuesDict['ProEndTime'] = []
    #valuesDict['Procedure'] = []
    valuesDict['InsuranceName'] = []
    valuesDict['HealthCodes'] = []
    valuesDict['HealthCodesLen'] = []
    valuesDict['SurgicalHistory'] = []
    valuesDict['MusculoskeletalHistory'] = []
    valuesDict['PertinentHistory'] = []
    valuesDict['RespiratoryHistory'] = []
    valuesDict['CardiovascularHistory'] = []
    valuesDict['GastrointestinalHistory'] = []
    valuesDict['GenitourinaryHistory'] = []
    valuesDict['EndocrineHistory'] = []
    valuesDict['HematologicHistory'] = []
    valuesDict['ImmunologicHistory'] = []
    valuesDict['SkinHistory'] = []
    valuesDict['UseHistory'] = []
    valuesDict['Procedure1']=[]
    valuesDict['Procedure2']=[]
    valuesDict['Procedure3']=[]
    valuesDict['Procedure4']=[]
    valuesDict['CPTCode'] = []
    valuesDict['CPTCodeLen'] =[]
    return valuesDict
    
 

