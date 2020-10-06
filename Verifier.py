# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:30:02 2020

@author: Frankie Austin
"""


def readFiles(inputFileName, outputFileName): 
        iFile = open(inputFileName, "r")
        oFile = open(outputFileName, "r")
        
        iDictionaryValues = iFile.readlines()
        numPoints = iDictionaryValues[0]  #I initially assigned the weight and the number of unique items to a value of 0. This way, I can pop them out and store them in the following variables. 
        numSets = iDictionaryValues[1] #Weight capacity
        iDictionaryValues.pop(0)
        iDictionaryValues.pop(0)
        
        iDictionaryKeys = list(range(1, len(iDictionaryValues)+1))
        
        iDictionaryOfPoints = {} 
        
        
        
        
        
        
        
        
        
        return True






def ourAlgorithm():
    
    pass


def main(): 
    testInput = "exampleinput.txt"
    testOutput = "exampleoutput.txt"
    
    verifier = readFiles(testInput, testOutput)
    print(verifier)

if __name__ == "__main__":   #Calls main
    main()
    