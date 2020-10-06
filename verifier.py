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
        
        iDictionaryOfPoints = dict(zip(iDictionaryKeys, iDictionaryValues))

        #read in the list of output sets
        oListSets = oFile.readlines()
        
        #pop the maxlength
        maxDistance = oListSets[0]
        oListSets.pop(0)
        
        
        return iDictionaryOfPoints, oListSets, numPoints, numSets, maxDistance


def allTests(iDictionaryOfPoints, oListSets, numPoints, numSets, maxDistance):
        
        testsPass = True
        numSortedPoints = 0
        
        #compare input k with the number of sets
        if (numSets != len(oListSets)):
            print("Incorrect number of sets")
            testPass = False
            
        #ensure that the sets are not empty
        for sets in oListSets:
            if len(sets) == 0:
                print("There is a non-empty set")
                testPass = False
            numSortedPoints += len(sets)
        
        #compare the input n with number of points
        if numSets != numSortedPoints:
            print("There are an incorrect number of points sorted")
            testPass = False
            
        #verify max distance is correct
        for sets in oListSets:
            key = 0
            for point in sets:
                
    
        
        
        
        
        
        



def ourAlgorithm():
    
    pass


def main(): 
    testInput = "exampleinput.txt"
    testOutput = "exampleoutput.txt"
    
    fileSpecs = readFiles(testInput, testOutput)
    verifier = allTests(fileSpecs[0], fileSpecs[1], fileSpecs[2], fileSpecs[3], fileSpecs[4])
    print(verifier)

if __name__ == "__main__":   #Calls main
    main()
    