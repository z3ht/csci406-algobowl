# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:30:02 2020

@author: Frankie Austin
"""
import time

start_time = time.time()

def readFiles(inputFileName, outputFileName): 
        iFile = open(inputFileName, "r")
        oFile = open(outputFileName, "r")
        
        iDictionaryValues = iFile.readlines()
        numPoints = int(iDictionaryValues[0])  #I initially assigned the weight and the number of unique items to a value of 0. This way, I can pop them out and store them in the following variables. 
        numSets = int(iDictionaryValues[1]) #Weight capacity
        iDictionaryValues.pop(0)
        iDictionaryValues.pop(0)
        i_point_values = []
        temp_list = []
        array_of_point_values = []
        for value in iDictionaryValues:
            temp_list = value.split()
            for point in range(len(temp_list)):
                temp_list[point] = int(temp_list[point])
            i_point_values.append(temp_list)
            
        iDictionaryKeys = list(range(1, len(iDictionaryValues)+1))
        
        iDictionaryOfPoints = dict(zip(iDictionaryKeys, i_point_values))

        #read in the list of output sets
        o_sets = oFile.readlines()
        #pop the maxlength
        maxDistance = int(o_sets[0])
        o_sets.pop(0)        
        oListSets = []   
        for value in o_sets:
            temp_list = value.split()
            for point in range(len(temp_list)):
                temp_list[point] = int(temp_list[point])
            oListSets.append(temp_list)
        
        
        
        
        return iDictionaryOfPoints, oListSets, numPoints, numSets, maxDistance


def allTests(iDictionaryOfPoints, oListSets, numPoints, numSets, maxDistance):
        
        testsPass = True
        numSortedPoints = 0
        duplicateChecker = []
       
        
        #compare input k with the number of sets
        if (numSets != len(oListSets)):
            print("Incorrect number of sets")
            testsPass = False
            
        #ensure that the sets are not empty
        for sets in oListSets:
            if len(sets) == 0:
                print("There is a empty set")
                testsPass = False
            numSortedPoints += len(sets)
        
        #compare the input n with number of points
        if numPoints != numSortedPoints:
            print("There are an incorrect number of points sorted")
            testsPass = False
            
        if (len(iDictionaryOfPoints.keys()) != numSortedPoints): 
             testsPass = False
            
        #verify max distance is correct
        calculated_max_distance = 0
        max_distance_set = 0
   
        try: 
            for sets in oListSets:
                for a in sets:
                    point1 = iDictionaryOfPoints[int(a)]
                    duplicateChecker.append(a) 
                    for b in sets:
                        point2 = iDictionaryOfPoints[int(b)]
                        duplicateChecker.append(b)
                        x_distance = point1[0] - point2[0]
                        y_distance = point1[1] - point2[1]
                        z_distance = point1[2] - point2[2]
                        total_distance = abs(x_distance) + abs(y_distance) + abs(z_distance)
                        if calculated_max_distance < total_distance:
                            calculated_max_distance = total_distance
                            max_distance_set = sets
        except: 
            print("There was a point in the output file that was out of bounds.")
            return False
        
        for n in range(1, numPoints): 
            if duplicateChecker.count(n):
                continue
            else: 
                print("There was a duplciate located in the output sets. Invalid output")
                return False
            
        
        
        #if (len(oListSets) != len(duplicateChecker)): 
          #  print("There was a duplciate located in the output sets. Invalid output")
           # return False
                
        if maxDistance != calculated_max_distance:
            print("The max distance was incorrect for the following set")
            print(max_distance_set)
            testsPass = False
                
        return testsPass


def ourAlgorithm():
    
    pass


def main(): 
    testInput = "TRUERandinput.txt"
    testOutput = "TRUERandoutput.txt"
    
    fileSpecs = readFiles(testInput, testOutput)
    verifier = allTests(fileSpecs[0], fileSpecs[1], fileSpecs[2], fileSpecs[3], fileSpecs[4])
    print("The time it took to run this program is: ")
    print(format((time.time() - start_time), '.014f'))
    print(verifier)

if __name__ == "__main__":   #Calls main
    main()
    