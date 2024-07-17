import csv
import operator
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter

cities = {}
sol = []

with open('2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader(tsp)
    count = 0
    for row in reader:
        cities[count] = [row[0], row[1]]
        count += 1

class City:
    def __init__(self, citynum, x, y):
        self.citynum = citynum
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(float(self.x) - float(city.x))
        yDis = abs(float(self.y) - float(city.y))
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return str(self.citynum)

class Calculation :
     @staticmethod
     def totalDistance(population):
        pathDistance = 0
        distance = 0
        for i in range(0, len(population)):
            fromCity = population[i]
            toCity = None
            if i + 1 < len(population):
                toCity = population[i + 1]
            else:
                toCity = population[0]
            pathDistance += fromCity.distance(toCity)
        distance = pathDistance
        return distance
    
class GA :
    @staticmethod
    def createRandomPath(cityList):
        path = random.sample(cityList[1:], len(cityList) - 1)
        path.insert(0, cityList[0])
        return path

def randomsearch(generation):
    random_list = []
    loopCount=0
    random_sample = GA.createRandomPath(cityList)
    current_path = Calculation.totalDistance(random_sample)
    shortest_path = current_path
    
    while(loopCount<generation):
        random_sample1 = GA.createRandomPath(cityList)
        new_path = Calculation.totalDistance(random_sample1)
        print("random_search_path :" +str(new_path))
        if new_path<shortest_path:
            shortest_path = new_path
        loopCount = loopCount+1
        random_list.append(new_path)
    print("best path : ",shortest_path)
    plt.plot(random_list)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title("RandomSearch, Generation = 500")
    plt.show()
    return shortest_path

if __name__ == '__main__':
    cityList = []

    for i in cities :
        location = i
        x = cities[i][0]
        y = cities[i][1]
        cityList.append(City(location, x=x, y=y))

    randomsearch(500)