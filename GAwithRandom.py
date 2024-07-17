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

class Selection:
    def tournament_standard(population, populationrank):
        k = int(len(populationrank) * 0.2)
        tsample=random.sample(populationrank, k)
        top_fitness = 0
        top_ind = 0
        low_fitness = math.inf
        low_ind = 0
        t = random.random()
        if t < 0.7 :
            for i in range(0, len(tsample)):
                if top_fitness < tsample[i][1]:
                    top_fitness = tsample[i][1]
                    top_ind = tsample[i][0]
            return population[top_ind]        
        elif t > 0.7 :
            for i in range(0, len(tsample)):
                if low_fitness > tsample[i][1]:
                    low_fitness = tsample[i][1]
                    low_ind = tsample[i][0]
            return population[low_ind]

    def roullete_wheel(population):
        max = sum(Calculation.pathFitness(chromosome) for chromosome in population)
        pick = random.uniform(0,max)
        current = 0
        for chromosome in population:
            current += Calculation.pathFitness(chromosome)
            if current > pick:
                return chromosome 
            
class Crossover:
    @staticmethod
    def pmx(parent1,parent2):
        child1 = []
        child2 = []
        child3 = []
        while True :
            crossover_points = sorted([random.randint(1, len(parent1)-1) for _ in range(2)])
            if crossover_points[0] != crossover_points[1] :
                break
        startPoint = crossover_points[0]
        endPoint = crossover_points[1]
        for i in range(startPoint,endPoint):
            child1.append(parent1[i])
        for i in range(0,startPoint):
            child2.append(parent2[i])
        for i in range(endPoint,len(parent1)):
            child3.append(parent2[i])
        for i in range(0,len(child2)):
            if child2[i] in child1:
                while True:
                    ind = parent1.index(child2[i])
                    child2[i] = parent2[ind]
                    if child2[i] not in child1:
                        break

        for i in range(0,len(child3)):
            if child3[i] in child1:
                while True:
                    ind = parent1.index(child3[i])
                    child3[i] = parent2[ind]
                    if child3[i] not in child1:
                        break

        Return_chlid = child2+child1+child3
        return Return_chlid


    @staticmethod
    def order(parent1, parent2): 
        Return_list = []
        crossover_points = sorted([random.randint(1, len(parent1)-1) for _ in range(2)])
        startPoint = crossover_points[0]
        endPoint = crossover_points[1]
        alreadyIn = set()
        for i in range(startPoint, endPoint) :
            alreadyIn.add(parent1[i])
        count = 0
        for i in range(0, len(parent2)) :
            if len(Return_list) == startPoint :
                break
            if parent2[i] not in alreadyIn :
                Return_list.append(parent2[i])
                alreadyIn.add(parent2[i])
                count += 1
        for i in range(startPoint, endPoint) :
            Return_list.append(parent1[i])
        for i in range(count, len(parent2)) :
            if parent2[i] not in alreadyIn :
                Return_list.append(parent2[i])
                alreadyIn.add(parent2[i])
        return Return_list

class Mutation:
    @staticmethod
    def swap(offspring, mutationRate):
        if(random.random() < mutationRate) :
            ranchoice = np.random.choice([i for i in range(1, len(offspring))], 2, replace=False)
            index1, index2 = ranchoice
            offspring[index1], offspring[index2] = offspring[index2], offspring[index1]
            return offspring
        else :
            return offspring
    @staticmethod
    def inversion(offspring, mutationRate):
        if(random.random() < mutationRate) :
            ranchoice = np.random.choice([i for i in range(1, len(offspring))], 2, replace=False)
            index1, index2 = sorted(ranchoice)

            end = [] if index2==len(offspring)-1 else offspring[index2+1:]
            return offspring[:index1] + list(reversed(offspring[index1:index2+1])) + end
        else :
            return offspring
        
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
    
    @staticmethod
    def pathFitness(population):
        return 1 / Calculation.totalDistance(population)
    
    @staticmethod
    def fitnessRank(population):
        Rank = {}
        for i in range(0, len(population)):
            Rank[i] = Calculation.pathFitness(population[i])
        return sorted(Rank.items(), key = operator.itemgetter(1), reverse = True)
    
class GA :
    @staticmethod
    def createRandomPath(cityList):
        path = random.sample(cityList[1:], len(cityList) - 1)
        path.insert(0, cityList[0])
        return path
    
    @staticmethod
    def initRandomPopulation(popSize, citylist):
        population = []
        for i in range(0, popSize):
            population.append(GA.createRandomPath(citylist)) 

        print('population length: ',len(population))
        return population

    @staticmethod
    def nextPopulation(population, popRanked, mutationRate):
        children = []

        count = 0
        while True:
            parent1 = Selection.tournament_standard(population, popRanked)
            parent2 = Selection.tournament_standard(population, popRanked)
            child_cx = Crossover.pmx(parent1, parent2)
            child = Mutation.swap(child_cx, mutationRate)
            children.append(child)
            count += 1   

            if count == len(population):
                break

        return children 

    @staticmethod
    def nextGeneration(preGene, mutationRate):
        popRanked = Calculation.fitnessRank(preGene)
        children = GA.nextPopulation(preGene, popRanked, mutationRate)
        return children

    @staticmethod
    def makeTSPGraph(population ,popSize, mutationRate, generations):
        pop= GA.initRandomPopulation(popSize, population)
        bestInGene = []
        bestInGene.append(1 / Calculation.fitnessRank(pop)[0][1])
        print(bestInGene)

        for i in range(0, generations):
            pop = GA.nextGeneration(pop, mutationRate)
            bestInGene.append(1 / Calculation.fitnessRank(pop)[0][1])
            print('Generation : ', i+1)
            print('Best in Generation : ', 1 / Calculation.fitnessRank(pop)[0][1])
        print("Final Generation best : " + str(1 / Calculation.fitnessRank(pop)[0][1]))
        bestRouteIndex = Calculation.fitnessRank(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        plt.plot(bestInGene)
        plt.ylabel('Best in Generation')
        plt.xlabel('Generation')
        plt.title("Tournament_Standard, Order, Inversion(Population = 200, MutationRate = 0.05, Generation = 200)")
        plt.show()
        return bestRoute

if __name__ == '__main__':
    TSP_cities = []

    for i in cities :
        location = i
        x = cities[i][0]
        y = cities[i][1]
        TSP_cities.append(City(location, x=x, y=y))

    GA.makeTSPGraph(TSP_cities, 200, 0.05, 200)

    