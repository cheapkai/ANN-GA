import numpy as np
import pandas as pd   
import ANN    
import tensorflow as tf 

max_population = 10
max_generations = 10
max_parents = 3
max_offspring = max_population - max_parents

def Population():
    global max_population

    population = []

    i = 0
    while i < max_population:
        chromosome = []
        #Setting chromosome genes 
        chromosome.append(np.random.uniform(10**(-8), 10**(-2))) #Learning Rate
        chromosome.append(np.random.randint(1,20)) #number of hidden layers
        chromosome.append(2**np.random.randint(1,5)) #number of nodes per layer
        chromosome.append(np.random.choice([0, 1])) #Activation function 
        population.append(chromosome)
        i += 1
   
    return population


def Fitness(population, dataset):
    global max_population

    accuracys = []
    i = 0
    while i < max_population:
        parameters = population[i]
        
        accuracy = ANN.Initialization(False, dataset, parameters)
        accuracys.append(accuracy)
        i += 1

    return accuracys


def Parents(population, fitness):
    global max_parents

    parents = []

    i = 0
    while i < max_parents:
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]

        chromosome = []
        chromosome.append(population[max_fitness_idx][0])
        chromosome.append(population[max_fitness_idx][1])
        chromosome.append(population[max_fitness_idx][2])
        chromosome.append(population[max_fitness_idx][3])
        parents.append(chromosome)

        fitness = np.delete(fitness, max_fitness_idx, 0)
        i += 1

    return parents


def Crossover(parents):
    global max_offspring

    crossover_point = len(parents[0])/2

    offsprings = []

    i = 0
    while i < max_offspring:
        #selecting parents to crossover 
        parent1  = parents[i%len(parents)]
        parent2 = parents[i%len(parents)]

        offspring = []
        for gene in range(len(parents[0])):
            
            if gene <= crossover_point:
                offspring.append(parent1[gene])
            else:
                offspring.append(parent2[gene])

        offsprings.append(offspring)
        i += 1
    return offsprings


def Mutation(crossover):

    for i in range(len(crossover)):
         #Generating a gene id to be mutated.
        gene_idx = np.random.randint(0, len(crossover[0]))
        #Saving current gene value
        current_value = crossover[i][gene_idx]

        while(True):

            if gene_idx == 0:
                new_value = np.random.uniform(10**(-8), 10**(-2))
                if current_value != new_value:
                    break
            elif gene_idx == 1:
                new_value = current_value + np.random.uniform(-1.0, 1.0, 1)
                if new_value >= 1 and new_value <= 20:
                    break
            elif gene_idx == 2:
                new_value = current_value +  np.random.uniform(-1.0, 1.0, 1)
                if new_value >= 2 and new_value <= 32:
                    break
            else:
                new_value = 1 if current_value == 0 else 0
                break

            crossover[i][gene_idx] = new_value

    return crossover 

def NewPopulation(population, parents, offspring_mutation):
    population.clear()

    for i in range(len(parents)):
        population.append(parents[i])

    for i in range(len(offspring_mutation)):
        population.append(offspring_mutation[i])
    
    return population


def main():
    global max_generations
    global max_parents

    dataset = ANN.ReadCSV()

    population = Population()

    #Current solution variables.
    score = None
    solution = None #Variable to keep in track the previous fitness value.

    i = 0
    while i < max_generations:
        fitness = Fitness(population, dataset)
        
        #--------------Final Solution variables---------------------
        print(fitness)
        score = max(fitness)
        solution = population[(np.where(fitness == score))[0][0]]
        #--------------END-------------------------------------------

        parents = Parents(population, fitness)

        offspring_crossover = Crossover(parents)

        offspring_mutation = Mutation(offspring_crossover)
        
        #generating a new population.
        population = NewPopulation(population, parents, offspring_mutation)

        if score >= 0.80:
            break
        else:
            i += 1 

    print("Parameters:", solution)
    print("Accuracy: "+"{:.2f} %".format(score))

main()
