import numpy as np
import matplotlib.pyplot as plt

def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 2

def generate_population(population_size, individual_size):
    return np.array([[np.random.randint(2) for _ in range(individual_size)] for _ in range(population_size)])

def decode_individual(individual):
    LOWER_BOUND, UPPER_BOUND = -1, 2
    return LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * sum([bit * 2**i for i, bit in enumerate(reversed(individual))]) / (2**len(individual) - 1)

def calculate_fitness(individual):
    return fitness_function(decode_individual(individual))

def mutate_individual(individual, mutation_rate):
    return [not bit if np.random.random() < mutation_rate else bit for bit in individual]

def reproduce_individuals(individual1, individual2, mutation_rate):
    split_point = np.random.randint(len(individual1))
    child1 = np.concatenate((individual1[:split_point], individual2[split_point:]))
    child2 = np.concatenate((individual2[:split_point], individual1[split_point:]))
    return [mutate_individual(child1, mutation_rate), mutate_individual(child2, mutation_rate)]

def form_next_generation(population, mutation_rate):
    population_size = len(population)
    fitness_values = [calculate_fitness(individual) for individual in population]
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    next_population = sorted_population[:POPULATION_SIZE]
    for i, individual1 in enumerate(sorted_population):
        for _, individual2 in enumerate(sorted_population[i+1:], start=i+1):
            children = reproduce_individuals(individual1, individual2, mutation_rate)
            for child in children:
                if child.tolist() not in next_population.tolist():
                    next_population = np.append(next_population, [child], axis=0)
                    if len(next_population) == POPULATION_SIZE:
                        return next_population
    return next_population

POPULATION_SIZE, INDIVIDUAL_SIZE, MUTATION_RATE = 10, 7, 0.01
POPULATION = generate_population(POPULATION_SIZE, INDIVIDUAL_SIZE)
generation = 0

x = np.linspace(-1, 2, 2**7)
y = fitness_function(x)

while True:
    fitness_values = [calculate_fitness(individual) for individual in POPULATION]
    individuals = [decode_individual(individual) for individual in POPULATION]
    plt.title(f"Generation {generation}")
    plt.plot(x, y)
    plt.scatter(individuals, fitness_values)
    plt.plot(individuals, fitness_values)
    plt.show()
    POPULATION = form_next_generation(POPULATION, MUTATION_RATE)
    generation += 1
