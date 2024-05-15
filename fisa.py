import numpy as np
import random
import math
import csv

# Population Initialization
def init_population(size, dim, lb, ub):
    population = np.array([[random.uniform(lb[i], ub[i]) for i in range(dim)] for _ in range(size)])
    return population

# Objective function definition
def myobj(solution):
    # sphere function
    # fitness = np.sum(solution**2)
    # return fitness

    # gear train
    # x1 = solution[0]
    # x2 = solution[1]
    # x3 = solution[2]
    # x4 = solution[3]
    # result = ((1/6.931) - (x3*x2)/(x1*x4))**2
    # return result

    # pressure vessel
    # x1 = solution[0]
    # x2 = solution[1]
    # x3 = solution[2]
    # x4 = solution[3]

    # g1 = -x1 + 0.0193 * x3
    # g2 = -x2 + 0.00954 * x3
    # g3 = 1296000 - (4/3)*math.pi*(x3 ** 3) - math.pi*(x3 ** 2)*(x4)
    # g4 = x4 - 240

    # if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
    #     return 0.6224*x1*x3*x4 + 1.7781*x2*x3*x3 + 3.1661*x1*x1*x4 + 19.84*x1*x1*x3
    # else:
    #     return 1e10

    # compression spring
    # x1 = solution[0]
    # x2 = solution[1]
    # x3 = solution[2]
    # result = (x3 + 2) * x1 * x1 * x2

    # g1 = 1 - ((x2*x2*x2*x3) / (71785*x1*x1*x1*x1))
    # g2 = (((4*x2*x2) - (x1*x2)) / (12566*(x1*x1*x1*x2 - x1*x1*x1*x1))) + (1 / (5108*x1*x1)) - 1
    # g3 = 1 - ((140.45*x1) / (x2*x2*x3))
    # g4 = ((x1+x2)/1.5) - 1

    # if(g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0):
    #     return result
    
    # return 1e10

    # welded beam
    x1 = solution[0]
    x2 = solution[1]
    x3 = solution[2]
    x4 = solution[3]

    # Calculating result.
    result = (1.10471*x1*x1*x2) + (0.04811*x3*x4*(14 + x2))

    # Initializing the given basic constraint values.
    G = 12000000    # psi
    E = 30000000    # psi
    P = 6000    #lb
    L = 14  #inch

    toumax = 13600  #psi
    sigmamax = 30000  #psi

    # Calculating Constraint Values
    del_x = ((4*P*L*L*L) / (E*x3*x3*x3*x4))
    sigma_x = ((6*P*L) / (x3*x3*x4))

    temp1 = math.sqrt((E*G*x3*x3*x4*x4*x4*x4*x4*x4) / 36)
    temp2 = E / (4*G)
    Pc_x = ((4.013*temp1) / L*L) * (1 - ((x3 / (2*L))*(temp2)))
    M_x = P*(L + (x2/2))
    R_x = math.sqrt((x2*x2/4) + ((x1+x2)/2)**2)
    J_x = 2 * ((x1*x2 / math.sqrt(2)) * (x2*x2/12) + ((x1+x3)/2)**2)

    tou1_x = P / (math.sqrt(2) * x1 * x2)
    tou2_x = M_x*R_x/J_x

    temp = (tou1_x**2) + 2*tou1_x*tou2_x*(x2/2*R_x) + (tou2_x)**2
    tou_x = math.sqrt(temp)

    g1 = tou_x - toumax
    g2 = sigma_x - sigmamax
    g3 = x1 - x4
    g4 = 0.125 - x1
    g5 = del_x - 0.25
    g6 = P - Pc_x

    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0 and g5 <= 0 and g6 <= 0:
        return result
    
    return 1e10



def trim(solution, lb, ub):
    for i in range(len(solution)):
        if solution[i] > ub[i]:
            solution[i] = ub[i]
        if solution[i] < lb[i]:
            solution[i] = lb[i]
    return solution

# Calculating mean.
def mean(arr):
    mean_array = np.mean(arr, axis=0)
    return mean_array

# Finding out the value of best generations for FISA.
def mx_best_gen(population, fitness):
    mx_best = []
    for i in range(len(population)):
        curr = fitness[i]
        best = []
        for j in range(len(population)):
            if fitness[j] <= curr:
                best.append(population[j])
        
        if best:
            mx_best.append(np.mean(best))
        else:
            mx_best.append(curr)
    return mx_best

# Finding out the value of worst generations for FISA.
def mx_worst_gen(population, fitness):
    mx_worst = []
    for i in range(len(population)):
        curr = fitness[i]
        worst = []
        for j in range(len(population)):
            if fitness[j] > curr:
                worst.append(population[j])
        
        if worst:  # Check if worst is not empty
            mx_worst.append(np.mean(worst))
        else:
            mx_worst.append(curr)  # Handle the case when worst is empty
        
    return mx_worst

def fisa(population, dim, itr, objective_function, lb, ub):
    for j in range(itr):
        fitness_values = np.array([objective_function(individual) for individual in population])

        # Find the best and worst individuals in the current population
        best_index = np.argmin(fitness_values)
        worst_index = np.argmax(fitness_values)

        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]

        # print("\nIteration ", j, "\nBest Solution:", best_solution)
        # print("Best Fitness:", best_fitness)

        mx_best = np.array(mx_best_gen(population, fitness_values))
        mx_worst = np.array(mx_worst_gen(population, fitness_values))

        # Update the position of each individual in the population
        for i in range(len(population)):
            # if i != best_index:
                r1= [random.uniform(0, 1) for _ in range(dim)]
                r2= [random.uniform(0, 1) for _ in range(dim)]
                new_sol = population[i] + r1*(mx_best[i] - np.abs(population[i])) + r2*(np.abs(population[i]) - mx_worst[i])

                # Clip the new position to be within the bounds
                new_sol = trim(new_sol, lb, ub)

                new_fitness = objective_function(new_sol)

                if new_fitness < fitness_values[i]:
                    population[i] = new_sol

    # Return the best solution found
    best_solution = population[best_index]
    best_fitness = fitness_values[best_index]
    return best_solution, best_fitness

# Initialization of algorithm parameters
pop_size = 30
dim = 4
itr = 10000
# gear train
lb = [12, 12, 12, 12]
ub = [60, 60, 60, 60]

# Pressure vessel
# lb = [0, 0, 10, 10]
# ub = [100, 100, 200, 200]

# welded beam
lb = [0.1, 0.1, 0.1, 0.1]
ub = [2, 10, 10, 2]

population = init_population(pop_size, dim, lb, ub)
# print("Initial Population:")
# print(population)

best_solution, best_fitness = fisa(population, dim, itr, myobj, lb, ub)

csv_file = "welded_beam_FISA.csv"

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)

    combined_data = best_solution.tolist() + [best_fitness]
    writer.writerow(combined_data)

print("\nBest Solution:", best_solution)
print("Best Fitness:", best_fitness)
