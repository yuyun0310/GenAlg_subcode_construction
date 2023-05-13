import numpy as np
import pandas as pd
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import argparse

# selection based on fitness score
def selection(pop, scores, T):
	sorted_pop_score = sorted(zip(scores, pop), key=lambda pair: pair[0], reverse=True)
	sorted_pop = [individual for (score, individual) in sorted_pop_score]
	selected_pop = sorted_pop[0:T]
	return selected_pop

# crossover two parents to create two children
def crossover(p1, p2):
	c = p1.copy()
	for i in range(0, len(c[0])):
		if p1[0][i] != p2[0][i]:
			c[0][i] = np.random.randint(0, 2)

	return [c]

# mutation operator
def mutation(bitstring, r_mut):
	result = bitstring.copy()
	for i in range(len(result[0])):
		# check for a mutation
		if np.random.rand() < r_mut:
			# flip the bit
			result[0][i] = 1 - result[0][i]

	return result

# initialization (varying density)
def init(n_bits, n_pop):
	pop = []
	for i in range(0, n_pop):
		zero_rate = np.random.rand()
		num_zero = int(n_bits * zero_rate)
		num_one = int(n_bits - num_zero)
		
		single = []
		row = np.array([0] * num_zero + [1] * num_one)
		np.random.shuffle(row)
		single.append(row.tolist())

		pop.append(single)
	
	pop = np.array(pop)
	return pop

# genetic algorithm
def genetic_algorithm(vectors, objective, n_bits, n_iter, n_pop, T, r_mut, delta, early_stopping=False, early_stop_rate=30):
	samples = np.copy(vectors)
	
	start_time = time.time()

	# keep track of best solution (Hash Table)
	H_hash_table = {}
	H = []
	
	# create matrix H line by line (delta)
	for index in range(0, delta):
		print("**************************\nDelta = %d\n**************************" % (index + 1))

		# initial population of random bitstring
		pop = init(n_bits=n_bits, n_pop=n_pop)

		best = pop[0]
		best_eval = objective(samples, pop[0], H_hash_table)
		best_iter_record = 0

		if(len(samples) == 0):
			print("Index: {}, Estimated FER: {}, Time cost: {:.2f} sec".format((index + 1), math.exp(-best_eval), (time.time() - start_time)))
			break
	
		# enumerate generations
		for gen in range(n_iter):
			# evaluate all candidates in the population
			scores = [objective(samples, c, H_hash_table) for c in pop]

			# check for new best solution
			for i in range(n_pop):
				if scores[i] > best_eval:
					best, best_eval = pop[i], scores[i]
					best_iter_record = gen
					print("> %d iteration, new best f(%s) = %.3f" % (gen,  pop[i], math.exp(-scores[i])))

			if early_stopping and (gen - best_iter_record) > early_stop_rate:
				print("Row index: {} - stop at iteration {}.".format(index, gen))
				break

			if gen == n_iter-1:
				print("Row index: {} - stop at iteration {}.".format(index, gen))

			# select parents
			selected = selection(pop, scores, T)
			
			# create the next generation
			children = list(selected)

			# mutation
			for p in selected:
				c = mutation(p, r_mut)
				children.append(c)

			# crossover
			for i in range(0, T-1):
				for j in range(i+1, T):
					p1, p2 = selected[i], selected[j]
					crossover_children = crossover(p1, p2)
					for c in crossover_children:
						children.append(c)

			# replace population
			pop = children
			pop = np.array(pop)

		H.append(best[0])

		print("Index: {}, Estimated FER: {}, Time cost: {:.2f} sec".format((index + 1), math.exp(-best_eval), (time.time() - start_time)))

		# update samples: only samples with binary multiplication = 0 will be remained
		if index != (delta - 1):
			samples = update_samples(samples, best)

	H = np.array(H)

	return H, get_cardinality(vectors, H)

# the function to solve binary multiplication
def binary_multiplication(vector, H_transpose):
	mul = np.matmul(vector, H_transpose)
	mul = mul % 2
	return mul

# fitness function (objective function)
def objective(vectors, H, H_hash_table):
	H_tuple = tuple(map(tuple, H))

	# get score from hash table if it is already been calculated
	if H_tuple in H_hash_table:
		return H_hash_table[H_tuple]

	cardinality = get_cardinality(vectors=vectors, H = H)
	try:
		fitness = -math.log(cardinality)
	except ValueError:
		fitness = math.inf
	H_hash_table[H_tuple] = fitness

	return fitness

# the function to get cardinality of the individual
def get_cardinality(vectors, H):
    cardinality = 0
    for vector in vectors:
        # Get binary matrix transpose
        H_transpose = H.transpose()
        mul = binary_multiplication(vector[:-1], H_transpose)

        num_non_zero = np.count_nonzero(mul)
        if num_non_zero == 0:
            cardinality += vector[-1]

    return cardinality

def update_samples(samples, H):
	new_samples = []
	for vector in samples:
		H_transpose = H.transpose()
		mul = binary_multiplication(vector[:-1], H_transpose)

		num_non_zero = np.count_nonzero(mul)
		if num_non_zero == 0:
			new_samples.append(vector)
	
	new_samples = np.array(new_samples)
	return new_samples

# the function to load dataset
def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path, header=None)
    vectors = np.array(df.iloc[:, 1:])
    return vectors

def save_parity_check_matrices(H, delta):
	with open("H_delta=" + str(delta) + "_weighted-stop=" + str(early_stop_rate) + "-T=" + str(T) + "-r_mut=" + str(r_mut) + ".csv", "wb") as f:
		np.savetxt(f, H, fmt='%s', delimiter=',', header='')

# main function
"""
row by row in H
"""
if __name__ == "__main__":
	# Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    parser.add_argument("-p", "--path")
    parser.add_argument("-d", "--delta")
    parser.add_argument("-n", "--num_iter")
    parser.add_argument("-t", "--T")
    parser.add_argument("-e", "--early_stop_rate")
    parser.add_argument("-r", "--r_mut")

    # Parse the argument
    args = parser.parse_args()
    
    # Load codewords
    file_path = args.path
    vectors = load_dataset(file_path)
    
    # Delta defines there are how many rows in binary matrix H.
    delta = int(args.delta)
    
    # num_iter definds the number of experimented random vector generated.
    num_iter = int(args.num_iter)
    T = int(args.T)
    num_pop = int((math.pow(T, 2) + 3 * T) / 2) # num_pop = T + (T 2) + T = (T^2 + 3T) / 2

	# Initially, the binary matrix H is with size 0.
	# Finally, the binary matrix H is with size (delta, 64) or (delta, 128).
    vector_len = len(vectors[0][:-1])
    H = None
    
    early_stop_rate = int(args.early_stop_rate)
    r_mut = float(args.r_mut)

	# Use greedy method to generate binary matrix H.
    H, cardinality = genetic_algorithm(vectors=vectors, 
										objective=objective, 
										n_bits=vector_len, 
										n_iter=num_iter, 
										n_pop=num_pop, 
										T=T,
										r_mut=r_mut, 
										delta=delta,
										early_stopping=True,
										early_stop_rate=early_stop_rate)
    
    print("The cardinality is: {}".format(cardinality))
    
    save_parity_check_matrices(H, delta)