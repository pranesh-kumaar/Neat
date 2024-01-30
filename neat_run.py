import pandas as pd
import neat
import numpy as np
import plot
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor

#Using multiprocessing
def parallel_evaluations(genomes, config, stock_data):
    num_processes = multiprocessing.cpu_count() -2
    pool = multiprocessing.Pool(processes=num_processes)
    results = []

    for genome_id, genome in genomes:
        result = pool.apply_async(evaluate_genome, (genome, config, stock_data))
        results.append((genome_id, result))

    pool.close()
    pool.join()

    evaluated_genomes = []
    for genome_id, result in results:
        fitness = result.get()
        evaluated_genomes.append((genome_id, fitness))

    return evaluated_genomes

########################################################################################################################################

# Function to calculate percentage stock return based on buy/sell signals
def calculate_return(data, signals):
    
    buy_price = None  # Initialize buy price
    holdings = 0
    cumulative_return = 0  # Initialize cumulative return
    daily_returns = []  # List to store daily returns
    strategy_returns = []  # List to store strategy returns
    cumulative_returns = []  # List to store cumulative returns

    for i in range(len(data)):
        signal = signals[i]  # Get the signal from the backtesting module

        # Calculate daily returns
        if i > 0:
            daily_return = (data['Adj Close'].iloc[i] - data['Adj Close'].iloc[i - 1]) / data['Adj Close'].iloc[i - 1]
            daily_returns.append(daily_return)
        else:
            daily_returns.append(0)  # First day has no return

        if holdings == 1:
            cumulative_return += daily_return
            cumulative_returns.append(cumulative_return)
        else:
            cumulative_return = cumulative_return
            cumulative_returns.append(cumulative_return)            

        if signal == 1 and holdings == 0:  # Buy signal and no current holdings
            holdings = 1
        elif signal == -1 and holdings == 1:  # Sell signal and holding stocks
            holdings = 0  # Reset holdings after selling

    return daily_returns, cumulative_returns, cumulative_returns[-1] if len(cumulative_returns) > 0 else 0



#########################################################################################################################################

def generate_signals(net, data, window_size):
        buy_signal = 0.90
        sell_signal = 0.1
        signals = []
        
        if 'Adj Close' in data.columns:
            data = data.drop('Adj Close', axis = 1)

        for i in range(len(data)):
            if i >= window_size:  # Start generating signals after 14 days
                # Extract the previous 14 days' data and preprocess it
                prev_14_days = data.iloc[i - window_size:i]  # Get the previous 14 days' data

                # Initialize a scaler for each window of 14 days' data
                scaler = MinMaxScaler()
                prev_14_days_scaled = np.empty_like(prev_14_days)

                for col_idx, column in enumerate(prev_14_days.columns):
                    column_data = prev_14_days[column].values.reshape(-1, 1)
                    scaled_column = scaler.fit_transform(column_data)
                    prev_14_days_scaled[:, col_idx] = scaled_column.flatten()

                # Generate signal based on the neural network prediction
                output = net.activate(prev_14_days_scaled.flatten()) #prev_14_days_scaled.flatten()
                if output[0] > buy_signal:
                    signals.append(1)  # Buy signal
                elif output[0] < sell_signal:
                    signals.append(-1)  # Sell signal
                else:
                    signals.append(0)  # Hold signal
            else:
                signals.append(0)  # Assign 'Hold' for the initial 14 days

        return signals


def generate_signals_with_pca(net, all_days_data, window_size, variance_threshold):
    buy_signal = 0.9
    sell_signal = 0.1
    signals = []

    for i in range(len(all_days_data)):
        if i >= window_size:
            # Extract the previous 14 days' data and preprocess it
            prev_14_days = all_days_data[i - window_size:i, :]  # Use array slicing for efficiency

            # Initialize a scaler for each window of 14 days' data and scale
            scaler = MinMaxScaler()
            prev_14_days_scaled = scaler.fit_transform(prev_14_days)

            # Apply PCA to retain at least 98% of the variance
            pca = PCA(n_components=variance_threshold)
            prev_14_days_pca = pca.fit_transform(prev_14_days_scaled)

            # Flatten the PCA-transformed data
            prev_14_days_pca_flatten = prev_14_days_pca.flatten()

            # Generate signal based on the neural network prediction
            output = net.activate(prev_14_days_pca_flatten)

            # Use NumPy vectorized operations for signal generation
            signals.append(1 if output[0] > buy_signal else (-1 if output[0] < sell_signal else 0))
        else:
            signals.append(0)  # Assign 'Hold' for the initial 14 days

    return signals
    

############################################################################################################################


def run(window_size, num_gen, feed_data): #### for single data

    stock_data = feed_data.drop('Adj Close', axis = 1)

    # Fitness function to evaluate the neural network based on returns
    def evaluate_genome(genome, config, stock_data):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        signals = generate_signals(net, stock_data, window_size)
        daily_returns, cumulative_returns, final_return = calculate_return(feed_data, signals)
        return final_return

    # NEAT configuration
    config_path = 'path_to_neat_config_file.txt'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create a NEAT population
    population = neat.Population(config)

    # Run NEAT evolution
    num_generations = num_gen
    best_returns = []  # Store best returns for each generation

    def eval_population(genomes, config):
        for genome_id, genome in genomes:
            fitness = evaluate_genome(genome, config, stock_data)
            genome.fitness = fitness

    for generation in range(num_generations):
        print(f"Generation {generation + 1}:")
        generation_returns = []

        for genome_id, genome in population.population.items():
            fitness = evaluate_genome(genome, config, stock_data)
            genome.fitness = fitness
            generation_returns.append(fitness)

        # Find the best return for this generation
        best_return = max(generation_returns)
        best_returns.append(best_return)
        print(f"Best return in Generation {generation + 1}: {best_return}")

        # Evolve the population
        population.run(eval_population, 1)

    # Initialize variables to track the best genome and its fitness
    best_genome_id = None
    best_fitness = -float('inf')  # Set initial best fitness to negative infinity

    # Iterate through the population to find the genome with the best fitness
    for genome_id, genome in population.population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome_id = genome_id

    # Ensure that a best genome is found
    if best_genome_id is not None:
        # Get the best genome and its associated network
        best_genome = population.population[best_genome_id]
        winning_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        # Use winning_net for further operations or analysis
    else:
        print("No best genome found with a valid fitness value.")

    winner = population.population[best_genome_id]
    winning_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Use the winning network to generate signals and calculate returns
    final_signals = generate_signals(winning_net, stock_data, window_size)
    daily_returns, cumulative_returns, final_return = calculate_return(feed_data, final_signals)

    adj_close = feed_data['Adj Close']

    print("Cumulative Final Return:", final_return, " Buy and Hold Return: "+str(adj_close[-1]/adj_close[0] - 1), final_signals)
    
    return winning_net, cumulative_returns, final_signals


#################################################################################################################################

def evaluate_genome_avg(genome, config, stock_data_list, window_size, variance_threshold):
    total_return = []
    for stock_data in stock_data_list:
        if 'Adj Close' in stock_data.columns:
            data = stock_data.drop('Adj Close', axis=1)
            all_days_data = data.values
        else:
            all_days_data = stock_data.values

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        signals = generate_signals_with_pca(net, all_days_data, window_size, variance_threshold)
        _, _, final_return = calculate_return(stock_data, signals)
        if final_return is not None:
            total_return.append(final_return)
    avg_return = sum(total_return) / len(stock_data_list) if total_return else None
    genome.total_return = total_return 
    return avg_return

# Update the NEAT evaluation function
def eval_population_avg(genomes, config, stock_data_list, window_size, variance_threshold):
    for genome_id, genome in genomes:
        fitness = evaluate_genome_avg(genome, config, stock_data_list, window_size, variance_threshold)
        if fitness is not None:
            genome.fitness = fitness
        
def evaluate_genome_avg_parallel(genomes, config, stock_data_list, window_size, variance_threshold):
    
    # num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    # pool = multiprocessing.Pool(processes=num_processes)
    
    # evaluated_genomes = []
    # for genome_id, genome in genomes:
    #     result = pool.apply_async(evaluate_genome_avg, (genome, config, stock_data_list, window_size, variance_threshold))
    #     evaluated_genomes.append((genome_id, result))

    # pool.close()
    # pool.join()

    # # Retrieve evaluated genomes and their fitness
    # evaluated = []
    # for genome_id, result in evaluated_genomes:
    #     fitness = result.get()
    #     evaluated.append((genome_id, fitness))

    # return evaluated

    num_processes = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        evaluated_genomes = pool.starmap_async(
            evaluate_genome_avg,
            [(genome, config, stock_data_list, window_size, variance_threshold) for genome_id, genome in genomes]
        ).get()

    return list(zip([genome_id for genome_id, _ in genomes], evaluated_genomes))


def run_multiple_datasets(window_size, num_gen, data_list, variance_threshold):
    # Create NEAT configuration and population (same as before)
    config_path = 'path_to_neat_config_file.txt'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)

    # Run NEAT evolution
    num_generations = num_gen

    for generation in range(num_generations):
        print(f"Generation {generation + 1}:")
        generation_returns = []

        # Evaluate population on multiple datasets and calculate average return
        # eval_population_avg(population.population.items(), config, data_list, window_size)
        
        evaluated_genomes = evaluate_genome_avg_parallel(population.population.items(), config, data_list, window_size, variance_threshold)
        # Assign fitness to genomes
        for genome_id, fitness in evaluated_genomes:
            population.population[genome_id].fitness = fitness

        # Find the best average return for this generation
        best_return = max(genome.fitness for _, genome in population.population.items())
        
        best_genome_id = None
        best_fitness = -float('inf')
        
        for genome_id, genome in population.population.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
        
        print(f"Best average return in Generation {generation + 1}: {best_fitness}")
        try:
            print(f"Best net's return of all data in Generation {generation + 1}: {best_genome.total_return}")
        except:
            pass
            

        num_processes = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            population.run(lambda genomes, config: eval_population_avg(genomes, config, data_list, window_size, variance_threshold), 1)

        

        # # Evolve the population
        # if __name__ == '__main__':
        #     with multiprocessing.Pool() as pool:
        #     # population.run(lambda genomes, config: pool.starmap(evaluate_genome, [(genome, config) for genome in genomes]), 1)
        #         population.run(lambda genomes, config: pool.starmap(eval_population_avg(genomes, config, data_list, window_size, variance_threshold), 1))

    # Retrieve the best genome and its associated network
    # Iterate through the population to find the genome with the best fitness
    best_genome_id = None
    best_fitness = -float('inf')
    
    for genome_id, genome in population.population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome_id = genome_id

    # Ensure that a best genome is found
    if best_genome_id is not None:
        # Get the best genome and its associated network
        best_genome = population.population[best_genome_id]
        winning_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        # Use winning_net for further operations or analysis
    else:
        print("No best genome found with a valid fitness value.")

    winner = population.population[best_genome_id]
    winning_net = neat.nn.FeedForwardNetwork.create(winner, config)

    return winning_net, best_return	


###############################################################################################################################

