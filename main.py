"""
This program performs Genetic Algorithm for Steiner Tree Problem
"""
from random import randrange

# problem values
chromosome_size = 31
edges_num = 74
colors_num = 4


class Chromosome:
    """
    class for chromosomes.
    """
    gens = []

    def __init__(self, gen_in):
        self.gens = gen_in

    def mutation(self):
        """
        mutates the chromosome
        :return: nothing
        """
        random_gen = randrange(0, chromosome_size)
        self.gens[random_gen] = randrange(0, colors_num)

    def get_sub_chromosome(self, first_half):
        """
        generates sub chromosomes for crossover operation
        :param first_half: specifies we need first half of chromosome or not
        :return: the sub chromosome
        """
        if first_half:
            return self.gens[0:15]
        else:
            return self.gens[15:31]

    def fitness(self):
        """
        calculates the fitness of the chromosome
        :return: fitness of chromosome
        """



def parent_selection(population, tournament_size):
    """
    selects parents from population
    :param population: the initial population
    :param tournament_size: the tournament size
    :return: parents array
    """
    parents = []
    loop_cond = int(len(population) / tournament_size)
    for k in range(loop_cond):
        best_fitness = -1  # a negative infinite number for this variable
        best_chromosome = Chromosome(None)
        for i in range(tournament_size):
            p_index = randrange(0, len(population))
            selected = population[p_index]
            if selected.fitness() > best_fitness:
                best_fitness = selected.fitness()
                best_chromosome = selected
            population.remove(selected)

        parents.append(best_chromosome)

    return parents


def crossover(parents, population_size):
    """
    performs crossover operation
    :param parents: parents array
    :param population_size: size of population we need to generate
    :return: generated generation
    """
    new_generation = []

    for i in range(population_size):
        new_genes = parents[randrange(0, len(parents))].get_sub_chromosome(True)
        new_genes.extend(parents[randrange(0, len(parents))].get_sub_chromosome(False))
        new_generation.append(Chromosome(new_genes))

    return new_generation


def main():
    """
    the main function
    :return: nothing
    """
    # population array
    population = []

    # problem variables
    generations_num: int
    population_size: int
    tournament_size: int
    mutation_rate: float

    # fitnesses values
    fitnesses = []

    # fitnesses attributes
    # max_fitnesses = []
    # min_fitnesses = []
    # avg_fitnesses = []

    # getting values
    population_size = int(input("Enter population size: "))
    tournament_size = int(input("Enter tournament size: "))
    mutation_rate = float(input("Enter mutation rate: "))
    generations_num = int(input("Enter generation number: "))

    # calculating mutation number
    mutation_num = population_size * chromosome_size * mutation_rate

    # initializing population
    for i in range(population_size):
        genes = []
        for j in range(chromosome_size):
            genes.append(randrange(0, colors_num))
        population.append(Chromosome(genes))

    for k in range(generations_num):
        parents = parent_selection(population, tournament_size)  # step 3
        new_generation = crossover(parents, population_size)  # step 4
        for i in range(int(mutation_num)):
            new_generation[randrange(0, len(new_generation))].mutation()  # step 5

        # calculating fitnesses L340
        fitnesses.clear()
        for x in new_generation:
            fitnesses.append(x.fitness())
        print(max(fitnesses))
        print(min(fitnesses))
        print(sum(fitnesses) / len(fitnesses))

        population = new_generation  # step 6

    # printing results
    print("\n\nSearch Completed!")
    # for i in range(generations_num):
    #     print("\nGeneration #" + str(i + 1) + ":\nMax Fitness: " + str(max_fitnesses[i]) + "\nMin Fitness: " +
    #           str(min_fitnesses[i]) + "\nAverage Fitness: " + str(avg_fitnesses[i]))


if __name__ == '__main__':
    main()
