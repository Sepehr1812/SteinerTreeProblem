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


def read_input(steiner_vertices: list, terminal_vertices: list, edges: list):
    f = open("steiner_in_test.txt")
    sv_num, tv_num, edge_num = f.readline()[:5].split(" ")

    for i in range(int(sv_num)):
        line = f.readline()[:3].split(" ")
        steiner_vertices.append((line[0], line[1]))

    for i in range(int(tv_num)):
        line = f.readline()[:3].split(" ")
        terminal_vertices.append((line[0], line[1]))

    for i in range(int(edge_num)):
        line = f.readline()[:3].split(" ")
        edges.append((line[0], line[1]))


def create_result(chromosome: list, edges_cost: list):
    """
    creates final result file
    :param chromosome: best chromosome (problem answer)
    :param edges_cost: edges cost as we calculated before
    :return: None
    """
    cost_sum = 0
    f = open("steiner_out.txt", "w")

    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            cost_sum += edges_cost[i]
            f.write(str(i))

    f.write(str(cost_sum))
    f.close()


def main():
    """
    the main function
    :return: nothing
    """
    # population array
    population = []

    # problem variables
    sv_num: int  # number of steiner vertices
    tv_num: int  # number of terminal vertices
    edge_num: int  # number of edges
    steiner_vertices = []
    terminal_vertices = []
    edges = []
    edges_cost = []

    # problem constants
    cal_fitness_num = 1000
    population_size = 20
    tournament_size = 4
    mutation_rate: float

    # fitnesses values
    fitnesses = []

    read_input(steiner_vertices, terminal_vertices, edges)
    print(steiner_vertices, terminal_vertices, edges)
    sv_num, tv_num, edge_num = len(steiner_vertices), len(terminal_vertices), len(edges)
    print(sv_num, tv_num, edge_num)

    # fitnesses attributes
    # max_fitnesses = []
    # min_fitnesses = []
    # avg_fitnesses = []

    # initializing population
    # for i in range(population_size):
    #     genes = []
    #     for j in range(chromosome_size):
    #         genes.append(randrange(0, colors_num))
    #     population.append(Chromosome(genes))
    #
    # # evaluation
    # for k in range(int(cal_fitness_num / population_size)):
    #     parents = parent_selection(population, tournament_size)  # step 3
    #     new_generation = crossover(parents, population_size)  # step 4
    #     # for i in range(int(mutation_num)):
    #     #     new_generation[randrange(0, len(new_generation))].mutation()  # step 5
    #
    #     # calculating fitnesses L340
    #     fitnesses.clear()
    #     for x in new_generation:
    #         fitnesses.append(x.fitness())
    #     print(max(fitnesses))
    #     print(min(fitnesses))
    #     print(sum(fitnesses) / len(fitnesses))
    #
    #     population = new_generation  # step 6
    #
    # # printing results
    # print("\n\nSearch Completed!")
    # for i in range(generations_num):
    #     print("\nGeneration #" + str(i + 1) + ":\nMax Fitness: " + str(max_fitnesses[i]) + "\nMin Fitness: " +
    #           str(min_fitnesses[i]) + "\nAverage Fitness: " + str(avg_fitnesses[i]))


if __name__ == '__main__':
    main()
