"""
This program performs Genetic Algorithm for Steiner Tree Problem
"""
from math import sqrt
from random import randrange


class Chromosome:
    """
    class for chromosomes.
    """
    gens = []

    def __init__(self, gen_in):
        self.gens = gen_in

    def mutate(self):
        """
        mutates the chromosome by changing a random gen to another digit
        """
        random_gen = randrange(0, len(self.gens))
        self.gens[random_gen] = abs(self.gens[random_gen] - 1)

    def get_sub_chromosome(self, cut_index: int, first_part: bool):
        """
        generates sub chromosomes for crossover operation
        :param cut_index the index we generate sub chromosome from or until there
        :param first_part: specifies we need first part of chromosome or not
        :return: the sub chromosome
        """
        if first_part:
            return self.gens[:cut_index]
        else:
            return self.gens[cut_index:]

    def fitness(self, edge_costs):
        """
        :return: fitness of chromosome
        """
        f = 0
        for i in range(len(self.gens)):
            if self.gens[i] == 1:
                f += edge_costs[i]

        return f

    def is_connected(self, terminal_vertices_num: list, edges: list):
        """
        :param terminal_vertices_num: terminal vertices number list
        :param edges: edges list
        :return: {True} if the graph with selected edges in this chromosome is connected, {False} otherwise
        """
        seen_vertices = []
        seen_terminal_vertices_number = 0

        seen_vertices.append(terminal_vertices_num[0])
        seen_terminal_vertices_number += 1
        j = 0
        while True:
            v = seen_vertices[j]
            for i in edges:
                if i[0] == v:
                    if not seen_vertices.__contains__(i[1]):
                        seen_vertices.append(i[1])
                        if i[1] in terminal_vertices_num:
                            seen_terminal_vertices_number += 1
                elif i[1] == v:
                    if not seen_vertices.__contains__(i[0]):
                        seen_vertices.append(i[0])
                        if i[0] in terminal_vertices_num:
                            seen_terminal_vertices_number += 1

            if seen_terminal_vertices_number == len(terminal_vertices_num):
                return True

            j += 1
            if j >= len(seen_vertices):
                break

        return False


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


def calculate_costs(all_vertices: list, edges: list, edge_costs: list):
    """
    calculates cost of each edge and set the result in edge_costs
    :param all_vertices: all vertices list
    :param edges: all edges list
    """
    for i in edges:
        a, b = i
        edge_costs.append(
            sqrt(abs(all_vertices[a][0] - all_vertices[b][0]) + abs(all_vertices[a][1] - all_vertices[b][1])))


def read_input(steiner_vertices: list, terminal_vertices: list, edges: list):
    """
    reads from input file and fills the arrays we need
    :param steiner_vertices: steiner vertices coordinates
    :param terminal_vertices: terminal vertices coordinates
    :param edges: edges start and end vertices
    """
    f = open("steiner_in_test.txt")
    sv_num, tv_num, edge_num = f.readline()[:5].split(" ")

    for i in range(int(sv_num)):
        line = f.readline()[:3].split(" ")
        steiner_vertices.append((int(line[0]), int(line[1])))

    for i in range(int(tv_num)):
        line = f.readline()[:3].split(" ")
        terminal_vertices.append((int(line[0]), int(line[1])))

    for i in range(int(edge_num)):
        line = f.readline()[:3].split(" ")
        edges.append((int(line[0]), int(line[1])))


def create_result(chromosome: list, edges_cost: list):
    """
    creates final result file
    :param chromosome: best chromosome (problem answer)
    :param edges_cost: edges cost as we calculated before
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
    all_vertices = []
    edges = []
    edge_costs = []

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
    all_vertices.extend(steiner_vertices)
    all_vertices.extend(terminal_vertices)
    print(sv_num, tv_num, edge_num)
    print(all_vertices)

    calculate_costs(all_vertices, edges, edge_costs)
    print(edge_costs)

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
