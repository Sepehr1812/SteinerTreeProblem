"""
This program performs Genetic Algorithm for Steiner Tree Problem
"""
from math import sqrt
from random import randrange, randint, uniform
from typing import List


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

    def fitness(self, terminal_vertices_num: list, edges: list, edge_costs: list):
        """
        :return: fitness of chromosome. 0 if self is not connected
        """
        if not self.is_connected(terminal_vertices_num, edges):
            return 0

        f = 0
        for i in range(len(self.gens)):
            if self.gens[i] == 0:
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

        available_edges = []
        for i in range(len(self.gens)):
            if self.gens[i] == 1:
                available_edges.append(edges[i])

        seen_vertices.append(terminal_vertices_num[0])
        seen_terminal_vertices_number += 1
        j = 0
        while True:
            v = seen_vertices[j]
            for e in available_edges:
                if e[0] == v:
                    if not seen_vertices.__contains__(e[1]):
                        seen_vertices.append(e[1])
                        if e[1] in terminal_vertices_num:
                            seen_terminal_vertices_number += 1
                elif e[1] == v:
                    if not seen_vertices.__contains__(e[0]):
                        seen_vertices.append(e[0])
                        if e[0] in terminal_vertices_num:
                            seen_terminal_vertices_number += 1

            if seen_terminal_vertices_number == len(terminal_vertices_num):
                return True

            j += 1
            if j >= len(seen_vertices):
                break

        return False


def parent_selection(population: List[Chromosome], parents_num: int, terminal_vertices_num: list, edges: list,
                     edge_costs: list):
    """
    selects parents
    :param population: current population
    :param parents_num: number of parents we need
    :return: parents
    """
    fitnesses = []
    for p in population:
        fitnesses.append(p.fitness(terminal_vertices_num, edges, edge_costs))

    _sum = sum(fitnesses)

    # create sum fitness array
    sum_fitnesses = []
    sum_fitness = 0
    for i in range(len(fitnesses)):
        sum_fitness += fitnesses[i]
        sum_fitnesses.append(sum_fitness)

    # print(sum_fitnesses)
    # choose parents due to their fitness
    parents = []
    i = 0
    while i < parents_num:
        rand = randint(1, _sum)
        for j in range(len(sum_fitnesses)):
            if rand < sum_fitnesses[j]:
                parents.append(population[j])
                i += 1
                break

    return parents


def child_selection(population: list, tournament_size: int, terminal_vertices_num: list, edges: list, edge_costs: list):
    """
    selects children from population via Q tournament procedure
    :param population: current population
    :return: children array
    """
    children = []
    tournament_number = int(len(population) / tournament_size)
    for k in range(tournament_number):
        best_fitness = -1  # minimum number for this variable
        best_chromosome = Chromosome(None)
        for i in range(tournament_size):
            p_index = randrange(0, len(population))
            selected: Chromosome = population[p_index]

            f = selected.fitness(terminal_vertices_num, edges, edge_costs)
            if f > best_fitness:
                best_fitness = f
                best_chromosome = selected
            population.remove(selected)

        children.append(best_chromosome)

    return children


def crossover(parents: List[Chromosome], population_size: int, edge_num: int, pc: float):
    """
    performs crossover operation
    :param parents: parents array
    :param population_size: size of population we need to generate
    :return: generated generation
    """
    new_generation = []

    for i in range(population_size):
        p = uniform(0, 1)
        a, b = parents[i * 2], parents[i * 2 + 1]
        if p < pc:
            new_genes = a.get_sub_chromosome(int(p * edge_num), True)
            new_genes.extend(b.get_sub_chromosome(int(p * edge_num), False))
            new_generation.append(Chromosome(new_genes))

            new_genes = b.get_sub_chromosome(int(p * edge_num), True)
            new_genes.extend(a.get_sub_chromosome(int(p * edge_num), False))
            new_generation.append(Chromosome(new_genes))
        else:
            new_generation.append(a)
            new_generation.append(b)

    return new_generation


def mutation(population: List[Chromosome], pm: float):
    """
    mutates chromosomes due to pm
    :param pm: probability of mutation
    """
    for i in range(len(population)):
        if uniform(0, 1) <= pm:
            population[i].mutate()


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


def create_result(chromosome: Chromosome, edges_cost: list):
    """
    creates final result file
    :param chromosome: best chromosome (problem answer)
    :param edges_cost: edges cost as we calculated before
    """
    cost_sum = 0
    f = open("steiner_out.txt", "w")

    for i in range(len(chromosome.gens)):
        if chromosome.gens[i] == 1:
            cost_sum += edges_cost[i]
            f.write(str(i) + "\n")

    f.write(str(cost_sum))
    f.close()


def main():
    """
    the main function
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
    terminal_vertices_num: list  # numbers of terminal vertices

    # problem constants
    cal_fitness_num = 1000
    population_size = 20
    parents_num = 40  # number of parents in each parents selection
    tournament_size = 3
    pc = 0.8
    pm = 0.01

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

    terminal_vertices_num = list(range(sv_num, sv_num + tv_num))
    print(terminal_vertices_num)

    c = Chromosome([0, 1, 1, 0])
    print(c.is_connected(terminal_vertices_num, edges))

    # initializing population
    for i in range(population_size):
        genes = []
        for j in range(edge_num):
            genes.append(randint(0, 1))
        population.append(Chromosome(genes))
        print(genes, end=", ")

    # evaluation
    for k in range(int(cal_fitness_num / population_size)):
        # parents selection
        parents = parent_selection(population, parents_num, terminal_vertices_num, edges, edge_costs)
        new_generation = crossover(parents, population_size, edge_num, pc)  # crossover
        mutation(new_generation, pm)  # mutation
        new_generation.extend(population)  # mu + lambda
        # children selection for mu + lambda
        population = child_selection(new_generation, tournament_size, terminal_vertices_num, edges, edge_costs)

        # calculating fitnesses
        fitnesses.clear()
        for p in population:
            print(p.gens, end=", ")
            fitnesses.append(p.fitness(terminal_vertices_num, edges, edge_costs))

        print("\nGeneration #" + str(k + 1) + ":")
        print("Best Fitness: " + str(max(fitnesses)))
        print("Worst Fitness: " + str(min(fitnesses)))
        print("Average Fitness: " + str(sum(fitnesses) / len(fitnesses)))

    # find best chromosome
    best_chromosome = population[0]
    best_fitness = max(fitnesses)
    for p in population:
        best_chromosome_fitness = best_chromosome.fitness(terminal_vertices_num, edges, edge_costs)
        if best_chromosome_fitness == best_fitness:
            break
        if p.fitness(terminal_vertices_num, edges, edge_costs) > best_chromosome_fitness:
            best_chromosome = p

    # printing results
    print("\n\nFinished!")
    print("the best chromosome is " + str(best_chromosome.gens))

    # creating final file
    create_result(best_chromosome, edge_costs)


if __name__ == '__main__':
    main()
