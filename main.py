"""
This program performs Genetic Algorithm for Steiner Tree Problem
"""
from math import sqrt, log2
from random import randrange, randint, uniform
from typing import List

import matplotlib.pyplot as plt


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
    selects parents via fitness proportional selection - round weal
    :param population: current population
    :param parents_num: number of parents we need
    :return: parents
    """
    parents = []

    fitnesses = []
    for p in population:
        fitnesses.append(p.fitness(terminal_vertices_num, edges, edge_costs))

    _sum = sum(fitnesses)
    if _sum < 1:  # all fitnesses are 0
        parents = population
        parents.extend(population)
        return parents

    # create sum fitness array
    sum_fitnesses = []
    sum_fitness = 0
    for i in range(len(fitnesses)):
        sum_fitness += fitnesses[i]
        sum_fitnesses.append(sum_fitness)

    # choose parents due to their fitness
    i = 0
    while i < parents_num:
        rand = randint(1, int(_sum))
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
    calculates log in base 2 of cost of each edge and set the result in edge_costs
    :param all_vertices: all vertices list
    :param edges: all edges list
    """
    for i in edges:
        a, b = i
        edge_costs.append(
            log2(sqrt(abs(all_vertices[a][0] - all_vertices[b][0]) + abs(all_vertices[a][1] - all_vertices[b][1]))))


def read_input(steiner_vertices: list, terminal_vertices: list, edges: list):
    """
    reads from input file and fills the arrays we need
    :param steiner_vertices: steiner vertices coordinates
    :param terminal_vertices: terminal vertices coordinates
    :param edges: edges start and end vertices
    """
    f = open("steiner_in.txt")
    sv_num, tv_num, edge_num = f.readline().split("\n")[0].split(" ")

    for i in range(int(sv_num)):
        line = f.readline().split("\n")[0].split(" ")
        steiner_vertices.append((int(line[0]), int(line[1])))

    for i in range(int(tv_num)):
        line = f.readline().split("\n")[0].split(" ")
        terminal_vertices.append((int(line[0]), int(line[1])))

    for i in range(int(edge_num)):
        line = f.readline().split("\n")[0].split(" ")
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
            cost_sum += pow(2, edges_cost[i])
            f.write(str(i) + "\n")

    f.write(str(cost_sum))
    f.close()


def draw_final_graph(all_vertices: list, terminal_vertices: list, edges: list, chromosome: Chromosome):
    x = []
    y = []
    tx = []
    ty = []

    for v in terminal_vertices:
        tx.append(v[0])
        ty.append(v[1])

    for i in range(len(chromosome.gens)):
        if chromosome.gens[i] == 1:
            e = edges[i]
            v1 = all_vertices[int(e[0])]
            v2 = all_vertices[int(e[1])]
            x.append(v1[0])
            x.append(v2[0])
            y.append(v1[1])
            y.append(v2[1])

    plt.plot(x, y, color='blue', linewidth=0.5)  # plotting edges
    plt.scatter(tx, ty, marker='o', color='red', s=30)  # scattering vertices
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)

    plt.show()


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
    cal_fitness_num = 5000
    population_size = 50
    parents_num = 100  # number of parents in each parents selection
    tournament_size = 3
    pc = 0.8  # crossover probability
    pm = 0.01  # mutation probability

    # fitnesses values
    fitnesses = []

    # initial variables
    read_input(steiner_vertices, terminal_vertices, edges)
    sv_num, tv_num, edge_num = len(steiner_vertices), len(terminal_vertices), len(edges)
    all_vertices.extend(steiner_vertices)
    all_vertices.extend(terminal_vertices)

    calculate_costs(all_vertices, edges, edge_costs)
    terminal_vertices_num = list(range(sv_num, sv_num + tv_num))

    # initializing population
    # adding an optional chromosome to avoid all zero fitnesses
    population.append(Chromosome([0, 1, 1, 1] * int(edge_num / 4)))
    for i in range(population_size - 1):
        genes = []
        for j in range(edge_num):
            genes.append(randint(0, 1))
        population.append(Chromosome(genes))

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
    print("\n\nStop Condition Reached!")
    print("The Best Chromosome Genome in the Last Generation is " + str(best_chromosome.gens))

    # creating final file
    create_result(best_chromosome, edge_costs)
    draw_final_graph(all_vertices, terminal_vertices, edges, best_chromosome)


if __name__ == '__main__':
    main()
