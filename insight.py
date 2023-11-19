# Provide insight into what the model has learned

import argparse
import functools
import math
import os
import random

import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import statistics




def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Provide insight to what the model has learned')
    parser.add_argument('-mo', '--model_file', help='The model to be evalutated', required=True)
    parser.add_argument('-N', '--sections', help='The number of sections in the document', required=True, type=int)

    args = vars(parser.parse_args())

    return ( args["model_file"], args["sections"])



def set_value( vector, x, y, value):
    """
    Change the value of the given cell in the vector
    :param vector:
    :param x:
    :param y:
    :param value:
    :return: the new vector
    """

    N = int(math.sqrt(len(vector)))
    vector[x * N + y] = value
    return vector


def high_score(vector, x, y):
    """
    Sets a high score in the given cell of the vector
    :param vector:
    :param x:
    :param y:
    :return: the new vector
    """

    return set_value(vector, x, y, random.uniform(0.8, 1.0))

def medium_score(vector, x, y):
    """
    Sets a medium score in the given cell of the vector
    :param vector:
    :param x:
    :param y:
    :return: the new vector
    """

    return set_value(vector, x, y, random.uniform(0.2, 0.79))

def low_score(vector, x, y):
    """
    Sets a low score in the given cell of the vector
    :param vector:
    :param x:
    :param y:
    :return: the new vector
    """

    return set_value(vector, x, y, random.uniform(0.0, 0.2))


def sections_at_start(vector, N, nr_of_sections, connected_to, value_function):
    """
    Creates a list of scores where sections at the start have good matches
    :param nr_of_sections:
    :param connected_to: connected_to sections
    :param value_function: gets the score for the vector
    :param vector:
    :param N:
    :return:
    """

    for x in range(0, nr_of_sections):
        for y in range( 0, nr_of_sections):
                vector = value_function(vector, x, y)

    return vector

def sections_in_middle(vector, N, nr_of_sections, connected_to, value_function):
    """
    Creates a list of scores where sections in the middle have good matches
    :param nr_of_sections:
    :param connected_to: connected_to sections
    :param value_function: gets the score for the vector
    :param vector:
    :param N:
    :return:
    """

    start = (N // 2) - (nr_of_sections // 2)
    for x in range(0, nr_of_sections):
        for y in range( 0, connected_to):
                vector = value_function(vector, start + x, start + y)

    return vector

def sections_at_end(vector, N, nr_of_sections, connected_to, value_function):
    """
    Creates a list of scores where sections in the middle have good matches
    :param nr_of_sections:
    :param connected_to: connected_to sections
    :param value_function: gets the score for the vector
    :param vector:
    :param N:
    :return:
    """

    start = N - nr_of_sections
    for x in range(0, nr_of_sections):
        for y in range( 0, connected_to):
                vector = value_function(vector, start + x, start + y)

    return vector



def initial_vector( vector, N):
    """
    returns the initialized vector
    :param vector:
    :param N:
    :return:
    """

    return vector



def get_score(fill_function, model, N, iteration_count):
    """
    Gets the average score of iteration_count iterations. The fill_function fills the vector with the right values
    :param fill_function: function( vector, N) where vector is an initialized vector and N * N is the dimension of the vector
    :param model: the model to evaluate on
    :param N:
    :param iteration_count:
    :return:
    """
    score = 0.0
    for i in range(0, iteration_count):
        vector = fill_function(create_vector(N), N)
        Y = model( torch.tensor(vector))[0]
        score += Y.item()

    return score / float(iteration_count)



def create_vector(N):
    """
    Create a vector with all low scores
    :param N:
    :return:
    """

    size = N * N
    vector =  [0] * size
    for x in range(0, N):
        for y in range( 0, N):
            vector = low_score(vector, x, y)

    return vector


class NeuralNetworkPlain(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkPlain, self).__init__()

        self.hidden1 = nn.Linear(N*N, 5)
        self.dropout1 = nn.Dropout(0.0)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.act_output = nn.Sigmoid()

        torch.nn.init.xavier_uniform( self.hidden1.weight)
        torch.nn.init.xavier_uniform( self.output.weight)

    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1(x1)
        x_out = self.act_output(self.output(do1))

        return x_out


## Main part
if __name__ == '__main__':
    (model_file, N) = read_arguments()

    model = torch.load( model_file);
    nr_of_iterations = 1000

    for value_type in ["high", "medium"]:
        value_function = high_score if value_type == "high" else medium_score

        inital_score = get_score(initial_vector, model, N, nr_of_iterations)
        print(f"Initial vector: {inital_score}")

        for connect in range(1, N):
            score = get_score(functools.partial(sections_at_start, nr_of_sections=N, connected_to=connect, value_function=value_function), model, N,
                              nr_of_iterations)
            print(f"All sections connected to {connect} sections with {value_type} values: {score}")

        for count in range(1, 5):
            for connect in range( 1, count):
                score = get_score(functools.partial(sections_at_start, nr_of_sections=count, connected_to=connect, value_function=value_function), model, N, nr_of_iterations)
                print(f"{count} sections connected to {connect} sections with {value_type} values, at start: {score}")

        for count in range(1, 5):
            for connect in range( 1, count):
                score = get_score(functools.partial(sections_in_middle, nr_of_sections=count, connected_to=connect, value_function=value_function), model, N, nr_of_iterations)
                print(f"{count} sections connected to {connect} sections with {value_type} values, at middle: {score}")

        for count in range(1, 5):
            for connect in range( 1, count):
                score = get_score(functools.partial(sections_at_end, nr_of_sections=count, connected_to=connect, value_function=value_function), model, N, nr_of_iterations)
                print(f"{count} sections connected to {connect} sections with {value_type} values, at end: {score}")

