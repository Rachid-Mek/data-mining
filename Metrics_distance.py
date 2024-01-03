
import math

import numpy as np


def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    sum_of_squares = sum(pow(p1 - p2, 2) for p1, p2 in zip(point1, point2))
    return math.sqrt(sum_of_squares)

def manhattan_distance(point1, point2):
    
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    
    # both point1 and point2 are vectors
    distance = sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

    return distance


def hamming_distance(point1, point2):
        
        if len(point1) != len(point2):
            raise ValueError("Points must have the same number of dimensions")
        
        distance = sum(abs(p1 - p2) for p1, p2 in zip(point1, point2)) / len(point1)
        
        return distance


def cosine_similarity(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")

    # point 1 and point 2 are vectors
    dot_product = np.dot(point1, point2)
    norm_point1 = np.linalg.norm(point1)
    norm_point2 = np.linalg.norm(point2)

    similarity = dot_product / (norm_point1 * norm_point2)

    # Clip the result to ensure it is between -1 and 1
    similarity = np.clip(similarity, 0, 1.0)

    return similarity



def jaccard_similarity(point1, point2):
            
            if len(point1) != len(point2):
                raise ValueError("Points must have the same number of dimensions")
            
            # point 1 and point 2 are vectors
            intersection = np.logical_and(point1, point2)
            union = np.logical_or(point1, point2)
            similarity = np.sum(intersection) / np.sum(union)

            return similarity

def minkowski_distance(point1, point2, p=2):
                    
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    # point 1 and point 2 are vectors
    distance = pow(sum(pow(abs(p1 - p2), p) for p1, p2 in zip(point1, point2)), 1/p)

    return distance

def mahalanobis_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    # point 1 and point 2 are vectors
    distance = np.sqrt(np.sum(np.square(np.subtract(point1, point2))))


    return distance
 
def combined_metric(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    # point 1 and point 2 are vectors
    distance = np.sqrt(np.sum(np.square(np.subtract(point1, point2)))) + np.sum(np.logical_and(point1, point2)) / np.sum(np.logical_or(point1, point2))

    return distance