
import math

import numpy as np


def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    sum_of_squares = sum(pow(p1 - p2, 2) for p1, p2 in zip(point1, point2))
    return math.sqrt(sum_of_squares)
    # return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def manhattan_distance(point1, point2):
    
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    
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
        
        dot_product = sum(p1 * p2 for p1, p2 in zip(point1, point2))
        magnitude = math.sqrt(sum(p ** 2 for p in point1)) * math.sqrt(sum(p ** 2 for p in point2))
        similarity = dot_product / magnitude
        
        return similarity

def jaccard_similarity(point1, point2):
            
            if len(point1) != len(point2):
                raise ValueError("Points must have the same number of dimensions")
            
            intersection = 0
            union = 0
            for p1, p2 in zip(point1, point2):
                if p1 == p2:
                    intersection += 1
                if p1 != 0 or p2 != 0:
                    union += 1
            similarity = intersection / union
            
            return similarity

def minkowski_distance(point1, point2, p):
                    
                    if len(point1) != len(point2):
                        raise ValueError("Points must have the same number of dimensions")
                    distance = sum(abs(p1 - p2) ** p for p1, p2 in zip(point1, point2)) ** (1 / p)
                    return distance


