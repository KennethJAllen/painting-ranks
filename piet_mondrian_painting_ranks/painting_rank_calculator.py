'''Calculates the rank of Piet Mondrian paintings.'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_image_array(img_filepath: str) -> np.ndarray:
    '''Returns a greyscale numpy array of the image in the file path.'''
    img = Image.open(img_filepath).convert('L')
    return np.array(img)

def compute_svd(array_2d: np.ndarray) -> np.ndarray:
    '''Returns the singular values of a 2d array normalized so that the maximum values is one.'''
    [_, singular_values, _] = np.linalg.svd(array_2d)
    singular_values = singular_values/singular_values[0]
    return singular_values

def get_rank_from_singular_values(singular_values: np.ndarray) -> int:
    '''Returns the rank of the matrix up to a certian threshold.'''
    first_singular_value = singular_values[0]
    if first_singular_value <= 0:
        raise ValueError('The first singular value must be positive.')
    
    rank_threshold = 0.05
    scaled_singular_values = singular_values/singular_values[0]
    for index, singular_value in enumerate(scaled_singular_values):
        if singular_value < rank_threshold:
            return index

def get_image_rank(img_filepath: str) -> int:
    img_array = get_image_array(img_filepath)
    singular_values = compute_svd(img_array)
    rank = get_rank_from_singular_values(singular_values)
    return rank

def plot_singular_values(singular_values: np.ndarray) -> None:
    plt.plot(singular_values)
    plt.ylabel('singular values')
    plt.show()

img_filepath = 'piet_mondrian_painting_ranks/piet_mondrian_paintings/1935 piet mondrian composition in black and white with blue square.jpg'
rank = get_image_rank(img_filepath) # true rank is 3
print(rank)

example_array = np.array([[1, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [1, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [1, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [1, 0, 2, 0, 1],
                          [1, 0, 0, 0, 1],
                          [1, 0, 1, 0, 1]])
print(np.linalg.matrix_rank(example_array))

