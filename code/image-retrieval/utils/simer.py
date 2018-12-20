import numpy as np
from scipy.spatial import distance


class Simer:
    # def __int__(self, name='dot'):
    #     self.name = name
    
    # calculate dot similarity between vector and matrix
    def calculate(self, vector, matrix):
        A = np.mat(vector)
        B = np.mat(matrix)
        result =  A * B.T
        return result.tolist()[0]



class CosineSimer(Simer):
    # calculate cosine similarity between vector and matrix
    def calculate(self, vector, matrix):
        A = np.array([vector])
        B = np.array(matrix)
        result = 1 - distance.cdist(A,B,'cosine')
        return result.tolist()[0]


class CosineSimer_vector(Simer):
    # calculate cosine similarity between vector and vector
    def calculate(self, vector_1, vector_2):
        A = np.array([vector_1])
        B = np.array([vector_2])
        result = 1 - distance.cdist(A,B,'cosine')
        return result.tolist()[0][0]



class CosineSimer_batch(Simer):
    # calculate cosine similarity between matrix and matrix
    def calculate(self, matrix_a, matrix_b):
        A = np.array(matrix_a)
        B = np.array(matrix_b)
        result = 1 - distance.cdist(A,B,'cosine')
        return result.tolist()


class InverseEucSimer(Simer):
    # calculate inverse Euclidean distance between vector and matrix
    def calculate(self, vector, matrix):
        A = np.array([vector])
        B = np.array(matrix)
        result = 1. / distance.cdist(A,B,'euclidean')
        return result.tolist()[0]




NAME_TO_ENCODER = {'dot': Simer, 'cosine_batch': CosineSimer_batch, 'cosine_vector': CosineSimer_vector,'cosine': CosineSimer, 'inverseEuc': InverseEucSimer}

def get_simer(name):
    return NAME_TO_ENCODER[name]



if __name__ == "__main__":
    simmer = get_simer('cosine_vector')()
    a = [1,2,3]
    b = [1,2,3]
    print simmer.calculate(a, b)
