import numpy as np



def create_matricization(tensor):
    K = len(tensor)
    I = len(tensor[0])
    J = len(tensor[0][0])
    X_1 = np.reshape(tensor, [I, J*K])
    X_2 = np.reshape(tensor, [J, I*K])
    X_3 = np.reshape(tensor, [K, I*J])
    return X_1, X_2, X_3
