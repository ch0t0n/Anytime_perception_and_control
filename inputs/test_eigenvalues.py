import numpy as np
from scipy.linalg import solve_continuous_are

# Define the system matrices A and B
A = np.array([[1, 2],
              [3, 4]])  # Replace with your system's A matrix
B = np.array([[0],
              [1]])     # Replace with your system's B matrix

# We describe some parameters to compute the matrices A and B. A is 12x12 square matrix and B is 12x4 matrix
g = 9.8 # ms^-2
m = 2 # kilograms
lx = 4 # meters
ly = 4 # meters
lz = 4 # meters

A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

B = np.array([[0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [-1/m,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,1/lx,0,0],
            [0,0,1/ly,0],
            [0,0,0,1/lz]])


# Define the desired closed-loop eigenvalues (negative real parts)
desired_eigenvalues = np.array([-1, -2])  # Adjust as needed

# Define the state and control weighting matrices Q and R
Q = np.eye(A.shape[0])  # Identity matrix of appropriate size
R = np.eye(B.shape[1])  # Identity matrix of appropriate size

# Solve the continuous-time algebraic Riccati equation (CARE) for P
P = solve_continuous_are(A, B, Q, R)

# Calculate the feedback gain matrix K
K = np.dot(np.dot(np.linalg.inv(R), B.T), P)

# Print the calculated K matrix
print("Feedback Gain Matrix K:")
print(K)
