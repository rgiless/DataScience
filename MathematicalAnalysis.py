import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def q1():
    # Define matrix A
    A = np.array([[-2, 1, 0, 4],[2, 0, -1, 0],[2, -1, 0, 3],[0, 1, 1, -3]])
    # Define matrix b 
    b = np.array([2, 0, -1, 1])
    # Solve system using numpy linear algebra function
    print(f"Solution: {np.linalg.solve(A,b)}")
    x = np.linalg.inv(A) @ b
    # Manually check numpy solution
    # np.isclose is used due to equating of integer and float values not being
    # exactly zero
    print(f"\nCheck if A@x = b: {np.isclose(A@x, b)}")

def q2():
    # Define system of equations
    def func(params):
        x, y = params
        eq1 = -(x**3) + 2*x - (y**2) + 2 
        eq2 = -1.5*(x**2) - 0.5*y + 1
        # Function returns value of both equations with inputted parameters
        return [eq1, eq2]
    
    # Two roots of system are found using scipy.optimize fsolve function with
    # initial guesses
    root1 = fsolve(func, [-5, 5])
    root2 = fsolve(func, [-1.5, -1])
    
    # Roots are printed and equations are checked to be 0 when roots are inputted
    # to confirm accuracy of solutions
    print(f'\nRoot 1 in form [x y] = {root1}')
    print(f'\nRoot 2 in form [x y] = {root2}')
    print(f'\nCheck if functions = 0 at first solution: {np.isclose(func(root1), [0, 0])}')
    print(f'\nCheck if functions = 0 at second solution: {np.isclose(func(root2), [0, 0])}')
    
    # Both equtions plotted with roots denoted by dots on the graph
    x_vals = np.linspace(-5, 5, 400) 
    y_vals = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    F1 = -(X**3) + 2*X - (Y**2) + 2
    F2 = -1.5*(X**2) - 0.5*Y + 1
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F1, levels=[0], colors='r')
    plt.contour(X, Y, F2, levels=[0], colors='b')
    plt.plot([], [], 'r', label="$-x^3 + 2x - y^2 + 2 = 0$")
    plt.plot([], [], 'b', label="$-1.5x^2 - 0.5y + 1 = 0$")
    plt.scatter([root1[0]], [root1[1]], color='black', marker='x', label="Root 1")
    plt.scatter([root2[0]], [root2[1]], color='black', marker='o', label="Root 2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def q3(a, r, tol, iteration):
    # Geometric sum is set to 0
    gsum = 0
    # Theoretical convergence limit defined
    theolimit = a/(1-r)
    # for loop created to iterate through geometric series
    for i in range(iteration):
        # if series converges within given tolerance, loop ends
        if abs(gsum - theolimit) < tol:
            print(f'Series converged on iteration {i} with difference {abs(gsum - theolimit)}')
            break
        # if series has not converged yet, keep looping and update geometric series
        else:
            gsum += a*(r**i)
    # if no convergence found in given iterations, denoted by i reaching maximum iteration number,
    # series has diverged
    if i == iteration - 1:
        print(f'Series not converged within {i+1} iterations, with final difference of {abs(gsum - theolimit)}')

print('Question 1: \n')
q1()
print('\nQuestion 2: \n')
q2()
print('\nQuestion 3: \n')
# Question 3 tested with given parameters
tolerances = [1e-7, 1e-12, 1e-20, 1e-7, 1e-7]
for tol in tolerances:
    print(f"\nTesting with tol = {tol}")
    q3(4, 0.9, tol, 10000)
    
            
