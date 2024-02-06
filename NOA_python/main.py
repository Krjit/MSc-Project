# Nutcracker Optimization Algorithm (NOA) source codes demo 1.0

# Developed in MATLAB R2019A

# Author and programmer: Reda Mohamed (E-mail: redamoh@zu.edu.eg) & Mohamed Abdel-Basset (E-mail: mohamedbasset@ieee.org)

# Main paper: Abdel-Basset, M., Mohamed, R.
#             Nutcracker optimizer,
#             Knowledge-Based Systems, in press,
#             DOI: https://doi.org/10.1016/j.knosys.2022.110248


import numpy as np
from Get_Functions_details import Get_Functions_details
from NOA import NOA


SearchAgents_no = 25 # Number of search agents
Max_iteration = 50#000 # Maximum number of Function evaluations
RUN_NO = 30 ## Number of independent runs

for i in range(1, 24): ## Test functions
    for j in range(1, RUN_NO+1):
        lb, ub, dim, fobj = Get_Functions_details(i)
        Best_score, Best_pos, Convergence_curve, t= NOA(SearchAgents_no, Max_iteration, ub, lb, dim, fobj)
        fitness = Best_score
    print('Function_ID\t', i, '\tAverage Fitness:', np.mean(fitness))
    print(fitness)

    ## Drawing Convergence Curve ##
    import matplotlib.pyplot as plt
    plt.figure(i)
    plt.semilogy(Convergence_curve)
    # h = plt.semilogy(Convergence_curve='-<', MarkerSize= 8, LineWidth=2, Color='r')
    # h.MarkerIndices = range(999, Max_iteration, 1000)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness obtained so-far')
    plt.axis('tight')
    plt.grid(False)
    plt.box(True)
    plt.legend({'NOA'})

plt.show()
