#_________________________________________________________________________#
#  Nutcracker Optimization Algorithm (NOA) source codes demo 1.0               #
#                                                                         #
#  Developed in MATLAB R2019A                                      #
#                                                                         #
#  Author and programmer: Reda Mohamed (E-mail: redamoh@zu.edu.eg) & Mohamed Abdel-Basset (E-mail: mohamedbasset@ieee.org)                              #
#                                                                         #
#   Main paper: Abdel-Basset, M., Mohamed, R.                                    #
#               Nutcracker optimizer,                         #
#               Knowledge-Based Systems, in press,              #
#               DOI: https://doi.org/10.1016/j.knosys.2022.110248   #
#                                                                         #
#_________________________________________________________________________#

# This function initialize the first population of search agents
import numpy as np
def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub)  # numnber of boundaries

    # If the boundaries of all variables are equal and user enter a signle
    # number for both ub and lb
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb

    # If each variable has a different lb and ub
    if Boundary_no > 1:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions
