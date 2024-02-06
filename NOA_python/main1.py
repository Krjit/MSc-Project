import numpy as np
from NOA import NOA  # assuming the NOA function is defined in a separate file
from Get_Functions_details import Get_Functions_details  

SearchAgents_no = 30  # Number of search agents
Function_name = 23 # Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)
Max_iteration = 500  # Maximum numbef of iterations

# Load details of the selected benchmark function
lb, ub, dim, fobj = Get_Functions_details(Function_name)

Best_score, Best_pos, Convergence_curve, t = NOA(SearchAgents_no, Max_iteration, ub, lb, dim, fobj)

print('The best solution obtained by NOA is: ', Best_pos)
print('The best optimal value of the objective function found by NOA is: ', Best_score)

# Best convergence curve
import matplotlib.pyplot as plt

plt.semilogy(Convergence_curve, color='r')
plt.title('Objective space')
plt.xlabel('Iteration')
plt.ylabel('Best score obtained so far')
plt.axis('tight')
plt.grid(True)
plt.box(True)
plt.legend(['NOA'])
plt.show()