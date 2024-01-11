from levy import levy
import numpy as np
import random
from initialization import initialization

# The Nutcracker Optimization Algorithm
def NOA(SearchAgents_no, Max_iter, ub, lb, dim, fobj):

    # Definitions
    Best_NC = np.zeros(dim)  # A vector to include the best-so-far Nutcracker(Solution) 
    Best_score = float('inf')  # A Scalar variable to include the best-so-far score
    LFit = float('inf') * np.ones(SearchAgents_no)  # A vector to include the local-best position for each Nutcracker
    RP = np.zeros((2, dim))  #  2-D matrix to include two reference points of each Nutcracker 
    Convergence_curve = np.zeros(Max_iter)

    # Controlling parameters
    Alpha = 0.05  # The percent of attempts at avoiding local optima
    Pa2 = 0.2  # The probability of exchanging between the cache-search stage and the recovery stage
    Prb = 0.2  # The percentage of exploration other regions within the search space.

    # Initialization
    Positions = initialization(SearchAgents_no, dim, ub, lb)  # Initialize the positions of search agents
    Lbest = Positions  # Set the local best for each Nutcracker as its current position at the beginning.
    t = 1  # Function evaluation counter 

    NC_Fit = [0 for i in range(SearchAgents_no)]
    # Evaluation
    for i in range(SearchAgents_no):
        NC_Fit[i] = fobj(Positions[i])
        LFit[i] = NC_Fit[i]  # Set the local best score for the ith Nutcracker as its current score.

        # Update the best-so-far solution
        if NC_Fit[i] < Best_score:  # Change this to > for maximization problem
            Best_score = NC_Fit[i]  # Update the best-so-far score
            Best_NC = Positions[i]  # Update the best-so-far solution


    while t < Max_iter:
        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)  # Levy random number vector
        l = np.random.rand() * (1 - t / Max_iter)  # Parameter in Eq. (3)
        # Parameter in Eq. (11)
        if np.random.rand() < np.random.rand():
            a = (t / Max_iter) ** (2 * 1 / t)
        else:
            a = (1 - (t / Max_iter)) ** (2 * (t / Max_iter))
        
        ## Foraging and storage strategy
        if np.random.rand() < np.random.rand():  
            mo = np.mean(Positions, axis=0)
            for i in range(SearchAgents_no):
                # Update the parameter mu according to Eq. (2)
                if np.random.rand() < np.random.rand():
                    mu = np.random.rand()
                elif np.random.rand() < np.random.rand():
                    mu = np.random.randn()
                else:
                    mu = RL[0, 0]
                    
                cv = np.random.randint(SearchAgents_no)  # An index selected randomly between 1 and SearchAgents_no
                cv1 = np.random.randint(SearchAgents_no)  # An index selected randomly between 1 and SearchAgents_no
                Pa1 = ((Max_iter - t) / Max_iter)

                if np.random.rand() > Pa1:  # Exploration phase 1: Foreging stage
                    cv2 = np.random.randint(SearchAgents_no)
                    r2 = np.random.rand()

                    for j in range(Positions.shape[1]):
                        if t <= Max_iter / 2:
                            if np.random.rand() > np.random.rand():
                                Positions[i, j] = (mo[j]) + RL[i, j] * \
                                    (Positions[cv, j] - Positions[cv1, j]) + mu * \
                                    (np.random.rand() < 0.5) * \
                                    (r2 * r2 * ub[j] - lb[j])  # Eq. (1)
                        else:
                            if np.random.rand() > np.random.rand():
                                Positions[i, j] = Positions[cv2, j] + mu * \
                                    (Positions[cv, j] - Positions[cv1, j]) + mu * \
                                    (np.random.rand() < Alpha) * \
                                    (r2 * r2 * ub[j] - lb[j])  # Eq. (1)

                else:  # Exploitation phase 1: Storage stage
                    mu = np.random.rand()

                    if np.random.rand() < np.random.rand():
                        r1 = np.random.rand()
                        for j in range(Positions.shape[1]):
                            Positions[i, j] = (Positions[i, j]) + mu * abs(RL[i, j]) * \
                                (Best_NC[j] - Positions[i, j]) + (r1) * \
                                (Positions[cv, j] - Positions[cv1, j])  # Eq. (3)

                    elif np.random.rand() < np.random.rand():
                        for j in range(Positions.shape[1]):
                            if np.random.rand() > np.random.rand():
                                Positions[i, j] = Best_NC[j] + \
                                    mu * (Positions[cv, j] - Positions[cv1, j])  # Eq. (3)

                    else:
                        for j in range(Positions.shape[1]):
                            Positions[i, j] = (Best_NC[j] * abs(l))  # Eq. (3)

                           
                # Return the search agents that exceed the search space's bounds
                if np.random.rand() < np.random.rand():
                    for j in range(Positions.shape[1]):
                        if Positions[i,j] > ub[j]:
                            Positions[i,j] = lb[j] + np.random.rand()  * (ub[j] - lb[j])
                        elif Positions[i,j] < lb[j]:
                            Positions[i,j] = lb[j] + np.random.rand()  * (ub[j] - lb[j])
                else:
                    Positions[i,:] = np.minimum(np.maximum(Positions[i,:], lb), ub)
                    
                # Evaluation
                NC_Fit[i] = fobj(Positions[i,:])
                
                # Update the local best according to Eq. (20)
                if NC_Fit[i] < LFit[i]: # Change this to > for maximization problem
                    LFit[i] = NC_Fit[i] # Update the local best fitness
                    Lbest[i,:] = Positions[i,:] # Update the local best position
                else:
                    NC_Fit[i] = LFit[i]
                    Positions[i,:] = Lbest[i,:]
                    
                # Update the best-so-far solution
                if NC_Fit[i] < Best_score: # Change this to > for maximization problem
                    Best_score = NC_Fit[i] # Update best-so-far fitness
                    Best_NC = Positions[i,:] # Update best-so-far position
                    
                t += 1
                if t > Max_iter:
                    break
                    
                Convergence_curve[t-1] = Best_score
        
        ## Cache-search and Recovery strategy
        else:
            # Compute the reference points for each Nutcraker
            for i in range(SearchAgents_no):
                ang = np.pi * np.random.rand()
                cv = np.random.randint(SearchAgents_no)
                cv1 = np.random.randint(SearchAgents_no)
                
                for j in range(Positions.shape[1]):
                    for j1 in range(2):
                        if j1 == 0:
                            # Compute the first reference point for the ith Nutcraker using Eq. (9)
                            if ang != np.pi / 2:
                                RP[j1, j] = Positions[i, j] + (a * np.cos(ang) * (Positions[cv, j] - Positions[cv1, j]))
                            else:
                                RP[j1, j] = Positions[i, j] + a * np.cos(ang) * (Positions[cv, j] - Positions[cv1, j]) + a * RP[np.random.randint(2), j]
                        else:
                            # Compute the second reference point for the ith Nutcraker using Eq. (10)
                            if ang != np.pi / 2:
                                RP[j1, j] = Positions[i, j] + (a * np.cos(ang) * ((ub[j] - lb[j]) + lb[j])) * (np.random.rand() < Prb)
                            else:
                                RP[j1, j] = Positions[i, j] + (a * np.cos(ang) * ((ub[j] - lb[j]) * np.random.rand() + lb[j]) + a * RP[np.random.randint(2), j]) * (np.random.rand() < Prb)

                    '''Positions, RP, lb, ub are assumed to be numpy arrays or lists
                    # Positions is a 2D array containing the positions of all reference points
                    # RP is a 1D array containing the position of the reference point being evaluated
                    # lb and ub are 1D arrays containing the lower and upper bounds of the search space
                    # rand is a scalar value
                    '''
                    # Return the reference points that exceed the boundary of search space
                    if np.random.rand() < np.random.rand():
                        for j in range(Positions.shape[1]):
                            if RP[1,j] > ub[j]:
                                RP[1,j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                            elif RP[1,j] < lb[j]:
                                RP[1,j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                    else:
                        RP[1,:] = np.minimum(np.maximum(RP[1,:], lb), ub)
                    
                    # Return the reference points that exceed the boundary of search space
                    if np.random.rand() < np.random.rand():
                        for j in range(Positions.shape[1]):
                            if RP[0,j] > ub[j]:
                                RP[0,j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                            elif RP[0,j] < lb[j]:
                                RP[0,j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                    else:
                        RP[0,:] = np.minimum(np.maximum(RP[0,:], lb), ub)

                if np.random.rand() > Pa2: # Exploitation phase 2: Recovery stage
                    cv = random.randint(0, SearchAgents_no-1)
                    if np.random.rand() < np.random.rand():
                        for j in range(Positions.shape[1]):
                            if np.random.rand() > np.random.rand():
                                Positions[i,j] = Positions[i,j] + np.random.rand()*(Best_NC[j]-Positions[i,j]) + np.random.rand()*(RP[0,j]-Positions[cv,j]) # Eq. (13)
                    else:
                        for j in range(Positions.shape[1]):
                            if np.random.rand() > np.random.rand():
                                Positions[i,j] = Positions[i,j] + np.random.rand()*(Best_NC[j]-Positions[i,j]) + np.random.rand()*(RP[1,j]-Positions[cv,j]) # Eq. (15)
                    
                    # Return the search agents that exceed the search space's bounds
                    if np.random.rand() < np.random.rand():
                        for j in range(Positions.shape[1]):
                            if Positions[i,j] > ub[j]:
                                Positions[i,j] = lb[j] + np.random.rand()*(ub[j]-lb[j])
                            elif Positions[i,j] < lb[j]:
                                Positions[i,j] = lb[j] + np.random.rand()*(ub[j]-lb[j])
                    else:
                        Positions[i,:] = np.minimum(np.maximum(Positions[i,:], lb), ub)

                    # Evaluation
                    NC_Fit[i] = fobj(Positions[i,:])

                    # Update the local best
                    if NC_Fit[i] < LFit[i]:  # Change this to > for maximization problem
                        LFit[i] = NC_Fit[i]
                        Lbest[i,:] = Positions[i,:]
                    else:
                        NC_Fit[i] = LFit[i]
                        Positions[i,:] = Lbest[i,:]

                    # Update the best-so-far solution
                    if NC_Fit[i] < Best_score:  # Change this to > for maximization problem
                        Best_score = NC_Fit[i] # Update best-so-far fitness
                        Best_NC = Positions[i,:] # Update best-so-far position

                    t += 1
                    if t > Max_iter:
                        break
                
                else: # Exploration stage 2: Cache-search stage
                    NC_Fit1 = fobj(RP[0,:])
                    # t = t + 1
                    # if t > Max_iter:
                    #     break
                    
                    # Evaluations
                    NC_Fit2 = fobj(RP[1,:])

                    # Apply Eq. (17) to trade-off between the exploration behaviors
                    if NC_Fit2 < NC_Fit1 and NC_Fit2 < NC_Fit[i]:
                        Positions[i,:] = RP[1,:]
                        NC_Fit[i] = NC_Fit2
                    elif NC_Fit1 < NC_Fit2 and NC_Fit1 < NC_Fit[i]:
                        Positions[i,:] = RP[0,:]
                        NC_Fit[i] = NC_Fit1
                    
                    # Update the local best
                    if NC_Fit[i] < LFit[i]: # Change this to > for maximization problem
                        LFit[i] = NC_Fit[i]
                        Lbest[i,:] = Positions[i,:]
                    else:
                        NC_Fit[i] = LFit[i]
                        Positions[i,:] = Lbest[i,:]

                    # t = t + 1
                    
                    # Update the best-so-far solution
                    if NC_Fit[i] < Best_score: # Change this to > for maximization problem
                        Best_score = NC_Fit[i]
                        Best_NC = Positions[i,:]
                    
                    t += 1
                    if t > Max_iter:
                        break
    
    return Best_score, Best_NC, Convergence_curve, t