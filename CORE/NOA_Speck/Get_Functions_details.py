#_________________________________________________________________________%

"""Nutcracker Optimization Algorithm (NOA) source codes demo 1.0 %
%
Developed in MATLAB R2019A %
%
Author and programmer: Reda Mohamed (E-mail: redamoh@zu.edu.eg) & Mohamed Abdel-Basset (E-mail: mohamedbasset@ieee.org) %
%
Main paper: Abdel-Basset, M., Mohamed, R. %
Nutcracker optimizer, %
Knowledge-Based Systems, in press, %
DOI: https://doi.org/10.1016/j.knosys.2022.110248 %
%"""
#_________________________________________________________________________%
"""
lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]
dim is the number of variables (dimension of the problem)
"""
import numpy as np
def Get_Functions_details(F):
    if F == 1:
        fobj = F1
        dim = 100
        lb = [-100] * dim
        ub = [100] * dim


    elif F == 2:
        fobj = F2
        dim = 100
        lb = [-10] * dim
        ub = [10] * dim
        
    elif F == 3:
        fobj = F3
        dim = 100
        lb = [-100] * dim
        ub = [100] * dim
        
    elif F == 4:
        fobj = F4
        dim = 100
        lb = [-100] * dim
        ub = [100] * dim
        
    elif F == 5:
        fobj = F5
        dim = 100
        lb = [-30] * dim
        ub = [30] * dim
        
    elif F == 6:
        fobj = F6
        dim = 100
        lb = [-100] * dim
        ub = [100] * dim
        
    elif F == 7:
        fobj = F7
        dim = 100
        lb = [-1.28] * dim
        ub = [1.28] * dim
        
    elif F == 8:
        fobj = F8
        dim = 100
        lb = [-500] * dim
        ub = [500] * dim
        
    elif F == 9:
        fobj = F9
        dim = 100
        lb = [-5.12] * dim
        ub = [5.12] * dim
        
    elif F == 10:
        fobj = F10
        dim = 100
        lb = [-32] * dim
        ub = [32] * dim
        
    elif F == 11:
        fobj = F11
        dim = 100
        lb = [-600] * dim
        ub = [600] * dim
    
    elif F == 12:
        fobj = F12
        dim = 100
        lb = [-50] * dim
        ub = [50] * dim
        
    elif F == 13:
        fobj = F13
        dim = 100
        lb = [-50] * dim
        ub = [50] * dim
        
    elif F == 14:
        fobj = F14
        dim = 2
        lb = [-65.536, -65.536]
        ub = [65.536, 65.536]
        
    elif F == 15:
        fobj = F15
        dim = 4
        lb = [-5] * dim
        ub = [5] * dim
        
    elif F == 16:
        fobj = F16
        dim = 2
        lb = [-5, -5]
        ub = [5, 5]
        
    elif F == 17:
        fobj = F17
        dim = 2
        lb = [-5, 0]
        ub = [10, 15]
        
    elif F == 18:
        fobj = F18
        dim = 2
        lb = [-2, -2]
        ub = [2, 2]
        
    elif F == 19:
        fobj = F19
        dim = 3
        lb = [0, 0, 0]
        ub = [1, 1, 1]
        
    elif F == 20:
        fobj = F20
        dim = 6
        lb = np.zeros(dim)
        ub = np.ones(dim)
    
    elif F == 21:
        fobj = F21
        dim = 4
        lb = np.zeros(dim)
        ub = 10 * np.ones(dim)
    elif F == 22:
        fobj = F22
        dim = 4
        lb = np.zeros(dim)
        ub = 10 * np.ones(dim)
    elif F == 23:
        fobj = F23
        dim = 4
        lb = np.zeros(dim)
        ub = 10 * np.ones(dim)

    return lb, ub, dim, fobj

# F1
def F1(x):
    return sum([i**2 for i in x])

# F2
def F2(x):
    return sum([abs(i) for i in x]) + np.prod([abs(i) for i in x])

# F3
def F3(x):
    dim = x.shape[0]
    o = 0
    for i in range(dim):
        o += sum(x[:i+1])**2
    return o

# F4
def F4(x):
    return max([abs(i) for i in x])

# F5
def F5(x):
    dim = x.shape[0]
    return sum(100*(x[1:dim]-(x[:dim-1]**2))**2+(x[:dim-1]-1)**2)

# F6
def F6(x):
    return np.sum(np.abs((x + 0.5))**2)

# F7
def F7(x):
    dim = x.shape[0] # np.size(x, axis=1)
    return np.sum(np.arange(1, dim+1) * (x**4)) + np.random.rand()

# F8
def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

# F9
def F9(x):
    dim = x.shape[0] # np.size(x, axis=1)
    return np.sum(x**2 - 10*np.cos(2*np.pi*x)) + 10*dim

# F10
def F10(x):
    dim = x.shape[0] # np.size(x, axis=1)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) - np.exp(np.sum(np.cos(2*np.pi*x))/dim) + 20 + np.exp(1)

# F11
def F11(x):
    dim = x.shape[0] # np.size(x, axis=1)
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, dim+1)))) + 1

# F12
def Ufun(x, a, k, m):
    return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a) + 0 * (np.abs(x) <= a)

def F12(x):
    dim = x.shape[0] # np.size(x, axis=1)
    return (np.pi/dim) * (10*((np.sin(np.pi*(1+(x[0]+1)/4)))**2) + np.sum((((x[0:dim-1]+1)/4)**2) * (1 + 10*((np.sin(np.pi*(1+(x[1:dim]+1)/4))))**2)) + ((x[dim-1]+1)/4)**2) + np.sum(Ufun(x, 10, 100, 4))

# F13
def F13(x):
    dim = x.shape[0]
    o = 0.1 * ((np.sin(3*np.pi*x[0]))**2 + np.sum((x[0:dim-1]-1)**2 * (1+(np.sin(3*np.pi*x[1:dim]))**2)) + 
               ((x[dim-1]-1)**2) * (1+(np.sin(2*np.pi*x[dim-1]))**2)) + np.sum(Ufun(x, 5, 100, 4))
    return o

# F14
def F14(x):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:,j])**6)
    o = (1/500 + np.sum(1/np.arange(1, 26) + bS))**(-1)
    return o

# F15
def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    bK = 1/bK
    o = np.sum((aK - ((x[0]*(bK**2 + x[1]*bK))/(bK**2 + x[2]*bK + x[3])))**2)
    return o

# F16
def F16(x):
    o = 4*(x[0]**2) - 2.1*(x[0]**4) + (x[0]**6)/3 + x[0]*x[1] - 4*(x[1]**2) + 4*(x[1]**4)
    return o

# F17
def F17(x):
    o = (x[1] - (x[0]**2)*5.1/(4*(np.pi**2)) + 5/np.pi*x[0] - 6)**2 + 10*(1-1/(8*np.pi))*np.cos(x[0]) + 10
    return o

# F18
def F18(x):
    o = (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*(x[0]**2)-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*\
    (30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*(x[0]**2)+48*x[1]-36*x[0]*x[1]+27*(x[1]**2)))
    return o

# F19
def F19(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    o = 0
    for i in range(4):
        o = o - cH[i] * np.exp(-np.sum(aH[i,:] * ((x - pH[i,:])**2)))
    return o

# F20
def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

    o = 0
    for i in range(4):
        o = o - cH[i]*np.exp(-np.sum(aH[i,:]*((x-pH[i,:])**2)))

    return o

# F21
def F21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    o = 0
    for i in range(5):
        o = o - 1/(np.sum((x-aSH[i,:])**2) + cSH[i])

    return o

def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
    o = 0
    for i in range(7):
        o = o - ((x - aSH[i, :]).dot(x - aSH[i, :].T) + cSH[i])**(-1)
    return o

def F23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
    o = 0
    for i in range(10):
        o = o - ((x - aSH[i, :]).dot(x - aSH[i, :].T) + cSH[i])**(-1)
    return o

# def Ufun(x, a, k, m):
#     o = k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a)
#     return o

def F24(x):
    # Beale function.
    # The number of variables n = 2.
    y = (1.5 - x[0]*(1 - x[1]))**2 + (2.25 - x[0]*(1 - x[1]**2))**2 + (2.625 - x[0]*(1 - x[1]**3))**2
    return y

def F25(xx):
    x1 = xx[0]
    x2 = xx[1]
    term1 = (x1 + 2*x2 - 7)**2
    term2 = (2*x1 + x2 - 5)**2
    y = term1 + term2
    return y

def F26(xx):
    # matya
    x1 = xx[0]
    x2 = xx[1]
    term1 = 0.26 * (x1**2 + x2**2)
    term2 = -0.48*x1*x2
    y = term1 + term2
    return y
