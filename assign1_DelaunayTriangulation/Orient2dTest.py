import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import collections  as mc
from random import random
from random import seed
import heapq
from time import time
import math
import getopt, sys
import math
from geompreds import orient2d

def orient2d_naive (a,b,c) :
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])     
#def orient2d (a,b,c) :
#    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])     


if __name__ == '__main__':

    a = (12.0,12.0)    
    b = (24.0,24.0)
    n = 100
    eps = 7./3 - 4./3 -1
#    print(eps)
#    exit(1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

#    for i in range(12) :
#        k = i * eps
#        print(23.5*(11.5-k)-11.5*(23.5-k), 11.5-k, 23.5-k)
#    exit(1)
        
    xs = []
    ys = []
    z1 = []
    z2 = []
    z3 = []
    z4 = []

    colors1 = []
    colors2 = []
    colors3 = []
    colors4 = []
    
    for i in range(n) :
        for j in range(n) :
            xs.append(float(i))
            ys.append(float(j))
            o1=orient2d_naive(a,b,(eps*i+0.5,eps*j+0.5))
            o2=orient2d_naive((eps*i+0.5,eps*j+0.5),a,b)
            o3=orient2d(a,b,(eps*i+0.5,eps*j+0.5))
            o4=orient2d((eps*i+0.5,eps*j+0.5),a,b)            
            if o1 > 0 :
                z1.append(10.0)
                colors1.append('r')
            elif o1 < 0 :
                z1.append(20.0)
                colors1.append('b')
            else :  
                z1.append(30.0)
                colors1.append('g')
            if o2 > 0 :
                z2.append(10.0)
                colors2.append('r')
            elif o2 < 0 :
                z2.append(20.0)
                colors2.append('b')
            else :  
                z2.append(30.0)
                colors2.append('g')
            if o3 > 0 :
                z3.append(10.0)
                colors3.append('r')
            elif o3 < 0 :
                z3.append(20.0)
                colors3.append('b')
            else :  
                z3.append(30.0)
                colors3.append('g')
            if o4 > 0 :
                z4.append(10.0)
                colors4.append('r')
            elif o4 < 0 :
                z4.append(20.0)
                colors4.append('b')
            else :  
                z4.append(30.0)
                colors4.append('g')

    W = 10
    H = 10
    fig, axs = plt.subplots(2,2,figsize=(W, H))

    axs[0][0].set_title("Orient2d Naive Predicate (a,b,c)")
    axs[0][0].set_xlabel("i")
    axs[0][0].set_ylabel("j")
    axs[1][0].set_title("Orient2d Naive Predicate (c,a,b)")
    axs[1][0].set_xlabel("i")
    axs[1][0].set_ylabel("j")
    axs[0][1].set_title("Orient2d Predicate (a,b,c)")
    axs[0][1].set_xlabel("i")
    axs[0][1].set_ylabel("j")
    axs[1][1].set_title("Orient2d Predicate (c,a,b)")
    axs[1][1].set_xlabel("i")
    axs[1][1].set_ylabel("j")
    axs[0][0].scatter(xs, ys, s=z1, c=colors1)
    axs[1][0].scatter(xs, ys, s=z2, c=colors2)
    axs[0][1].scatter(xs, ys, s=z3, c=colors3)
    axs[1][1].scatter(xs, ys, s=z4, c=colors4)

    plt.show()    
    
