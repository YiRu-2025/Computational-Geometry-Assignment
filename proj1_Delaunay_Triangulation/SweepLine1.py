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

# PREDICATE --------------------------------------------------------

#def orient2d (a,b,c) :
#    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])     

def myprint(*args, **kwargs) :
#    print(args)
    return 

# INTERSECTION --------------------------------------------------------

def intersect (a,b,c,d) :
    abc = orient2d(a,b,c)
    bad = orient2d(b,a,d)
    if abc*bad <= 0 :
        return False, []
    cdb = orient2d(c,d,b)
    dca = orient2d(d,c,a)
    if cdb*dca <= 0 :
        return False, []
    t = abc/(abc+bad)
#    print(t,c,b,(1-t)*c + t*b)
    return True, (1-t)*c + t*d

def intersections_naive (n,lines) :
    for i in range (n) :
        for j in range (i,n) :
           ai = lines[i,0,:]
           bi = lines[i,1,:]
           aj = lines[j,0,:]
           bj = lines[j,1,:]
           answer, result = intersect (ai,bi,aj,bj)
           if answer == True :
               listOfIntersections[(min(i,j),max(i,j))] = result

# GRAPHICS --------------------------------------------------------

def draw_lines (lines, lc, axs) :     
    axs.cla()
    axs.add_collection(lc)
    axs.plot(lines[:,0,0],lines[:,0,1],'o',color='blue')
    axs.plot(lines[:,1,0],lines[:,1,1],'o',color='blue')
    font1 = {'family':'serif','color':'blue','size':20}
    plt.title("Bentley–Ottmann Algorithm",fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)

def drawIntersections ( ) :
    global listOfIntersections
    global lc, lines,axs,fig, n
    for p,x in listOfIntersections.items() :
        axs.plot(x[0],x[1],'o',color='red', markersize='10')

def drawSweepLine (ev) :        
    global lines,axs,fig

    axs.plot(ev[0][0],ev[0][1],'o',color='black', markersize=10)
    axs.plot([ev[0][0],ev[0][0]],[0,1],color='black')
    for i  in range(len(sweepLine)) :
        line = sweepLine[i]
        ai = lines[line,0,:]
        bi = lines[line,1,:]
        axs.plot([ai[0],bi[0]],[ai[1],bi[1]],color='green')
        axs.text((ai[0]+bi[0])/2.0, (ai[1]+bi[1])/2.0, "("+str(line)+","+str(i)+")", fontsize=15)

    axs.text(ev[0][0]+.01, 1.01 , "Sweep Line", fontsize=15)


# EVENTS -------------------------------------------------------

def init_events (n, lines) : 
    for i in range(n) :
        ai = lines[i,0,:]
        bi = lines[i,1,:]
        heapq.heappush(eventQueue,((ai[0],ai[1]),1,i,-1)) # 1 start, line, intersectline
        heapq.heappush(eventQueue,((bi[0],bi[1]),2,i,-1)) # 2 end, line, intersectline

def insertLineInSweepLine (newLine) :
    global lines

    aNew = lines[newLine,0,:]

    if (len(sweepLine)==0) :
        sweepLine.append(newLine)
        return 0

    for i in range(len(sweepLine)) :
        currentLine = sweepLine[i]
        bNew = lines[currentLine,0,:]
        cNew = lines[currentLine,1,:]
        area = orient2d(aNew, bNew, cNew)
        if (area < 0.0) : 
            sweepLine.insert(i,newLine)            
            return  i
        
    sweepLine.append(newLine)
    return len(sweepLine) -1

def deleteFromSweepLine (toDelete) :
    for i in range(len(sweepLine)) :
        currentLine = sweepLine[i]
        if currentLine == toDelete :
            sweepLine.pop(i)
            return i
    return -1

def swapInSweepLine (a,b) :
    i1 = -1
    i2 = -1
    for i in range(len(sweepLine)) :
        currentLine = sweepLine[i]
        if currentLine == a :
            i1 = i
        if currentLine == b :
            i2 = i
    if i1 >=0 and i2 >= 0 :
        sweepLine[i1] = b
        sweepLine[i2] = a
    return min(i1,i2),max(i1,i2)
            

def checkForIntersectionInSweepLine (i,j) :
    global lines, listOfIntersections,sweepLineCoordinate

    if (min(i,j),max(i,j)) in listOfIntersections :
#        print(i,j," is there");
        return False
    
    ai = lines[i,0,:]
    bi = lines[i,1,:]
    aj = lines[j,0,:]
    bj = lines[j,1,:]
    answer, result = intersect (ai,bi,aj,bj)
    if answer == True and result[0] > sweepLineCoordinate:
        listOfIntersections[(min(i,j), max(i,j))] = result
        heapq.heappush(eventQueue,((result[0],result[1]),3,i,j)) # 3 intersection, line, intersectline
        return True
    return False
#    print (    listOfIntersections)

def removeCrossingPoint (i,j,xsl) :
    if (min(i,j),max(i,j)) in listOfIntersections :
        x = listOfIntersections[(min(i,j),max(i,j))]
        if (x[0] > xsl) :
            listOfIntersections.pop((min(i,j),max(i,j)))
            return True
    return False
        
def treatEvent ( ev ) :
    global     listOfIntersections;
    global     prev0,prev1
    myprint("sweep line",sweepLine)
    if ev[1] == 1 :
        position = insertLineInSweepLine (ev[2])
        myprint("  ev left of line",ev[2]," -- insert line at position ",position)
        myprint("  new sweep line",sweepLine)
        if (position < len(sweepLine) - 1) :
            res = checkForIntersectionInSweepLine ( sweepLine[position], sweepLine[position + 1] )
            myprint("   intersection ", sweepLine[position], sweepLine[position + 1], res)
        if (position > 0) :
            res = checkForIntersectionInSweepLine ( sweepLine[position], sweepLine[position - 1] )
            myprint("   intersection ", sweepLine[position], sweepLine[position - 1], res)
        if position < len(sweepLine) - 1 and position > 0 :
            res = removeCrossingPoint (sweepLine[position+1], sweepLine[position-1], ev[0][0])
            myprint("   removing cr pt ", sweepLine[position+1], sweepLine[position - 1], res)
    elif ev[1] == 2 :
        position = deleteFromSweepLine (ev[2])
        myprint("  ev right of line",ev[2]," -- removing line at position ",position)
        myprint("  new sweep line",sweepLine)
        if position != len(sweepLine) :
            res = checkForIntersectionInSweepLine ( sweepLine[position], sweepLine[position - 1] )
            myprint("   intersection ", sweepLine[position], sweepLine[position - 1], res)
    elif ev[1] == 3:
        if prev0 !=min(ev[2],ev[3]) or prev1 != max(ev[2],ev[3]) :
            prev0 = min(ev[2],ev[3])
            prev1 = max(ev[2],ev[3])
            low, up = swapInSweepLine (ev[2],ev[3])
            myprint("  ev swap lines",ev[2],ev[3]," -- swap lines at position ",low, up)
            myprint("  new sweep line",sweepLine)
            myprint("  ",listOfIntersections)
            if low > 0 :
                res = checkForIntersectionInSweepLine (sweepLine[low], sweepLine[low - 1] )            
                myprint("   intersection ", sweepLine[low], sweepLine[low - 1], res)
                res = removeCrossingPoint (sweepLine[up], sweepLine[low - 1], ev[0][0] )            
                myprint("   removing cr pt ", sweepLine[up], sweepLine[low - 1], res)
            if up < len(sweepLine) - 1 :
                res=checkForIntersectionInSweepLine ( sweepLine[up], sweepLine[up + 1] )
                myprint("   intersection ", sweepLine[up], sweepLine[up+1], res)
                res = removeCrossingPoint (sweepLine[low], sweepLine[up + 1], ev[0][0] )            
                myprint("   removing cr pt ", sweepLine[low], sweepLine[up + 1], res)
    
def onKey(event):
    global lc, lines,axs,fig, n

    if event.key == 'r':
        draw_lines(lines, lc, axs)
        init_events (n, lines)  
        fig.canvas.draw()
    elif event.key == 'n':
        draw_lines(lines, lc, axs)
        res,inds = intersections_naive (n,lines)    
        axs.plot(np.array(res)[:,0],np.array(res)[:,1],'o',color='red')
        fig.canvas.draw()
    elif event.key == 'a':
        while True :
            if len(eventQueue) == 0 :
                break
            ev = heapq.heappop(eventQueue)
            treatEvent(ev)
        draw_lines(lines, lc, axs)
        drawIntersections ( ) 
        fig.canvas.draw()
        
    elif event.key == 'e' :
        draw_lines(lines, lc, axs)
        if len(eventQueue) == 0 :
            init_events (n, lines)  
            drawIntersections ( ) 
            fig.canvas.draw()
            return        
        ev = heapq.heappop(eventQueue)

        treatEvent(ev)
        drawIntersections ( ) 
        drawSweepLine ( ev) 
        fig.canvas.draw()
    elif event.key == 'q' :
        exit(1)
            
## MAIN function

if __name__ == '__main__':

    eventQueue = []
    sweepLine = []
    listOfIntersections = {}
    sweepLineCoordinate = -1.0
    argumentList = sys.argv[1:]
    prev0 = -1
    prev1 = -1
    # Options
    options = "hn:d:xa"
    # Long options
    long_options = ["Help", "NumPoints=", "Smallness=", "Execute", "Naive"]
    arguments, values = getopt.getopt(argumentList, options, long_options)
    
    n = 10
    d = 3.0
    executeOnly = False
    naiveAlgorithm = False
    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-h", "--Help"):
            print ("Usage : python "+sys.argv[0]+" [-n NumPoints] [-s Smallness] [-x] [-h]")
            print ("Press e for running the algorithm step by step")
            print ("Press a for running the algorithm all at once")
            exit(0)
        elif currentArgument in ("-n", "--NumPoints"):
            n = int(currentValue)
        elif currentArgument in ("-s", "--Smallness"):
            d = float(currentValue)
        elif currentArgument in ("-x", "--Execute"):
            executeOnly = True
        elif currentArgument in ("-a", "--Execute"):
            naiveAlgorithm = True
    
    seed(10)

    if executeOnly == False :
        W, H = 10, 10  # figure aspect
        fig, axs = plt.subplots(figsize=(W, H))

    alines = []
    dx = 1./d
    for i in range(n) :
#        r = random()*dx
        p1 = (random(),random())
        p2 = (random(),random())
#        c = 2*(q1+q2);
#        p1 = [c[0] + 0.5*r*(q2[0]-q1[0]),c[1] + r*0.5*(q2[1]-q1[1])]
#        p2 = [c[0] -r*0.5*(q2[0]-q1[0]),c[1] - r*0.5*(q2[1]-q1[1])]
        if p1 < p2 :
            alines.append([p1,p2])
        else :
            alines.append([p2,p1])
    
    lines = np.array(alines,dtype='d')
    init_events (n, lines)      
    if executeOnly == False :
        lc = mc.LineCollection(lines, color='C1')    
        draw_lines(lines, lc, axs)
        fig.canvas.mpl_connect('key_press_event', onKey)
        plt.show()
    else :
 
        start = time()
        if naiveAlgorithm :
            intersections_naive (n,lines) 
        else :
            while True :
                if len(eventQueue) == 0 :
                    break
                ev = heapq.heappop(eventQueue)
                treatEvent(ev)
        end = time() 
        print("Time elapsed during the calculation:", end - start," ", len(listOfIntersections)," intersections computed")    
