import mdtraj as md
import numpy as np
import os.path

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pylab

maxrep=4
trajpath='/storage/mi/pycon/TTApplications/ALA10TT2/Results/'
for r in range(maxrep):
    print 'Using rep '+str(r)
    if os.path.exists(trajpath+'EigenfunctionFramesCenter'+str(r)+'.xtc')==True:
        print "file exists"
        Traj=md.load(trajpath+'EigenfunctionFramesCenter'+str(r)+'.xtc', top=trajpath+'md_production0_noWater.pdb')
        trajlen=len(Traj)
        Dssp=md.compute_dssp(Traj,simplified=False)
        # Plot as gromacs dssp
        fig1 = plt.figure(1)
        ax=fig1.add_subplot(111)  
        skip=1
        for t in range(trajlen/skip):
            for m in range(1,9):
                if Dssp[skip*t,m]=='H':
                    colore='b'
                elif Dssp[skip*t,m]=='B':
                    colore='k'
                elif Dssp[skip*t,m]=='E':
                    colore='r'
                elif Dssp[skip*t,m]=='G':
                    colore='gray'
                elif Dssp[skip*t,m]=='I':
                    colore='m'
                elif Dssp[skip*t,m]=='T':
                    colore='y'
                elif Dssp[skip*t,m]=='S':
                    colore='g'
                elif Dssp[skip*t,m]=='':
                    colore='w'
                else:
                    colore='w'
                    #plt.scatter(t,m, c=colore, marker="s", markeredgecolor =colore, s=10, markeredgewidth=5)        
                plt.plot(t,m, c=colore, marker="s", markeredgecolor =colore, markeredgewidth=1)
        ax.set_ylim(-1,9)
                #ax.set_xlim(0,7000)
        plt.xlabel('Trajectory')
        plt.ylabel('Sequence') 
        plt.savefig(trajpath+'EigenfunctionFramesCenter'+str(r)+'_'+str(skip)+'.pdf')
        plt.close()
        del Traj