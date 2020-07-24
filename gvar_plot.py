#!/usr/bin/env python3

import gvar as gv
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def ellipses(ax, x, y, sigma=[1], **kwargs):
    defaults = {
        'alpha': 0.2,
    }

    defaults.update(kwargs)

    for a, b in zip(x, y):

        C = gv.evalcov((a,b))
        w, v = np.linalg.eigh(C)
        w = np.sort(w)[::-1]

        if C[0,1] == 0 and C[0,0] >= C[1,1]:
            angle = 0
        elif C[0,1] == 0 and C[0,0] < C[1,1]:
            angle = np.pi/2
        else:
            angle = np.arctan2(w[0]-C[0,0], C[0,1])

        for s in sigma:
            # mpl.patches.Ellipse wants full axes, not semi-{minor,major} axes.
            # Hence the 2*.
            axes = 2*np.sqrt(s*w)
            # Angle is in degrees...
            e = patches.Ellipse([a.mean, b.mean], *axes, angle=angle*180/np.pi, **defaults)
            ax.add_patch(e)

# We might be able to even do better, by picking the turning points of the ellipse.
# In the mean time, continue to just draw ellipses.
band = ellipses

def mean(ax, x, y, **kwargs):
    defaults = {
        'zorder': 1,
    }
    defaults.update(kwargs)

    A = [a.mean for a in x]
    B = [b.mean for b in y]

    ax.plot(A, B, **defaults)

def errorbar(ax, x, y, **kwargs):
    defaults = {
        'zorder': 2,
    }
    defaults.update(kwargs)
    
    A = [a.mean for a in x]
    dA =[a.sdev for a in x]
    B = [b.mean for b in y]
    dB= [b.sdev for b in y]

    ax.errorbar(A, B, xerr = dA, yerr = dB, **defaults)

###
### Some examples
###

def _four_points(rng):
    fix, ax = plt.subplots()
    ax.set_title("Four points")

    point = gv.gvar([1, 0.75], [[0.1**2, 0.001], [0.001, 0.02**2]])

    ellipses(ax, [point[0]], [point[1]], sigma=[1,2,3], color='blue')
    errorbar(ax, [point[0]], [point[1]], marker='o',    color='blue')
    ellipses(ax, [point[1]], [point[0]], sigma=[1,2,3], color='red')
    errorbar(ax, [point[1]], [point[0]], marker='o',    color='red')
    # Invert to get the anticorrelation.
    ellipses(ax, [point[0]], [1/point[0]], sigma=[1,2,3], color='green')
    errorbar(ax, [point[0]], [point[0]], marker='o',    color='green')
    ellipses(ax, [point[1]], [point[1]], sigma=[1,2,3], color='purple')
    errorbar(ax, [point[1]], [point[1]], marker='o',    color='purple')
    ax.set_aspect(1)

def _cosine(rng, time=np.arange(0, 2*np.pi, 0.2)):
    fig, ax = plt.subplots()
    ax.set_title("Cosine")
    
    xy = []

    for t in time:
        # Make up sdevs and covariance
        da = 0.25*rng.uniform()
        db = 0.25*rng.uniform()
        cv = (1-2*rng.uniform()) * da * db
        # Add correlated point to list
        xy+= [ gv.gvar([t, np.cos(t)], [[ da**2, cv], [cv, db**2]]) ]
    
    xy = np.array(xy).transpose()

    ellipses(ax, *xy)
    errorbar(ax, *xy)
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-2, 2)

   

def _main():
    rng = np.random.default_rng(seed=7)

    _four_points(rng)
    _cosine(rng)
    plt.show()



if __name__ == "__main__":
    _main()
