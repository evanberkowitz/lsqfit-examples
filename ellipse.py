#!/usr/bin/env python3

import gvar as gv
import lsqfit as lsf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

import gvar_plot

def banner(message):
    print(f"\n#\n#\t{message}\n#\n")

# We need to make up some data that comes from an ellipse with errors + uncertainties.

def ellipse(a, b, phi=0, x0=0, y0=0):
    def curried(theta):
        return np.array([
                x0 + a*np.cos(theta)*np.cos(phi) - b*np.sin(theta)*np.sin(phi),
                y0 + b*np.sin(theta)*np.cos(phi) + a*np.cos(theta)*np.sin(phi)
               ])
    return curried

a=3
b=1
phi=0.2
x0=10
y0=12

theta = np.arange(-np.pi, +np.pi, 0.005)
e = ellipse(a, b, phi, x0, y0)
exact = e(theta)

# Make up random thetas
SEED=42
POINTS=20
rng = np.random.default_rng(seed=SEED)
sample = np.sort(np.mod(rng.normal(0,100, size=POINTS), 2*np.pi))
(X, Y) = e(sample)

# Offsets
bumpX = 0.15*(2*rng.random(size=POINTS)-1)
bumpY = 0.15*(2*rng.random(size=POINTS)-1)
# Uncertainties
eX = rng.normal(0.15, 0.05, size=POINTS)
eY = rng.normal(0.15, 0.05, size=POINTS)

X = np.array([gv.gvar(x+dx, ex) for x, dx, ex in zip(X, bumpX, eX)])
Y = np.array([gv.gvar(y+dy, ey) for y, dy, ey in zip(Y, bumpY, eY)])

#plt.errorbar([x.mean for x in X], [y.mean for y in Y], xerr = [x.sdev for x in X], yerr = [y.sdev for y in Y],
#        linestyle='none',
#        color='black',
#        )
#plt.plot(*exact, linestyle='none', marker='.')
#plt.title("Exact ellipse and data points")
#plt.xlim(6.5, 13.5)
#plt.ylim(10.5, 13.5)

#print("Looking at that figure, we come up with...")

banner("PRIORS")
priors = {
    'x0': gv.gvar('10(3)'),
    'y0': gv.gvar('12(2)'),
    'phi': gv.gvar('0(0.78)'),
    'log(a-b)':  gv.log(gv.gvar('2(2)')),
    'b':  gv.gvar('2(2)'),
}

for p in priors:
    print(f"{p} = {priors[p]}")

banner("METHOD")

print("\nWe're hoping to find the best-fit ellipse.  We should think of both the X and Y points as input and demand that they satisfy the constraint")
print("\n   [ (x-x0) * cos phi + (y-y0)* sin phi ]^2 / a^2 + [ (x-x0) * sin phi - (y-y0) * cos phi ]^2 / b^2 == 1\n")

print("So... what do we take as the 'data' to fit?  We should rewrite the constraint as a level curve,")
print("\n   g(x, y) = [ (x-x0) * cos phi + (y-y0)* sin phi ]^2 / a^2 + [ (x-x0) * sin phi - (y-y0) * cos phi ]^2 / b^2 - 1\n")
print("and the ellipse is when g(x,y) = 0.")

print("\n\nSo, our dependent data is exactly known---it's 0 for each independent data point.")
print("Therefore, take the approach of https://lsqfit.readthedocs.io/en/latest/overview.html#y-has-no-error-marginalization")
print("We give a very small uncertainty on the constraint:\n")

constraint = gv.gvar(['0(0.00001)'] * POINTS)
print(constraint)


print("\n\nAlso, our independent data---the points---have errors.  So we adopt the approach of https://lsqfit.readthedocs.io/en/latest/overview.html#x-has-errors and put those data into the priors.")
priors['X'] = X
priors['Y'] = Y

print("\n\nNow, our fit is independent-data free (in the sense that we put them in the priors instead.")
print("So we can build a very simple fit function that takes only parameters.")

def ellipse_fit(p):
    dX = p['X'] - p['x0']
    dY = p['Y'] - p['y0']
    c = np.cos(p['phi'])
    s = np.sin(p['phi'])
    asq = (p['a-b']+p['b'])**2
    bsq = p['b']**2
    return (dX * c + dY * s)**2 / asq + (dX * s - dY * c)**2 / bsq - 1

banner("RESULTS")

fit = lsf.nonlinear_fit(data=constraint, prior=priors, fcn=ellipse_fit)
# Store a in the fit, as some functions have an argument called a.
# Is there a way to show this in the formatted fit result?
# Like where a-b is shown, under the line of dashes.
fit.p['a'] = fit.p['a-b'] + fit.p['b']
print(fit)

print(f"              a     {fit.p['a-b']+fit.p['b']}")

banner("VISUALIZATION")

print("Now we would like to visualize the best-fit ellipse.")
print("That isn't so bad, just plug the fit results to get a new ellipse.")

best_fit = ellipse(fit.p['a-b']+fit.p['b'], fit.p['b'], fit.p['phi'], fit.p['x0'], fit.p['y0'])
bf = best_fit(theta)

fig, ax = plt.subplots()

ax.set_title('Zillions of ellipses make an error band.')
ax.plot(*exact, color='black', marker=',', linestyle='none')
# First we draw the best-fit value.
gvar_plot.mean(ax, *bf, color='darkgreen')
# For each point on the best fit, we draw three ellipses around it; superposed they make a band.
gvar_plot.band(ax, *bf, sigma=[1,2,3], color='green', alpha=1/128)

gvar_plot.errorbar(ax, X, Y, marker=',', color='black', linestyle='none')

ax.set_xlim(6.5, 13.5)
ax.set_ylim(10.5, 13.5)


plt.show()
