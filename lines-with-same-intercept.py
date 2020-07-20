#!/usr/bin/env python3


import gvar as gv
import lsqfit as lsf
import numpy as np
import matplotlib.pyplot as plt

def banner(message):
    print(f"\n#\n#\t{message}\n#\n")

# I lifted this from https://lsqfit.readthedocs.io/en/latest/lsqfit.html#classes-for-bayesian-integrals, four sets of data which all have the same interecept but different slopes.

banner("SETUP")

data = gv.gvar(dict(
    d1=['1.154(10)', '2.107(16)', '3.042(22)', '3.978(29)'],
    d2=['0.692(10)', '1.196(16)', '1.657(22)', '2.189(29)'],
    d3=['0.107(10)', '0.030(16)', '-0.027(22)', '-0.149(29)'],
    d4=['0.002(10)', '-0.197(16)', '-0.382(22)', '-0.627(29)'],
    ))

print("Here is a dataset---four lines which we know can have different slopes but all have the same intercept.")
for d in data:
    print(f"{d} = {data[d]}")

class Linear(lsf.MultiFitterModel):
    def __init__(self, datatag, x, intercept, slope):
        super(Linear, self).__init__(datatag)
        # the independent variable
        self.x = np.array(x)
        # keys used to find the intercept and slope in a parameter dictionary
        self.intercept = intercept
        self.slope = slope

    def fitfcn(self, p):
        try:
            return p[self.intercept] + p[self.slope] * self.x
        except KeyError:
            # slope parameter marginalized/omitted
            return len(self.x) * [p[self.intercept]]

    def buildprior(self, prior, mopt=None):
        " Extract the model's parameters from prior. "
        newprior = {}
        newprior[self.intercept] = prior[self.intercept]
        if mopt is None:
            # slope parameter marginalized/omitted if mopt is not None
            newprior[self.slope] = prior[self.slope]
        return newprior

    def builddata(self, data):
        " Extract the model's fit data from data. "
        return data[self.datatag]

models = [
   Linear('d1', x=[1,2,3,4], intercept='a', slope='s1'),
   Linear('d2', x=[1,2,3,4], intercept='a', slope='s2'),
   Linear('d3', x=[1,2,3,4], intercept='a', slope='s3'),
   Linear('d4', x=[1,2,3,4], intercept='a', slope='s4'),
   ]

prior = gv.gvar(dict(a='0(1)', s1='0(1)', s2='0(1)', s3='0(1)', s4='0(1)'))



banner("SIMULTANEOUS FIT")
fitter = lsf.MultiFitter(models=models)
fit = fitter.lsqfit(data=data, prior=prior)

print(fit)
print(f"intercept = {fit.p['a']}")

fit.show_plots(view='std')

banner("Marginalizing over different slopes")
marginalized = fitter.lsqfit(data=data, prior=prior, mopt=True)

print(marginalized)
print(f"intercept = {marginalized.p['a']}")

banner("CHAINED FIT")
chained = fitter.chained_lsqfit(data=data, prior=prior)
print(chained.formatall())
print(f"intercept = {chained.p['a']}")

