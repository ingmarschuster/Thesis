# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:52:10 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

import pylab

import mc #McSample
import distributions as dist


num_samp = 1000

post_d = dist.mvnorm(0,100)# stats.norm(0,10)
lpost = post_d.logpdf


s_high = mc.mcmc.sample(num_samp+200, np.zeros(1), mc.mcmc.GaussMHKernel(lpost,1))[0][-num_samp:]
s_low  = mc.mcmc.sample(num_samp+200, np.zeros(1), mc.mcmc.GaussMHKernel(lpost,10000))[0][-num_samp:]

s_opt  = mc.mcmc.sample(num_samp+200, np.zeros(1), mc.mcmc.GaussMHKernel(lpost,550))[0][-num_samp:] # for 1d target: acceptance rate around 0.44 is optimal
s_opt_mala  = mc.mcmc.sample(num_samp+200, np.zeros(1), mc.mcmc.MalaKernel(post_d.log_pdf_and_grad, 1, 340))[0][-num_samp:] # for 1d target: acceptance rate around 0.57 is optimal
s_am  = mc.mcmc.sample(num_samp+200, np.zeros(1), mc.mcmc.HaarioKernel(post_d.logpdf, 0, 1))[0][-num_samp:] 

s_iid = post_d.rvs(num_samp)


f, ax = pylab.subplots(2,2,sharex=True,sharey=True)
ax[0][0].plot(np.arange(num_samp), s_high)
ax[1][0].plot(np.arange(num_samp), s_opt)
ax[0][1].plot(np.arange(num_samp), s_low)
ax[1][1].plot(np.arange(num_samp), s_iid)

ylim = np.hstack([s.flatten() for s in (s_high, s_opt, s_low, s_iid)])
ylim = (ylim.min()-1,ylim.max()+1)


ax[0][0].set_ylim(ylim)
pylab.show()


f, ax = pylab.subplots(1,3,sharex=True,sharey=True)

cutoff=500

ylim = np.hstack([s.flatten() for s in (s_opt_mala[:cutoff], s_opt[:cutoff], s_iid[:cutoff])])
ylim = (ylim.min()-1,ylim.max()+1)

ax[0].plot(s_opt.flatten()[:cutoff])
ax[1].plot(s_opt_mala.flatten()[:cutoff])
ax[2].plot(s_iid.flatten()[:cutoff])
ax[0].set_ylim(ylim)
pylab.show()