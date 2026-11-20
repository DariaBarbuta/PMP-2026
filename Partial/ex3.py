
from matplotlib.pylab import gamma
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gamma

#a)
alpha_prior = 2
beta_prior = 8

alpha_post = alpha_prior + 5
beta_post = beta_prior + 15


date_simulate1=gamma.rvs(a=alpha_post, scale=1/beta_post, size=100000, random_state=42)
date_simulate2=gamma.rvs(a=alpha_prior, scale=1/beta_prior, size=100000, random_state=42)

interval_hdi = az.hdi(date_simulate1, prob=0.94)

modul_map = (alpha_prior - 1) / beta_prior

print(interval_hdi)
print(modul_map)

plt.hist(date_simulate1, bins=100, density=True, alpha=0.7)
plt.hist(date_simulate2, bins=100, density=True, alpha=0.7)
plt.show()

