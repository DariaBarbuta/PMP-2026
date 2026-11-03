import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gamma

alpha_post = 181.8
beta_post = 10.1

date_simulate = gamma.rvs(a=alpha_post, scale=1/beta_post, size=100000, random_state=42)

interval_hdi = az.hdi(date_simulate, prob=0.94)
modul_map = (alpha_post - 1) / beta_post

print(interval_hdi)
print(modul_map)

plt.hist(date_simulate, bins=100, density=True, alpha=0.7)
plt.show()