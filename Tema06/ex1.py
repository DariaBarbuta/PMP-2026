import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import betaln

# a) Y | theta ~ Binomial(n, theta)
n_obs, y_obs = 50, 8
alpha_prior, beta_prior = 2, 18

# b) posterior conjugat: Beta(alpha + y, beta + n - y)
alpha_post = alpha_prior + y_obs
beta_post = beta_prior + n_obs - y_obs
print(f"Posterior: Beta({alpha_post}, {beta_post})")

# c) estimatori
mle = y_obs / n_obs
map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
mean_post = alpha_post / (alpha_post + beta_post)
print(f"MLE: {mle:.4f}  MAP: {map_est:.4f}  Media a posteriori: {mean_post:.4f}")
print(f"Deviatia standard prior: {beta.std(alpha_prior, beta_prior):.4f}, posterior: {beta.std(alpha_post, beta_post):.4f}")

# d) grafic
x = np.linspace(0, 1, 500)
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label='A priori Beta(2,18)')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label='A posteriori Beta(10,60)')
plt.axvline(mle, color='gray', ls=':', label='MLE')
plt.xlabel(r'$\theta$ (rata de defectare)'); plt.ylabel('densitate')
plt.legend(); plt.title("Distribuții pentru rata de defectare")
plt.tight_layout(); plt.savefig('grafic_ex1.png', dpi=150); plt.close()

# e) interval credibil central 95%
lo, hi = beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"Interval credibil 95%: [{lo:.4f}, {hi:.4f}]")

# f) P(theta < 0.10 | y)
print(f"P(theta < 0.10 | y=8) = {beta.cdf(0.10, alpha_post, beta_post):.4f}")

# g) predictiv: P(X=0|y) = B(alpha_post, beta_post+10) / B(alpha_post, beta_post)
p_pred = np.exp(betaln(alpha_post, beta_post + 10) - betaln(alpha_post, beta_post))
print(f"P(X=0 | y=8) = {p_pred:.4f}")