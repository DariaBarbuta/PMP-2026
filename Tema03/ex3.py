import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from scipy.stats import binom
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def joc():
    s = random.randint(0, 1)
    n = random.randint(1, 6)
    p = 4/7 if s == 0 else 0.5
    m = sum(random.random() < p for _ in range(2 * n))
    return s if n >= m else 1 - s

N = 10000
victorii = [0, 0]
for _ in range(N):
    victorii[joc()] += 1

print("a)")
print("J0 a câștigat:", victorii[0], "->", victorii[0] / N)
print("J1 a câștigat:", victorii[1], "->", victorii[1] / N)
print("Cele mai mari șanse:", "J0" if victorii[0] > victorii[1] else "J1")

model = DiscreteBayesianNetwork([("S", "M"), ("N", "M")])

cpd_s = TabularCPD("S", 2, [[0.5], [0.5]])
cpd_n = TabularCPD("N", 6, [[1/6]] * 6)

valori = np.zeros((13, 12))
for s in range(2):
    p = 4/7 if s == 0 else 0.5
    for n in range(1, 7):
        valori[:, s * 6 + (n - 1)] = binom.pmf(np.arange(13), 2 * n, p)

cpd_m = TabularCPD("M", 13, valori, evidence=["S", "N"], evidence_card=[2, 6])

model.add_cpds(cpd_s, cpd_n, cpd_m)
assert model.check_model()

infer = VariableElimination(model)
print("\nc) P(S | M=1)")
print(infer.query(["S"], evidence={"M": 1}))