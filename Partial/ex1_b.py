#b)

"""
4 camere: 1,2,3,4

camera      red     green   blue
1            0.7    0.2     0.1
2            0.2    0.6     0.2
3            0.1    0.3     0.6
4            0.5    0.2     0.3

sceventa: red, green, blue, blue, red

"""
import numpy as np
from hmmlearn import hmm
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

camere = ["1", "2", "3", "4"]
culori = ["red", "green", "blue"]

model=hmm.CategoricalHMM(n_components=4, init_params="")
model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])
model.transmat_ = np.array([[0.0, 0.5, 0.0, 0.5],
                            [0.5, 0.0, 0.5, 0.0],
                            [0.0, 0.5, 0.0, 0.5],
                            [0.5, 0.0, 0.5, 0.0]])
model.emissionprob_ = np.array([[0.7, 0.2, 0.1],
                                [0.2, 0.6, 0.2],
                                [0.1, 0.3, 0.6],
                                [0.5, 0.2, 0.3]])
model.n_features = len(culori)

secventa = ["red", "green", "blue", "blue", "red"]
X = np.array([[culori.index(o)] for o in secventa])
logp, cale = model.decode(X, algorithm="viterbi")
print("secventa:", [camere[i] for i in cale])
print("log prob:", logp, " prob:", np.exp(logp))





