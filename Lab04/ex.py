import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pyagrum as gum

beta = 1.0   # enuntul nu da valoarea lui beta; o alegem noi (poti incerca si alte valori)

# ---------------- a) Graful si clicile ----------------
noduri = ["A1", "A2", "A3", "A4", "A5"]
muchii = [("A1", "A2"), ("A1", "A3"),
        ("A2", "A4"), ("A2", "A5"),
        ("A3", "A4"), ("A4", "A5")]

# model pyAgrum: 5 variabile binare
mrf = gum.MarkovRandomField("Retea")
for v in noduri:
    mrf.add(gum.LabelizedVariable(v, v, 2))
for u, v in muchii:
    mrf.addFactor([u, v])  # temporar, vezi mai jos clicile reale

# vizualizare + clici cu networkx
G = nx.Graph(muchii)
clici = [sorted(c) for c in nx.find_cliques(G)]
clici.sort()
print("a) Clicile maximale:", clici)

nx.draw(G, with_labels=True, node_color="lightblue", node_size=1500, font_size=12)
plt.title("Graful retelei Markov")
plt.show()

# ---------------- model pyAgrum cu potentiale pe clici maximale ----------------
mn = gum.MarkovRandomField("Retea")
for v in noduri:
    mn.add(gum.LabelizedVariable(v, v, 2))

potentiale = []
for c in clici:
    pot = mn.addFactor(c)
    for stare in itertools.product([0, 1], repeat=len(c)):
        energie = sum(stare[i] * stare[j]
                    for i in range(len(c)) for j in range(i + 1, len(c)))
        pot[dict(zip(c, stare))] = np.exp(beta * energie)   # phi_C = e^{beta * sum a_i a_j}
    potentiale.append((c, pot))

# ---------------- b) Probabilitatea comuna ----------------
stari = list(itertools.product([0, 1], repeat=5))
nenorm = []
for s in stari:
    a = dict(zip(noduri, s))
    prod = 1.0
    for c, pot in potentiale:
        prod *= pot[{v: a[v] for v in c}]
    nenorm.append(prod)

Z = sum(nenorm)
probe = [x / Z for x in nenorm]

print("\nb) Z =", Z)
print("A1 A2 A3 A4 A5 | P")
for s, p in zip(stari, probe):
    print(*s, "|", round(p, 5))

pmax = max(probe)
print("\nProbabilitate maxima:", pmax)
print("Stari de probabilitate maxima:",
    [s for s, p in zip(stari, probe) if np.isclose(p, pmax)])
print("Suma probabilitatilor:", sum(probe))