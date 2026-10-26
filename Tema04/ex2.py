import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

stari = ["Soare", "Ploaie", "Nori"]
n = 6
muchii = [(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)]
p = np.array([0.5, 0.3, 0.2])

Psi = np.array([[0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6]])
Psi2 = np.array([[0.6, 0.1, 0.1],
                [0.1, 0.6, 0.1],
                [0.1, 0.1, 0.6]])

graf = nx.Graph()
graf.add_nodes_from(range(n))
graf.add_edges_from(muchii)

print("Noduri:", graf.number_of_nodes(), "Muchii:", graf.number_of_edges())
print("Verificare conexitate:", nx.is_connected(graf))
print("Verificare arbore:", nx.is_tree(graf))
for i in graf.nodes:
    print(f"S{i} are vecinii:", [f"S{j}" for j in graf.neighbors(i)])

pozitii = nx.spring_layout(graf, seed=3)

plt.figure()
nx.draw(graf, pozitii, with_labels=True, labels={i: f"S{i}" for i in range(n)},
        node_color="lightpink", node_size=1200)
plt.title("Graful ales (a)")
plt.savefig("graf_a.png", dpi=150, bbox_inches="tight")

try:
    import pyagrum as gum
    model = gum.MarkovRandomField()
    for i in range(n):
        model.add(gum.LabelizedVariable(f"S{i}", f"S{i}", stari))
    for i in range(n):
        f = model.addFactor([f"S{i}"])
        for k in range(3):
            f[{f"S{i}": stari[k]}] = float(p[k])
    for i, j in muchii:
        f = model.addFactor([f"S{i}", f"S{j}"])
        for a in range(3):
            for b in range(3):
                f[{f"S{i}": stari[a], f"S{j}": stari[b]}] = float(Psi[a, b])
except Exception as e:
    model = None
    print("pyAgrum nu a mers:", e)

graf_model = nx.Graph()
graf_model.add_nodes_from(f"S{i}" for i in range(n))
graf_model.add_edges_from((f"S{i}", f"S{j}") for i, j in muchii)
pozitii_model = {f"S{i}": pozitii[i] for i in range(n)}

plt.figure()
nx.draw(graf_model, pozitii_model, with_labels=True,
        node_color="lightgreen", node_size=1200)
plt.title("Graful modelului Markov (c)")
plt.savefig("graf_c.png", dpi=150, bbox_inches="tight")

print()
print("P(s1..sn) = (1/Z) * produs(phi_i(s_i)) * produs(phi_ij(s_i, s_j))")

toate = list(itertools.product(range(3), repeat=n))


def pondere(x, Psi):
    w = 1.0
    for i in range(n):
        w *= p[x[i]]
    for i, j in muchii:
        w *= Psi[x[i], x[j]]
    return w


def analiza(Psi, observatii):
    libere = [i for i in range(n) if i not in observatii]
    posibile = [x for x in toate
                if all(x[i] == v for i, v in observatii.items())]
    total = sum(pondere(x, Psi) for x in posibile)

    marginale = {}
    for i in libere:
        m = np.zeros(3)
        for x in posibile:
            m[x[i]] += pondere(x, Psi) / total
        marginale[i] = m

    cel_mai_bun = max(posibile, key=lambda x: pondere(x, Psi))
    map_comun = [stari[cel_mai_bun[i]] for i in libere]
    map_marginal = [stari[marginale[i].argmax()] for i in libere]

    print()
    print("Observatii:", {f"S{i}": stari[v] for i, v in observatii.items()})
    for i in libere:
        print(f"  S{i}:", np.round(marginale[i], 4), "->", stari[marginale[i].argmax()])
    print("  MAP comun       :", map_comun)
    print("  Maxime marginale:", map_marginal)
    print("  Coincid:", map_comun == map_marginal)
    return marginale


print("\nPsi dat")
marginale = analiza(Psi, {1: 0, 5: 2})
analiza(Psi, {1: 0, 5: 1})
analiza(Psi, {3: 0, 5: 2})

print("\nPsi2 (diagonala 0.6, restul 0.1)")
analiza(Psi2, {0: 1, 1: 0})

if model is not None:
    print("\nVerificare pyAgrum (S1=Soare, S5=Nori)")
    try:
        ie = gum.ShaferShenoyMRFInference(model)
        ie.setEvidence({"S1": "Soare", "S5": "Nori"})
        ie.makeInference()
        for i in [0, 2, 3, 4]:
            post = ie.posterior(f"S{i}")
            valori = [post[{f"S{i}": s}] for s in stari]
            print(f"S{i} pyAgrum  :", np.round(valori, 4))
            print(f"S{i} enumerare:", np.round(marginale[i], 4))
    except Exception as e:
        print("Inferenta pyAgrum nu a mers:", e)

plt.show()

