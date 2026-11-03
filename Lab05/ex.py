import numpy as np
from hmmlearn import hmm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

stari = ["Dificil", "Mediu", "Usor"]
note = ["FB", "B", "S", "NS"]

model = hmm.CategoricalHMM(n_components=3, init_params="")
model.startprob_ = np.array([1/3, 1/3, 1/3])
model.transmat_ = np.array([[0.0, 0.5, 0.5],
                            [0.5, 0.25, 0.25],
                            [0.5, 0.25, 0.25]])
model.emissionprob_ = np.array([[0.1, 0.2, 0.4, 0.3],
                                [0.15, 0.25, 0.5, 0.1],
                                [0.2, 0.3, 0.4, 0.1]])
model.n_features = 4

secv = ["FB","FB","S","B","B","S","B","B","NS","B","B","S"]
X = np.array([[note.index(o)] for o in secv])

logp, cale = model.decode(X, algorithm="viterbi")
print("secventa:", [stari[i] for i in cale])
print("log prob:", logp, " prob:", np.exp(logp))


A = model.transmat_

G = nx.DiGraph()
for i in range(3):
    for j in range(3):
        if i != j and A[i, j] > 0:
            G.add_edge(stari[i], stari[j], w=A[i, j])

pos = {"Dificil": (0, 1), "Mediu": (-1, 0), "Usor": (1, 0)}
fig, ax = plt.subplots(figsize=(6.5, 5.5))
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.15",
        node_size=3000, arrowsize=20, ax=ax)
nx.draw_networkx_edge_labels(G, pos,
    edge_labels={(u, v): d_["w"] for u, v, d_ in G.edges(data=True)},
    label_pos=0.3, font_size=8, ax=ax)

dir_bucla = {"Dificil": (0, 1), "Mediu": (-1, -0.6), "Usor": (1, -0.6)}
for k, nume in enumerate(stari):
    if A[k, k] > 0:
        x, y = pos[nume]
        dx, dy = dir_bucla[nume]
        n = np.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        px, py = -dy, dx
        a = (x + 0.21 * dx - 0.09 * px, y + 0.21 * dy - 0.09 * py)
        b = (x + 0.21 * dx + 0.09 * px, y + 0.21 * dy + 0.09 * py)
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle="arc3,rad=-2.2",
            arrowstyle="-|>", mutation_scale=15, color="black", zorder=5))
        ax.text(x + 0.5 * dx, y + 0.5 * dy, f"{A[k,k]}", fontsize=8, ha="center", va="center")

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.6, 1.5)
ax.axis("off")
ax.set_title("Diagrama de stare")
plt.savefig("diagrama.png", dpi=130, bbox_inches="tight")