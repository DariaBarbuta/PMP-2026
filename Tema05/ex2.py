# stari ascunse: dificultatea testului (D, M, U); observatii: nota (FB, B, S, NS)

import numpy as np

np.set_printoptions(linewidth=120, formatter={"float": lambda v: f"{v:.4e}"})

# a) definirea modelului
stari = ["D", "M", "U"]
note = ["FB", "B", "S", "NS"]
N = len(stari)

pi = np.array([1/3, 1/3, 1/3])            # primul test are aceeasi probabilitate pentru toate

A = np.array([[0.0, 0.5,  0.5 ],          # randuri si coloane: D, M, U
                [0.5, 0.25, 0.25],
                [0.5, 0.25, 0.25]])

B = np.array([[0.10, 0.20, 0.40, 0.30],   # coloane: FB, B, S, NS
                [0.15, 0.25, 0.50, 0.10],
                [0.20, 0.30, 0.40, 0.10]])

# verific ca fiecare rand are suma 1
assert np.isclose(pi.sum(), 1) and np.allclose(A.sum(axis=1), 1) and np.allclose(B.sum(axis=1), 1)
print("Sumele pe randuri: A =", A.sum(axis=1).round(4).tolist(),"| B =", B.sum(axis=1).round(4).tolist(), "| pi =", round(pi.sum(), 4))

secv = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B", "S"]
o = [note.index(x) for x in secv]         # notele transformate in indici
T = len(o)

# b) initializarea: delta_1(j) = pi_j * b_j(o_1)
delta1 = pi * B[:, o[0]]
print("\ndelta_1 =", delta1)

# c) recurenta
# delta[t, j] = scorul celui mai bun traseu care se termina in j la momentul t
# psi[t, j] = starea de la t-1 din care vine acel traseu (ma ajuta la reconstructie)
def viterbi(pi, A, B, o):
    T, N = len(o), len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = pi * B[:, o[0]]
    for t in range(1, T):
        for j in range(N):
            scoruri = [delta[t-1, i] * A[i, j] for i in range(N)]
            psi[t, j] = int(np.argmax(scoruri))       # la egalitate ia primul indice
            delta[t, j] = B[j, o[t]] * max(scoruri)
    return delta, psi

delta, psi = viterbi(pi, A, B, o)

# d) verificare intermediara (dupa primele doua note FB)
print("\ndelta_2 =", delta[1])

# delta_2(D) = b_D(FB) * max(delta_1(i) * a_iD); din D nu pot ajunge in D (a_DD = 0)
manual_D = 0.1 * max(delta1[1] * 0.5, delta1[2] * 0.5)
# delta_2(M) = b_M(FB) * max(delta_1(i) * a_iM), cu i = D, M, U
manual_M = 0.15 * max(delta1[0] * 0.5, delta1[1] * 0.25, delta1[2] * 0.25)
assert np.isclose(manual_D, delta[1, 0]) and np.isclose(manual_M, delta[1, 1])
print(f"Verificare manuala: delta_2(D) = {manual_D:.4e}, delta_2(M) = {manual_M:.4e} (la fel ca in cod)")

# e) reconstructia traseului: pornesc de la starea finala cu scor maxim si merg inapoi cu psi
x = [int(np.argmax(delta[-1]))]
for t in range(T-1, 0, -1):
    x.append(int(psi[t, x[-1]]))
x = x[::-1]

traseu = [stari[i] for i in x]
p_max = delta[-1].max()

print("\nSecventa de dificultati:", traseu)
print(f"max P(x_1..x_12, o_1..o_12) = {p_max:.4e}  (ln = {np.log(p_max):.4f})")

# f) interpretare
# - Viterbi cauta cea mai probabila secventa completa, nu cea mai probabila dificultate la fiecare test.
# - P(x|o) = P(x,o)/P(o) si P(o) nu depinde de x, deci maximizez direct P(x,o).
# - Valoarea de mai sus este probabilitate comuna P(x,o), nu conditionata. Este mica pentru ca
#   inmultesc 12 emisii si 11 tranzitii.
# - Maximul nu e unic: M si U au randuri identice in A, deci la t=2 apare o egalitate.
#   Sunt 2 secvente optime (difera doar prima stare, D sau U), cu acelasi scor.

# optional: Forward si P(x*|o)
# alpha[t, j] = P(o_1..o_t, X_t = j)
def forward(pi, A, B, o):
    T, N = len(o), len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, o[0]]
    for t in range(1, T):
        alpha[t] = B[:, o[t]] * (alpha[t-1] @ A)      # b_j(o_t) * suma_i alpha[t-1, i] * a_ij
    return alpha

alpha = forward(pi, A, B, o)
P_o = alpha[-1].sum()                                  # P(o) = suma alpha_T(j)
print(f"\nP(o_1..o_12) (Forward) = {P_o:.4e}")
print(f"P(x* | o) = P(x*, o) / P(o) = {p_max / P_o:.4e}")

# comparatie cu hmmlearn
try:
    from hmmlearn import hmm
    m = hmm.CategoricalHMM(n_components=3, init_params="")
    m.startprob_, m.transmat_, m.emissionprob_, m.n_features = pi, A, B, 4
    logp, cale = m.decode(np.array(o).reshape(-1, 1), algorithm="viterbi")
    print("hmmlearn:", [stari[i] for i in cale], f"{np.exp(logp):.4e}")
    print("(difera doar prima stare: egalitate intre doua secvente optime, acelasi scor)")
except Exception as e:
    print("hmmlearn indisponibil:", e)