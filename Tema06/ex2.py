import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA = 20 / 60      # clienti / minut
MEAN_PAY = 2.0        # minute
LIMITA = 15.0         # minute
SEED = 42
M = 20000             # ore simulate


# a) o singura ora; coada 1 = casa, coada 2 = statia de gatit (ambele FIFO)
def simulare_ora(alpha, rng):
    n = rng.poisson(LAMBDA * 60)
    sosiri = np.sort(rng.uniform(0, 60, n))       # proces Poisson pe [0,60]
    plata = rng.exponential(MEAN_PAY, n)
    gatit = rng.exponential(alpha, n)
    iesire_casa = np.zeros(n); primire = np.zeros(n)
    for i in range(n):
        start_casa = sosiri[i] if i == 0 else max(sosiri[i], iesire_casa[i - 1])
        iesire_casa[i] = start_casa + plata[i]
        start_gatit = iesire_casa[i] if i == 0 else max(iesire_casa[i], primire[i - 1])
        primire[i] = start_gatit + gatit[i]
    return sosiri, primire


# Simulare vectorizata a M ore, cu numere aleatoare comune pentru orice alpha
# (timp gatit = alpha * E, E ~ Exp(1)), ca probabilitatea sa fie monotona in alpha.
def genereaza_baza(M, rng):
    N = rng.poisson(LAMBDA * 60, M)
    nmax = N.max()
    masca = np.arange(nmax) < N[:, None]
    A = np.sort(np.where(masca, rng.uniform(0, 60, (M, nmax)), np.inf), axis=1)
    A = np.where(masca, A, 0.0)
    return N, masca, A, rng.exponential(MEAN_PAY, (M, nmax)), rng.exponential(1.0, (M, nmax))


def rulare(alpha, baza):
    N, masca, A, S, E = baza
    M, nmax = A.shape
    p_casa = np.zeros(M); p_gatit = np.zeros(M)
    T = np.full((M, nmax), np.nan)
    for j in range(nmax):
        m = masca[:, j]
        casa = np.maximum(A[:, j], p_casa) + S[:, j]
        gata = np.maximum(casa, p_gatit) + alpha * E[:, j]
        T[m, j] = (gata - A[:, j])[m]
        p_casa = np.where(m, casa, p_casa); p_gatit = np.where(m, gata, p_gatit)
    asteptare = T - S - alpha * E     # timp total - servire (casa + gatit)
    return T, asteptare


def p_toti(alpha, baza):              # P(toti clientii din ora <= 15 min)
    T, _ = rulare(alpha, baza)
    return np.mean(np.nan_to_num(T, nan=-1).max(axis=1) <= LIMITA)


def bisectie(prag, baza, lo=1e-3, hi=2.999):   # cel mai mare alpha cu p >= prag
    if p_toti(lo, baza) < prag:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_toti(mid, baza) >= prag else (lo, mid)
    return lo


rng = np.random.default_rng(SEED)

# a) exemplu
s, p = simulare_ora(1.0, rng)
print(f"a) Exemplu (alpha=1): {len(s)} clienti, primul: sosire {s[0]:.2f}, mancare {p[0]:.2f}")

# b)
baza = genereaza_baza(M, rng)
p0 = p_toti(1e-6, baza)
print(f"b) P(toti <= 15 min) pentru alpha -> 0: {p0:.4f}")
a95 = bisectie(0.95, baza)
print("   alpha pentru 95%: " + ("NU EXISTA (blocajul este casa)" if a95 is None else f"{a95:.4f}"))

# c) nu exista alpha cu 95%; raportam cazul cel mai favorabil (alpha -> 0)
alpha_c = 1e-3
T, W = rulare(alpha_c, baza)
print(f"c) alpha->0: timp mediu sosire->mancare: {np.nanmean(T):.2f} min; asteptare efectiva: {np.nanmean(W):.2f} min")

# d) variabilitate: 10 seminte diferite
ps = [p_toti(alpha_c, genereaza_baza(M, np.random.default_rng(s))) for s in range(100, 110)]
print(f"d) M={M}, seed={SEED}, bisectie 40 pasi pe [0.001, 3)")
print(f"   P la alpha->0 pe 10 seminte: medie {np.mean(ps):.4f}, std {np.std(ps):.4f}")
print(f"   eroare standard a lui p: {np.sqrt(p0*(1-p0)/M):.4f}")

# grafic P(alpha)
alfe = np.linspace(0.05, 2.5, 25)
plt.plot(alfe, [p_toti(a, baza) for a in alfe], 'o-')
plt.axhline(0.95, color='r', ls='--', label='95%')
plt.xlabel(r'$\alpha$ (min)'); plt.ylabel('P(toți clienții ≤ 15 min)')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('grafic_ex2.png', dpi=150)