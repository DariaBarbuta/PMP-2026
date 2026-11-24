# Tema 7 – Inferență Bayesiană cu PyMC

Fișiere: `Tema07.ipynb` (cod), `img/` (grafice), `README.md`.
Toate rulările folosesc `random_seed=0`. HDI = 95%.

## Exercițiul 1 – Dimensiunea unei populații
**Model:** N ~ DiscreteUniform(20, 200), Y | N ~ Hypergeometric(N, M=20, n=15), Y=4 observat.
**Setări PyMC:** 4 lanțuri, 2000 tune, 5000 draws/lanț (20 000 eșantioane), sampler Metropolis (variabilă discretă).
**Diagnostic:** r_hat = 1.00, ESS bulk ≈ 3920. Am verificat și cu calculul exact pe grilă.

| | MCMC | Exact |
|---|---|---|
| Media | 101.54 | 102.61 |
| Mediana | 94.0 | 95.0 |
| HDI 95% | [42, 177] | [44, 180] |
| P(N > 100 \| Y=4) | 0.4429 | 0.4494 |

Estimarea clasică Lincoln–Petersen: 20·15/4 = 75.
**Interpretare:** posteriorul este asimetric, cu coadă spre valori mari ale lui N, deci media și mediana depășesc 75. Estimarea clasică este un singur punct, pe când posteriorul arată un domeniu larg de valori plauzibile (44–181). Diferențele MCMC vs. exact sunt eroare de eșantionare.
**Predictiv (d):** numărul de animale marcate într-o nouă captură are modul la 3 (P≈0.21), cu 2 și 4 apropiate. Posteriorul lui N descrie incertitudinea despre parametru, predictivul descrie date viitoare și combină incertitudinea despre N cu variabilitatea aleatoare a eșantionării.
Grafice: `img/ex1_posterior.png`, `img/ex1_predictiv.png`.

## Exercițiul 2 – Compararea tratamentelor
**Model:** λA, λB ~ Gamma(2, 4) (formă–rată), X | λ ~ Exponential(λ).
**Setări PyMC:** 4 lanțuri, 2000 tune, 2000 draws/lanț (8000 eșantioane), NUTS. r_hat = 1.00, ESS > 5000.

| | Medie | HDI 95% |
|---|---|---|
| μA = 1/λA | 3.80 zile | [1.72, 6.40] |
| μB = 1/λB | 2.92 zile | [1.29, 4.85] |

- P(μB < μA | date) ≈ 0.726 
- P(X_B,nou < X_A,nou | date) ≈ 0.556

**Interpretare:** există dovezi moderate că B are timp mediu mai mic, dar HDI-urile se suprapun mult. Probabilitatea predictivă e mult mai aproape de 0.5 decât cea de la c), pentru că timpii individuali (exponențiali) au variabilitate mare. Dacă știm destul de bine ordinea mediilor, nu rezultă că fiecare pacient B se recuperează mai repede decât fiecare pacient A.
Grafice: `img/ex2_mu.png`.

## Exercițiul 3 – Schimbare graduală a ratei
**Model:** Y_t ~ Poisson(λ_t), λ_t = λ1 + (λ2−λ1)·sigmoid(k(t−τ)); λ1, λ2 ~ Exponential(rată 0.2), τ ~ Uniform(1,30), k ~ Exponential(rată 1).
**Setări PyMC:** 4 lanțuri, 2000 tune, 2000 draws/lanț (8000 eșantioane), NUTS, target_accept=0.9. r_hat = <...>, ESS = <...>, divergențe = <...>.

| Param. | Medie | HDI 95% |
|---|---|---|
| λ1 | 2.93 | [0.48, 4.65] |
| λ2 | 9.85 | [6.93, 14.33] |
| τ  | 16.0 | [9.88, 24.35]  |
| k  | 0.60 | [0.05, 1.81] |

**Semnificație:** λ1, λ2 = rata inițială/finală; τ = ziua la jumătatea tranziției; k = cât de abruptă e schimbarea (k mare ≈ salt, k mic ≈ tranziție lentă).
P(λ2 > λ1 | date) = <...>.
**Predictiv:** rata estimată crește de la ≈<3.3> la ≈<9.3> solicitări/zi; <...>% din punctele observate cad în intervalul predictiv 95%. Modelul reproduce creșterea graduală. Limite: τ și k sunt incerte (HDI larg), iar modelul Poisson presupune media = varianța și independență între zile, ceea ce poate fi nerealist.
Grafice: `img/ex3_trace.png`, `img/ex3_tau_k.png`, `img/ex3_lambda_t.png`, `img/ex3_ppc.png`.