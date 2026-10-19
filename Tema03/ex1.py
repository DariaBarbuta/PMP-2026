"""
In jocul prezentat sunt 3 usi(notate 0 ,1 si 2). Doar in spaele uneia singure se afla premiul. Jucatorul alege o usa(fara a o deschide momentan), dupa care gazda(care stie ce se afla in spatele fiecarei usi) deschide una dintre celelalte doua usi, care nu contine premiul. Jucatorul are apoi optiunea de a ramane la usa aleasa initial sau de a schimba alegerea.
Raspunsul matematic este ca jucatorul are sanse mai mari de a castiga daca schimba alegerea initiala. Daca jucatorul ramane la alegerea initiala, sansele de castig sunt de 1/3, in timp ce daca schimba alegerea, sansele cresc la 2/3.


Prblema se transpune intr un model de retea bayesiana folosind 3 variabile. Fiecare dintre ele poate lua una din 3 stari (usa 0,1 sau 2).
    -C(Contestant) - usa aleasa de jucator; distributia este uniforma deci sansa de a alege fiecare usa este 1/3
    -P(Prize) - usa care contine premiul; distributia este uniforma deci sansa de a alege fiecare usa este 1/3
    -H(Host) - usa deschisa de gazda; distributia depinde de alegerile facute de jucator(C) si de locul premiului(P), deci nu este uniforma. Daca jucatorul a ales usa cu premiul, gazda poate deschide oricare dintre celelalte doua usi. Daca jucatorul a ales o usa fara premiu, gazda va deschide singura usa care nu este nici cea aleasa, nici cea cu premiul.

Structura (C → H ← P)
Alegerea ta (C) și locul unde e ascuns premiul (P) sunt evenimente complet independente. În schimb, ușa deschisă de prezentator (H) se bazează pe ambele. De aceea, schema arată că C și P dictează împreună valoarea lui H.

Modul în care prezentatorul alege ce ușă să deschidă (H) este reprezentat printr-un tabel numit CPD. Acest tabel are 3 rânduri (reprezentând ușa pe care o deschide el) și 9 coloane (reprezentând toate combinațiile posibile dintre alegerea ta, C, și locul premiului, P).

Regulile după care se completează tabelul sunt foarte stricte:

    -Prezentatorul nu deschide niciodată ușa aleasă de tine (C) și nici ușa cu premiul (P).

    -Dacă ai nimerit din prima ușa cu premiul (C și P sunt aceeași ușă), prezentatorului îi rămân două uși goale. El va deschide una dintre ele la întâmplare (șansă de 0.5 și 0.5).

    -Dacă ai ales o ușă goală (C și P sunt diferite), prezentatorul este forțat să deschidă singura ușă goală rămasă (șansă de 1).

Explicatia codului:

Construim graful cu două arce, C → H și P → H, pentru că ușa deschisă de gazdă depinde de alegerea jucătorului și de locul premiului. C și P nu au părinți, deci primesc distribuții uniforme: fiecare dintre cele 3 uși are probabilitatea 1/3.

from pgmpy.models import DiscreteBayesianNetwork   # în versiunile vechi: BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([("C", "H"), ("P", "H")])

cpd_c = TabularCPD("C", 3, [[1/3], [1/3], [1/3]])
cpd_p = TabularCPD("P", 3, [[1/3], [1/3], [1/3]])


Fiecare coloană corespunde unei combinații (C, P), iar fiecare rând unei uși deschise de gazdă. Coloanele sunt în ordinea (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2).

Tabelul codifică regulile gazdei:

    Când C = P (coloanele 0, 4, 8), jucătorul a nimerit ușa cu premiul, deci gazda alege la întâmplare între celelalte două uși (0.5 și 0.5).
    Când C ≠ P, gazda are o singură ușă posibilă, deci pe acel rând apare 1.
Ușa aleasă de jucător și ușa cu premiul au mereu 0 pe rândul lor, pentru că gazda nu le deschide niciodată.

Exemplu: coloana (C=0, P=1) are 1 pe rândul H=2, pentru că singura ușă care nu e nici aleasă, nici cu premiul este ușa 2.


cpd_h = TabularCPD(
    "H", 3,
    [[0,   0, 0,  0, 0.5, 1,  0, 1, 0.5],   # H = 0
     [0.5, 0, 1,  0, 0,   0,  1, 0, 0.5],   # H = 1
     [0.5, 1, 0,  1, 0.5, 0,  0, 0, 0  ]],  # H = 2
    evidence=["C", "P"], evidence_card=[3, 3],
)

Atașăm cele trei tabele la graf și verificăm că modelul este coerent, adică fiecare coloană de probabilități însumează 1.

model.add_cpds(cpd_c, cpd_p, cpd_h)
print(model.check_model())

Simulăm o situație concretă: jucătorul a ales ușa 0, iar gazda a deschis ușa 2. Întrebăm modelul unde este premiul, știind aceste două informații.

infer = VariableElimination(model)
print(infer.query(["P"], evidence={"C": 0, "H": 2}))

Rezultatul este P(P=0) = 1/3, P(P=1) = 2/3 și P(P=2) = 0.

Ușa 2 are probabilitatea 0, pentru că gazda a deschis-o și era goală.
Ușa 0, aleasă inițial, rămâne la 1/3, pentru că deschiderea ușii 2 nu aduce nicio informație nouă despre ea.
Ușa 1 preia toată probabilitatea rămasă, adică 2/3.

Deci jucătorul trebuie să schimbe alegerea, ceea ce confirmă răspunsul matematic de la începutul explicației.
"""