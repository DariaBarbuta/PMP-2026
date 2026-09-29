import numpy as np

zar=np.random.randint(1,7,size=100000)

rosu = np.full(100000, 3)
albastru = np.full(100000, 4)
negru = np.full(100000, 2)

prime=np.isin(zar,[2,3,5])
sase=(zar==6)
other=~(prime|sase)

negru[prime] += 1
rosu[sase] += 1
albastru[other] += 1

probabilitati_rosu = rosu / 10 #3+4+2+1(bila adaugata dupa zar) = 10
extragere_rosu=np.random.rand(100000) < probabilitati_rosu

prob_estimata=np.mean(extragere_rosu)

print(f"Probabilitatea estimata de a extrage o bila rosie este: {prob_estimata:.4f}")