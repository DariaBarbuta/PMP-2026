#a) distibutie Bernoulli repetata pana la primul succes(distributiegeometrica)
#b)
import random
N=0
S=0

while True:
    N+=1
    coin=random.choice([0,1]) #1=stema, 0= ban

    if coin==1:
        dice=random.randint(1,6)
        S+=(dice-3)
        break
    else:
        S-=0.5

print(f"Numarul de pasi: {N}")
print(f"Suma totala: {S}")
