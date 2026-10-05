import random
import matplotlib.pyplot as plt

nr_experiments = 10000
results=[]
for i in range(nr_experiments):
    S=0
    while True:
        coin=random.choice([0,1]) #1=stema, 0= ban
        if coin==1:
            dice=random.randint(1,6)
            S+=(dice-3)
            break
        else:
            S-=0.5
    results.append(S)

m=sum(results)/nr_experiments
print(f"Media rezultatelor dupa {nr_experiments} experimente este: {m}")

plt.hist(results, bins=30, edgecolor='black', color='pink')
plt.title('Distributia rezultatelor')
plt.xlabel('Suma totala')
plt.ylabel('Frecventa')
plt.show()