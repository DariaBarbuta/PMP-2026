import random

N=100000

X_par=0
#a="suma>=10 stiind ca X e par"
a=0

X_sau_Y_par=0
#b="suma>=10 stiind ca X sau Y e par"
b=0

suma_opt=0
#c="X si Y sunt pare si suma=8"
c=0

A=0 #A={X+Y>=10}
B=0 #B={X=par}
A_si_B=0

for i in range(N):
    X=random.randint(1,6)
    Y=random.randint(1,6)
    suma=X+Y
    if X%2==0:
        X_par+=1
        if suma>=10:
            a+=1
    if X%2==0 or Y%2==0:
        X_sau_Y_par+=1
        if suma>=10:
            b+=1
    if suma==8:
        suma_opt+=1
        if X%2==0 and Y%2==0:
            c+=1
    if suma>=10:
        A+=1
    if X%2==0:
        B+=1
    if suma>=10 and X%2==0:
        A_si_B+=1


prob_1=a/X_par
prob_2=b/X_sau_Y_par
prob_3=c/suma_opt

prob_A=A/N
prob_B=B/N
prob_A_si_B=A_si_B/N

print("Probabilitatea ca suma sa fie mai mare sau egala cu 10 stiind ca X este par: ", prob_1)
print("Probabilitatea ca suma sa fie mai mare sau egala cu 10 stiind ca X sau Y este par: ", prob_2)
print("Probabilitatea ca X si Y sa fie pare stiind ca suma este 8: ", prob_3)    

diferenta = abs(prob_A_si_B - (prob_A * prob_B))
marja_eroare = 0.005

sunt_independente = diferenta < marja_eroare

print(sunt_independente)

if sunt_independente:
    print("Evenimentele A și B sunt independente.")
else:
    print("Evenimentele A și B sunt dependente.")
