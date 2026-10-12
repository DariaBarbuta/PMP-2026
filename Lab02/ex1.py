"""
P(B)=0.01
sesibilitate: P(pozitiv|B)=0.95
specificitate: P(negativ|~B)=0.90

P(B|pozitiv)=(P(pozitiv|B)*P(B))/P(pozitiv)

P(pozitiv)=P(pozitiv|B)*P(B)+P(pozitiv|~B)*P(~B)`

P(pozitiv|~B)=1-specificitate=0.10
P(~B)=1-P(B)=0.99

P(pozitiv)=0.95*0.01+0.10*0.99=0.0095+0.099=0.1085

P(B|pozitiv)=(0.95*0.01)/0.1085=0.0876


"""
import numpy as np
import random

p_b=0.01
p_notB=1-p_b#=0.99
sensitivity=0.95
specificity=0.90

p_positive=sensitivity*p_b+(1-specificity)*p_notB
p_b_given_positive=(sensitivity*p_b)/p_positive
print(f"P(B|pozitiv)={p_b_given_positive}")

N=10000
positives=0
true_positives=0
for i in range(N):
    sick=random.random()<p_b
    if sick:
        test_positive=random.random()<sensitivity
    else:
        test_positive=random.random()<(1-specificity)
    if test_positive:
        positives+=1
        if sick:
            true_positives+=1

if positives > 0:
    print(true_positives / positives)
