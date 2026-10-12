import random

prob_spam=0.03
p_legitim=1-prob_spam

p_oferta_in_spam=0.8
p_oferta_in_legitim=0.15

p_oferta=prob_spam*p_oferta_in_spam+p_legitim*p_oferta_in_legitim #formula probabilitatii totale
p_spam_oferta=(prob_spam*p_oferta_in_spam)/p_oferta #formula lui Bayes

#1.
print(f"Probabilitatea ca un email sa fie spam stiind ca contine cuvantul oferta: {p_spam_oferta:.4f}")

p_gratuit_in_spam=0.60
p_gratuit_in_legitim=0.10

p_both_in_spam=p_oferta_in_spam*p_gratuit_in_spam
p_both_in_legitim=p_oferta_in_legitim*p_gratuit_in_legitim

P_both=(p_both_in_spam*prob_spam)+(p_both_in_legitim*p_legitim)
p_spam_both=(p_both_in_spam*prob_spam)/P_both #probabilitatea sa fie in spam stiind ca contine cuvintele oferta si gratuit

#2.
print(f"Probabilitatea ca un email sa fie spam stiind ca contine cuvintele oferta si gratuit: {p_spam_both:.4f}")

N=100000
total_oferta=0
spam_oferta=0
total_both=0
spam_both=0

for i in range(N):
    span=random.random()<prob_spam
    if span:
        oferta=random.random()<p_oferta_in_spam
        gratuit=random.random()<p_gratuit_in_spam
    else:
        oferta=random.random()<p_oferta_in_legitim
        gratuit=random.random()<p_gratuit_in_legitim
    if oferta:
        total_oferta+=1
        if span:
            spam_oferta+=1
    if oferta and gratuit:
        total_both+=1
        if span:
            spam_both+=1

sim_spam_oferta = spam_oferta / total_oferta
sim_spam_both = spam_both / total_both

print(f"{sim_spam_oferta:.4f}")
print(f"{sim_spam_both:.4f}")

#3.
"""
    Ipoteza independentei conditionate este nerealista in aceste caz, deoarece, alegerea cuvintelor "oferta" si "gratuit" in acelasi timp este mai probabila in cazul unui email de spam. In acest caz, folosirea cuvantului "oferta" creste probabilitatea ca si cuvantul "gratuit" sa fie folosit, deci cele doua evenimente nu sunt independente.
"""

#4.
"""
    O regula de clasificare a emailurilor: daca un email este foarte porbabil sa fie spam( P(spam|cuvinte_alese)> 90%  ), atunci acesta este clasificat ca spam. In caz contrar, emailul este clasificat ca legitim.
"""