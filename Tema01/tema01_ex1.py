import csv
import random
import numpy as np
n=4

l=[]
file = open('students.csv', mode='r',encoding='utf-8')
reader=csv.reader(file)
for row in reader:
    if row:
        l.append(row[0])
file.close()

chosen=np.random.choice(l, size=n, replace=False)
print(f"Studentii alesi sunt: {chosen}")
