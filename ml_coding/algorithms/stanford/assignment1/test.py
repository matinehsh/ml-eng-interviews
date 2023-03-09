import time
from math import log
N = 600000000

t0 = time.time()
for i in range(int(log(N,2))):
	a = 1
t1 = time.time()
print(t1-t0)

for i in range(N):
	a = 1
t2 = time.time()
print(t2-t1)

for i in range(N*N):
	a = 1
t3 = time.time()
print(t3-t2)

