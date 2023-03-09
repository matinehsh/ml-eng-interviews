import numpy as np

# create data
x1 = np.random.randn(150,2) + [-3,-3]
x2 = np.random.randn(150,2) 
x3 = np.random.randn(200,2) + [-3,1]
X = np.vstack((x1, x2, x3)) # (500,2)
np.random.shuffle(X)
n, d = X.shape
k = 3

# create init centroids
cinds = np.random.choice(n, k, replace=False)
c = X[cinds] # (3,2)

# update centroid
diff = np.linalg.norm(c)
step = 0
while diff > 0.001:
    dists = np.sqrt(np.sum((X - c[:, np.newaxis, :])**2, axis=2)) # (3,500)
    closest = np.argmin(dists, axis=0) # (500,)
    last_c = c.copy()
    for i in range(k):
        c[i,:] = X[np.where(closest==i)[0]].mean(axis=0)
    diff = np.linalg.norm(c - last_c)
    print(step, diff)
    step += 1