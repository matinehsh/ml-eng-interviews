from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
X = [['apple', 1,0.1], 
	 ['pear',  0,0.2], 
	 ['apple', 2,0.1],
	 ['orange',1,0.2],
	 ['pear',  1,0.1]]
enc.fit(X1)
print(enc.categories_)
a = enc.transform([['pear',1,0.1], ['apple',0,0.2], ['pear',5,0.1]]).toarray()
print(a)

#####################
enc = OneHotEncoder(handle_unknown='ignore')
X1 = [['apple'], ['pear'], ['apple'], ['orange'], ['pear']]
enc.fit(X1)
print(enc.categories_)
a = enc.transform([['pear'], ['apple'], ['orange'], ['peach'], ['peach']]).toarray()
print(a) # unknown category will be assigned vector of all zeros