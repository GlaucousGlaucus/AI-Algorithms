delicious(cakes).
delicious(pickles).
delicious(biryani).
spicy(pickles).
likes(priya,coffee).

likes(priya,Y):- delicious(Y).
likes(prakash,Y):- delicious(Y),spicy(Y).

