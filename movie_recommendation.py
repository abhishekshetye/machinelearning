import numpy as np
import pandas as pd
import scipy as sc
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# filename = 'Reviews.csv'
# names = ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']
# data = pd.read_csv(filename, names=names)
# print(data.shape)

data = fetch_movielens(min_rating=4.0)

print(repr(data['train']))
print(repr(data['test']))


#create model 
#weighted appr rank pairwise
#content + collabarative
model = LightFM(loss='warp')
#training
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendations(model, data, user_ids):
	#generate recomm for
	#num of users and moviesw
	n_users, n_items = data['train'].shape

	for user_id in user_ids:

		#movies already liked
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movies our model predicts they will like
		scores = model.predict(user_id, np.arange(n_items))
		#sort from most liked
		top_items = data['item_labels'][np.argsort(-scores)]

		#print results
		print("User %s " % user_id)
		print("		Known positives: ")

		for x in known_positives[:3]:
			print(" %s " % x)

		print(" 	Recommended: ")

		for x in top_items[:3]:
			print("	%s " % x)


sample_recommendations(model, data, [3,24,450])




