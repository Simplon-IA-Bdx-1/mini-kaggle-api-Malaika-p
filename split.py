from pandas import read_csv
from sklearn.model_selection import train_test_split

data = read_csv('cs-training.csv', index_col=0)

train, test = train_test_split(data, test_size=0.30, random_state=42)

train.to_csv(r'train2.csv')
test.to_csv(r'test2.csv')