import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

look_back = 50 # number of input points network needs to predict the next
batch_size = 128
epochs = 70 # number of training rounds (recommended 50-100)
prediction_length = 1000 # number of points you want the network to predict 

# creating target function
t = np.arange(0, 1000, 1)
dataset = np.cos(t * 2 * np.pi * 1/200) * np.cos(t * 2 * np.pi * 1/100)

# creates the training data
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back):
		dataX.append(dataset[i : (i + look_back)])
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

# configures traininga nd testing data
train_size = int(len(dataset) * 0.67)
test_size = int(len(dataset)) - train_size
train, test = dataset[:train_size], dataset[train_size : len(dataset)]
train = np.expand_dims(train, 2)
test = np.expand_dims(test, 0)
trainX, trainY = create_dataset(train, look_back)

model = Sequential()
model.add(LSTM(128, input_shape=(None, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# training model
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

t = np.arange(0, prediction_length, 1)
x = np.cos(t * 2 * np.pi * 1/200) * np.cos(t * 2 * np.pi * 1/100) 
pred_dataset = list(x[:look_back])

i = 0
while(len(pred_dataset) < prediction_length):
	input_data = np.array(pred_dataset[i : look_back + i])
	input_data = np.expand_dims(input_data, 2)
	input_data = np.expand_dims(input_data, 0)
	predicted_next = model.predict(input_data)[0][0]
	pred_dataset.append(predicted_next)
	i += 1

plt.plot(t, x, label='ground truth', color='r')
plt.plot(t, pred_dataset,label='predictions', color='b')
plt.ylim(-1.1, 1.1)
plt.legend()
plt.show()



