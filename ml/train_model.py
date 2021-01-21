import numpy
# import matplotlib.pyplot as plt
import pandas

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 2])
    return numpy.array(dataX), numpy.array(dataY)

numpy.random.seed(7)

dataframe = pandas.read_csv('../csv/test.csv', engine='python') 
dataset = dataframe.values

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67) 
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

look_back = 3
trainX, trainY = create_dataset(train, look_back)  
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, 3))
testX = numpy.reshape(testX, (testX.shape[0],look_back, 3))

model = Sequential()
model.add(LSTM(4, input_shape=(look_back,3)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history= model.fit(trainX, trainY,validation_split=0.33, nb_epoch=200, batch_size=32)
model.save('test.h5')
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('pérdida')
# plt.xlabel('época')
# plt.legend(['entrenamiento', 'validación'], loc='upper right')
# plt.show()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict_extended = np.zeros((len(trainPredict),3))
# Put the predictions there
trainPredict_extended[:,2] = trainPredict
# Inverse transform it and select the 3rd column.
trainPredict = scaler.inverse_transform(trainPredict_extended)[:,2]