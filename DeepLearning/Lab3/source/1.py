import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from tensorboardcolab import *
tbc=TensorBoardColab()

#Variables
dataset=np.loadtxt("cars.csv", delimiter=",")
x=dataset[:,0:5]
y=dataset[:,5]
y=np.reshape(y, (-1,1))
scaler = MinMaxScaler()
print(scaler.fit(x))
print(scaler.fit(y))
xscale=scaler.transform(x)
yscale=scaler.transform(y)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
#model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
#model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.summary()
opt=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0 )
#opt=Adam(lr=0.01)
#opt=Adagrad(lr=0.0001, epsilon=None, decay=0.0)
#opt=Adagrad(lr=0.01)
#opt=SGD(lr=0.0001)
#opt=SGD(lr=0.01)
#opt=RMSprop(lr=0.0001)
#opt=RMSprop(lr=0.01)
model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'])
history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2, callbacks=[TensorBoardColabCallback(tbc)]))
#history = model.fit(X_train, y_train, epochs=150, batch_size=128,  verbose=1, validation_split=0.2, callbacks=[TensorBoardColabCallback(tbc)])
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#Prediction
Xnew = np.array([[40, 0, 26, 9000, 8000]])
ynew=model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))