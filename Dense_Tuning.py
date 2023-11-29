from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#number of layers
#activation function
#optimizer
#learning rate
#batch size
#epoch
#regularization

Y = pd.read_csv('yTrain.csv').head(1000000).values
Y = np.asarray(Y).astype('float32').ravel()

def handle_data(dataset):
    xTrain = pd.read_csv(f'{dataset}_df_train.csv').head(1000000).values
    xTrain = np.asarray(xTrain).astype('float32')
    return xTrain

xFeat = handle_data('binary')
#xFeat = handle_data('bow')
#xFeat = handle_data('tfidf')
#xFeat = handle_data('hash')
#xFeat = handle_data('lda')
#xFeat = handle_data('pca')
"""
# hidden layer af
hid_act_func = ['relu', 'tanh', 'elu', 'swish', 'selu', 'softplus']
hid_act_acc = []
highest = 0
best_func = ""
kf = KFold(n_splits=5)
for af in hid_act_func:
    acc = 0
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation=af, input_shape=(xFeat.shape[1],)))
        model.add(tf.keras.layers.Dense(64, activation=af))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        model.fit(xTrain, yTrain, epochs=5, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
        loss, accuracy = model.evaluate(xTest, yTest)
        acc += accuracy
    func_acc = acc/5
    if func_acc > highest:
        highest = func_acc
        best_func = af
    hid_act_acc.append(func_acc)
    print('Finished: ', af)
print("The best activation function is:", best_func)
plt.figure(figsize=(10, 6))
plt.bar(hid_act_func, hid_act_acc)
plt.xlabel('Hidden Layer Activation Functions')
plt.ylabel('Accuracy')
plt.title(f'Dense Neural Networks 10-Fold Cross-Validation on PCA Dataset for Hidden Layer Activation Functions')
plt.ylim(0.92, 0.96)
plt.show()
"""
# epoch
epochs = [i for i in range(1, 21)]
epoch_acc = []
highest = 0
best_ep = 0
kf = KFold(n_splits=5)
for ep in epochs:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(xFeat.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    acc = 0
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model.fit(xTrain, yTrain, epochs=ep, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
        loss, accuracy = model.evaluate(xTest, yTest)
        acc += accuracy
    ep_acc = acc/5
    if ep_acc > highest:
        highest = ep_acc
        best_ep = ep
    epoch_acc.append(ep_acc)
    print('Finished: ', ep)
print("The best epoch is:", best_ep)
plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title(f'Dense Neural Networks 10-Fold Cross-Validation on Binary Dataset for Epochs')
plt.show()
"""
# batch size
bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
bs_acc = []
kf = KFold(n_splits=5)
for batch in bs:
    for train, test in kf.split(xFeat):
        acc = 0
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(xFeat.shape[1],)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        model.fit(xTrain, yTrain, epochs=5, batch_size=batch, callbacks=[early_stopping], validation_split=0.1)
        loss, accuracy = model.evaluate(xTest, yTest)
        acc += accuracy
    bs_acc.append(acc/5)
plt.figure(figsize=(10, 6))
plt.plot(bs, bs_acc, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title(f'Dense Neural Networks 10-Fold Cross-Validation on Binary Dataset for Batch Size')
plt.show()

# output layer af
out_act_func = ['sigmoid', 'softmax', 'linear']
out_act_acc = []
highest = 0
best_func = ""
kf = KFold(n_splits=5)
for af in out_act_func:
    acc = 0
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(xFeat.shape[1],)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation=af))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        model.fit(xTrain, yTrain, epochs=5, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
        loss, accuracy = model.evaluate(xTest, yTest)
        acc += accuracy
    func_acc = acc/5
    if func_acc > highest:
        highest = func_acc
        best_func = af
    out_act_acc.append(func_acc)
    print('Finished: ', af)
print("The best activation function is:", best_func)
plt.figure(figsize=(10, 6))
plt.bar(out_act_func, out_act_acc)
plt.xlabel('Output Layer Activation Functions')
plt.ylabel('Accuracy')
plt.title(f'Dense Neural Networks 10-Fold Cross-Validation on PCA Dataset for Output Layer Activation Functions')
plt.show()
"""