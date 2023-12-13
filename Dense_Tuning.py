from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#activation function
#learning rate
#batch size
#epoch


Y = pd.read_csv('yTrain.csv').head(10000).values
Y = np.asarray(Y).astype('float32').ravel()

def handle_data(dataset):
    xTrain = pd.read_csv(f'{dataset}_df_train.csv').head(10000).values
    xTrain = np.asarray(xTrain).astype('float32')
    return xTrain

def f1_score(y_true, y_pred):
    # Convert probabilities to binary predictions
    y_pred_binary = tf.keras.backend.round(y_pred)
    tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred_binary, 'float'), axis=0)
    fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred_binary, 'float'), axis=0)
    fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred_binary), 'float'), axis=0)

    # Calculate precision and recall
    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())

    # Calculate F1 score
    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.keras.backend.mean(f1)

#xFeat = handle_data('binary')
#xFeat = handle_data('bow')
#xFeat = handle_data('tfidf')
#xFeat = handle_data('hash')
#xFeat = handle_data('lda')
xFeat = handle_data('pca')
"""
# hidden layer af
hid_act_func = ['relu', 'tanh', 'elu', 'swish', 'selu', 'softplus']
hid_act_f1 = []
highest = 0
best_func = ""
kf = KFold(n_splits=5)
for af in hid_act_func:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=af, input_shape=(xFeat.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation=af))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    f_1 = 0
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model.fit(xTrain, yTrain, epochs=5, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
        loss, f1 = model.evaluate(xTest, yTest)
        f_1 += f1
    func_f1 = f_1/5
    if func_f1 > highest:
        highest = func_f1
        best_func = af
    hid_act_f1.append(func_f1)
    print('Finished: ', af)
print("The best activation function is:", best_func)
plt.figure(figsize=(10, 6))
plt.bar(hid_act_func, hid_act_f1)
plt.xlabel('Hidden Layer Activation Functions')
plt.ylabel('F-1 Score')
plt.title(f'Dense Neural Networks 5-Fold Cross-Validation on PCA Dataset for Hidden Layer Activation Functions')
plt.ylim(0.85, 0.92)
plt.show()

# epoch
epochs = [i for i in range(1, 31)]
epoch_f1 = []
highest = 0
best_ep = 0
kf = KFold(n_splits=5)
for ep in epochs:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(xFeat.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    f_1 = 0
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model.fit(xTrain, yTrain, epochs=ep, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
        loss, f1 = model.evaluate(xTest, yTest)
        f_1 += f1
    ep_f1 = f_1/5
    if ep_f1 > highest:
        highest = ep_f1
        best_ep = ep
    epoch_f1.append(ep_f1)
    print('Finished: ', ep)
print("The best epoch is:", best_ep)
plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_f1)
plt.xlabel('Number of Epochs')
plt.ylabel('F1-Score')
plt.title(f'Dense Neural Networks 5-Fold Cross-Validation on PCA Dataset for Epochs')
plt.show()

# batch size
bs = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048']
batch_f1 = []
highest = 0
best_bs = 0
kf = KFold(n_splits=5)
for batch in bs:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(xFeat.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    f_1 = 0
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model.fit(xTrain, yTrain, epochs=5, batch_size=int(batch), callbacks=[early_stopping], validation_split=0.1)
        loss, f1 = model.evaluate(xTest, yTest)
        f_1 += f1
    bs_f1 = f_1/5
    if bs_f1 > highest:
        highest = bs_f1
        best_bs = batch
    batch_f1.append(bs_f1)
    print('Finished: ', batch)
print("The best batch size is:", best_bs)
plt.figure(figsize=(10, 6))
plt.bar(bs, batch_f1)
plt.xlabel('Batch Size')
plt.ylabel('F-1 Score')
plt.title(f'Dense Neural Networks 5-Fold Cross-Validation on PCA Dataset for Batch Size')
plt.ylim(0.8, 1.0)
plt.show()
"""
# learning rate
lrs = ['1', '0.1', '0.5', '0.01', '0.05', '0.001', '0.005', '0.0001', '0.0005', '0.00001', '0.00005', '0.000001']
learn_f1 = []
highest = 0
best_lr = 0
kf = KFold(n_splits=5)
for lr in lrs:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(xFeat.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    f_1 = 0
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], Y[train], Y[test]
        model.fit(xTrain, yTrain, epochs=5, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
        loss, f1 = model.evaluate(xTest, yTest)
        f_1 += f1
    lr_f1 = f_1/5
    if lr_f1 > highest:
        highest = lr_f1
        best_lr = lr
    learn_f1.append(lr_f1)
    print('Finished: ', lr)
print("The best learning rate is:", best_lr)
plt.figure(figsize=(10, 6))
plt.bar(lrs, learn_f1)
plt.xlabel('Learning Rate')
plt.ylabel('F-1 Score')
plt.title(f'Dense Neural Networks 5-Fold Cross-Validation on PCA Dataset for Learning Rate')
plt.show()
