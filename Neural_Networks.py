import pandas as pd
import tensorflow as tf
import numpy as np
import warnings
from sklearn.metrics import auc, classification_report, precision_recall_curve
warnings.filterwarnings('ignore')

yTrain = pd.read_csv('yTrain.csv').head(100000).values
yTest = pd.read_csv('yTest.csv').head(100000).values
yTrain = np.asarray(yTrain).astype('float32').ravel()
yTest = np.asarray(yTest).astype('float32').ravel()

def handle_data(dataset):
    xTrain = pd.read_csv(f'{dataset}_df_train.csv').head(100000).values
    xTest = pd.read_csv(f'{dataset}_df_test.csv').head(100000).values
    xTrain = np.asarray(xTrain).astype('float32')
    xTest = np.asarray(xTest).astype('float32')
    return xTrain, xTest

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

def Dense(xTrain, xTest, yTrain, yTest, dataset, shape, hlaf, epoch, bs, lr):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=hlaf, input_shape=(shape,)))
    model.add(tf.keras.layers.Dense(64, activation=hlaf))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model.fit(xTrain, yTrain, epochs=epoch, batch_size=bs, callbacks=[early_stopping], validation_split=0.1)
    loss, f1 = model.evaluate(xTest, yTest)
    print(f"Dense Neural Networks test F1-score for {dataset} dataset: {f1}")
    y_pred_probs = model.predict(xTest).ravel()
    precision, recall, thresholds = precision_recall_curve(yTest, y_pred_probs)
    auprc = auc(recall, precision)
    print(f"Dense Neural Networks AUPRC for {dataset} dataset: {auprc}")
    y_pred_classes = np.where(y_pred_probs > 0.5, 1, 0) 
    print("Classification Report:")
    print(classification_report(yTest, y_pred_classes))

def RNN(xTrain, xTest, yTrain, yTest, dataset, shape, unit, bs, patience):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(units=unit, return_sequences=True, input_shape=(shape,1)))
    model.add(tf.keras.layers.SimpleRNN(units=unit))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(xTrain, yTrain, epochs=5, batch_size=bs, callbacks=[early_stopping], validation_split=0.1)
    loss, f1 = model.evaluate(xTest, yTest)
    print(f"Recurrent Neural Networks (RNN) test F1-score for {dataset} dataset: {f1}")
    y_pred_probs = model.predict(xTest).ravel()
    precision, recall, thresholds = precision_recall_curve(yTest, y_pred_probs)
    auprc = auc(recall, precision)
    print(f"Recurrent Neural Networks (RNN) AUPRC for {dataset} dataset: {auprc}")
    y_pred_classes = np.where(y_pred_probs > 0.5, 1, 0) 
    print("Classification Report:")
    print(classification_report(yTest, y_pred_classes))

"""
# Binary Dataset
binary_df_train, binary_df_test = handle_data('binary')
Dense(binary_df_train, binary_df_test, yTrain, yTest, 'binary', binary_df_train.shape[1],'relu', 26, 32, 0.01)
RNN(binary_df_train, binary_df_test, yTrain, yTest, 'binary', binary_df_train.shape[1], 30, 1024, 3)

# Bag of Words Dataset
bow_df_train, bow_df_test = handle_data('bow')
Dense(bow_df_train, bow_df_test, yTrain, yTest, 'bow', bow_df_train.shape[1], 'relu', 15, 16, 0.01)
RNN(bow_df_train, bow_df_test, yTrain, yTest, 'bow', bow_df_train.shape[1], 30, 1024, 3)

# Tfidf Dataset
tfidf_df_train, tfidf_df_test = handle_data('tfidf')
Dense(tfidf_df_train, tfidf_df_test, yTrain, yTest, 'tfidf', tfidf_df_train.shape[1],'relu', 25, 8, 0.05)
RNN(tfidf_df_train, tfidf_df_test, yTrain, yTest, 'tfidf', tfidf_df_train.shape[1], 20, 512, 4)
"""
# Hashing Dataset
hash_df_train, hash_df_test = handle_data('hash')
Dense(hash_df_train, hash_df_test, yTrain, yTest, 'hashing', hash_df_train.shape[1],'relu', 22, 2, 0.05)
RNN(hash_df_train, hash_df_test, yTrain, yTest, 'hashing', hash_df_train.shape[1], 50, 1024, 2)
"""
# LDA Dataset
lda_df_train, lda_df_test = handle_data('lda')
Dense(lda_df_train, lda_df_test, yTrain, yTest, 'LDA', lda_df_train.shape[1],'relu', 29, 2, 0.05)
RNN(lda_df_train, lda_df_test, yTrain, yTest, 'LDA', lda_df_train.shape[1], 50, 512, 4)

# PCA Dataset
pca_df_train, pca_df_test = handle_data('pca')
Dense(pca_df_train, pca_df_test, yTrain, yTest, 'PCA', pca_df_train.shape[1], 'relu', 30, 2, 0.05)
RNN(pca_df_train, pca_df_test, yTrain, yTest, 'PCA', pca_df_train.shape[1], 50, 512, 4)
"""