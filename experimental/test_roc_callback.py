import numpy as np

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(Callback):
    def __init__(self, interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.roc = 0.0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            if len(self.validation_data) == 5:
                y_pred = self.model.predict([self.validation_data[0], self.validation_data[1]],
                                            verbose=0)
                score = roc_auc_score(self.validation_data[2], y_pred)
            else:
                y_pred = self.model.predict(self.validation_data[0], verbose=0)
                score = roc_auc_score(self.validation_data[1], y_pred)

            print('ROC AUC score on validation data - epoch: {:d} - ROC AUC: {:.6f}'
                  .format(epoch + 1, score))
            if score > self.roc:
                print('Best ROC score increased from {:.6f} to {:.6f}\n'
                      .format(self.roc, score))
                self.roc = score
            else:
                print('ROC score did not improve\n')


model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

data = np.array(np.random.randn(1000, 5))
labels = np.array(np.random.random_sample((1000, 3)) > 0.5)

ROC_callback = RocAucEvaluation(interval=1)
model.fit(data, labels, epochs=5, batch_size=100, validation_split=0.2, callbacks=[ROC_callback])
