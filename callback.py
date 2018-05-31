import keras
from sklearn.metrics import roc_auc_score

class roc_curve_area(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        roc_curve_area = roc_auc_score(self.validation_data[1], y_pred)
        
        print("Roc curve area: ", roc_curve_area)