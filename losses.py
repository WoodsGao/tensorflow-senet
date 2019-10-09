from tensorflow.keras.losses import Loss


class FocalLoss(Loss):
    def __init__(self, loss_fun):
        pass

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.square(y_pred - y_true), axis=-1)