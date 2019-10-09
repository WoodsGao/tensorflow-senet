from cv_utils.simple_dataloader import simple_dataloader
from cv_utils.processors import Resize
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from model import SENet
import os

device = 'cpu'
if 'COLAB_TPU_ADDR' in os.environ:
    device = 'tpu'


def train(train_dir, val_dir):
    train_inputs, train_targets = simple_dataloader(
        train_dir, processor_list=[Resize((64, 64))])
    val_inputs, val_targets = simple_dataloader(
        val_dir, processor_list=[Resize((64, 64))])
    train_targets = to_categorical(train_targets)
    val_targets = to_categorical(val_targets)
    model = SENet(train_inputs.shape[1:], train_targets.shape[1])
    if device == 'tpu':
        TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        tf.logging.set_verbosity(tf.logging.INFO)
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        x=train_inputs,
        y=train_targets,
        batch_size=128,
        epochs=1,
        validation_data=(val_inputs, val_targets),
    )
    if device == 'tpu':
        model = model.sync_to_cpu()
    model.save('senet')


if __name__ == "__main__":
    train_dir = 'data/road_mark/train'
    val_dir = 'data/road_mark/val'
    train(train_dir, val_dir)