from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Activation, Dropout
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Sequential
from keras.applications import VGG16, ResNet50
from keras.applications.densenet import DenseNet201
from keras import backend as K
from keras import optimizers
import os
import numpy as np
import errno
from matplotlib import pyplot as plt
import time
import tensorflow as tf
from sklearn.utils import class_weight

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# test >> normal = 62 | abnormal - 34
# training >> normal =  145 | abnormal - 81

# NN Parameters
image_size = 256   
train_batchsize = 8 
epochs = 500
model_dir = 'trained_models'
csv_logger_file = 'epoch_logger.csv'
# Image Dataset Directory
train_dir = 'datasets/madhacks/train'

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# Train datagenerator
def train_datagenerator(train_batchsize):
    
    train_datagen = ImageDataGenerator(
              rescale=1 / 255.0,
              rotation_range=20,
              zoom_range=0.05,
              width_shift_range=0.05,
              height_shift_range=0.05,
              shear_range=0.05,
              horizontal_flip=True,
              fill_mode="nearest",
              validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                target_size=(image_size, image_size),
                                batch_size=train_batchsize,
                                class_mode='categorical',
                                subset='training')

   
    return train_generator

# Train datagenerator
def validation_datagenerator(train_batchsize):
    
    train_datagen = ImageDataGenerator(
              rescale=1 / 255.0,
              rotation_range=5,
              zoom_range=0.05,
              width_shift_range=0.05,
              height_shift_range=0.05,
              shear_range=0.05,
              horizontal_flip=True,
              fill_mode="nearest",
              validation_split=0.2)

    
    validation_generator  = train_datagen.flow_from_directory(train_dir,
                                target_size=(image_size, image_size),
                                batch_size=train_batchsize,
                                class_mode='categorical',
                                subset='validation'
                                )
    return validation_generator


def vgg16_finetuned():

  vgg_conv = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(image_size, image_size, 3))

  for layer in vgg_conv.layers[:-2]:
    layer.trainable = True

  model = Sequential()
  model.add(vgg_conv)
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(3, activation= 'sigmoid'))

  return model

def residual_network_tuned():

  resnet_conv = ResNet50(
            include_top=False,
            input_shape=(image_size, image_size, 3))

  for layer in resnet_conv.layers[:-1]:
    layer.trainable = True

  model = Sequential()
  model.add(resnet_conv)
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='sigmoid'))

  return model
  
def densenet_tuned():

  resnet_conv = DenseNet201(
            include_top=False,
            input_shape=(image_size, image_size, 3))

  for layer in resnet_conv.layers[:-1]:
    layer.trainable = True

  model = Sequential()
  model.add(resnet_conv)
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='sigmoid'))

  return model


def custome_architecture():
    kernel_size = (8,8)
    model = Sequential()

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
        padding='valid',
        strides=4,
        input_shape=(image_size, image_size, 3)))
    model.add(Activation('relu'))


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2,2)))


    kernel_size = (16,16)
    model.add(Conv2D(64, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))


    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)


    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
    




def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def train(model):
    train_generator = train_datagenerator(train_batchsize)
    validation_generator = validation_datagenerator(train_batchsize)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['acc'])
    train_start = time.clock()
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                  cooldown=0,
                                  patience=5,
                                  min_lr=0.5e-6)
                                  
    model_checkpoint = ModelCheckpoint(model_dir+'/model.{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5', 
                                        monitor='acc', 
                                        verbose=1, 
                                        save_best_only=True, 
                                        save_weights_only=False, 
                                        mode='auto', period=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', 
                                            min_delta=0, 
                                            patience=0, 
                                            verbose=0, 
                                            mode='auto', 
                                            baseline=None,
                                             restore_best_weights=False)
    
    csv_logger = CSVLogger(csv_logger_file, separator=',', append=False)
    callbacks = [lr_reducer, lr_scheduler, model_checkpoint, csv_logger]
    
    true_classes = train_generator.classes
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(true_classes),
                                                 true_classes)
                                                 
    
                                                 
    class_weights = dict(enumerate(class_weights))
    print('Started training...')
    history = model.fit_generator(train_generator,
 				  #use_multiprocessing=True,
                   #               workers=18,
                                  steps_per_epoch=train_generator.samples / train_generator.batch_size,
                                  validation_data = validation_generator, 
                                  validation_steps = validation_generator.samples // validation_generator.batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  class_weight=class_weights,
                                  callbacks=callbacks)

    train_finish = time.clock()
    train_time = train_finish - train_start
    print('Training completed in {0:.3f} minutes!'.format(train_time / 60))

    #print('Saving the trained model...')
    #model.save('trained_models/model.h5')
    #print("Saved trained model in 'traned_models/ folder'!")

    return model, history


def show_graphs(history):
    
    train_acc = history.history['acc']
    train_loss = history.history['loss']
    
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    

    epochs1 = range(len(train_acc))

    plt.plot(epochs1, train_acc, 'b', label='Training acc')
    plt.title('Training accuracy')
    plt.legend()
    #plt.savefig('Training accuracy')

    #plt.figure()

    plt.plot(epochs1, train_loss, 'r', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    
    plt.plot(epochs1, val_acc, 'g', label='Validation acc')
    plt.title('Validation loss')
    plt.legend()
    
    plt.plot(epochs1, val_loss, 'c', label='Validation loss')
    plt.title('Validation loss')
    plt.legend()
    
    plt.savefig('Training Result')
    plt.show()


def Main():

    #create a convolutional autoencoder
    #autoencoder  = make_convolutional_autoencoder()
    #X_test_decoded = autoencoder.predict(X_test_noisy)
    #print(X_test_decoded)
    model = residual_network_tuned()

    print("epochs, train_batchsize", epochs, train_batchsize)
    _, history = train(model)  

    #show_graphs(history)

if __name__ == '__main__':
    Main()
