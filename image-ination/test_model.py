# Imports
import keras
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import load_img
from IPython.display import display
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

#trained_model = 'model.82-0.97-0.96.hdf5'
#trained_model = 'model.238-0.98-0.94.hdf5'
trained_model = 'model.357-0.98-0.98.hdf5'



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

test_batchsize = 16   
image_size = 256      
test_dir = 'datasets/madhacks/train'

def test_datagenerator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                              target_size=(image_size, image_size),
                              batch_size=test_batchsize,
                              class_mode='categorical',
                              shuffle=False)
 
    return test_dir, test_generator


def test():
    test_dir1, test_generator = test_datagenerator()

    print('loading trained model...')
   # new_model = keras.models.load_model('trained_models/model.241-0.87-0.73.hdf5')
    new_model = keras.models.load_model('trained_models/'+trained_model)
    print('loading complete')

    print('summary of loaded model')
    new_model.summary()

    #ground_truth = test_generator.classes

    print('predicting on the test images...')

    prediction_start = time.clock()
    predictions = new_model.predict_generator(test_generator,
                                              use_multiprocessing=True,
                                              workers=18,
                                              steps=test_generator.samples / test_generator.batch_size,
                                              verbose=1)
                                              
    

    prediction_finish = time.clock()
    prediction_time = prediction_finish - prediction_start
    predicted_classes = np.argmax(predictions, axis=1)
    print(predictions)
    print(predicted_classes)
    print('Confusion Matrix')
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())   
    confusion_matrices = confusion_matrix(test_generator.classes, predicted_classes)
    print(confusion_matrices)
    
    print('Classification Report')
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)    
       
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_generator.classes, predicted_classes)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Area Under Curve ',auc_keras)
    
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('ROC Curve')
    
    errors = np.where(predicted_classes != true_classes)[0]
    print("No. of errors = {}/{}".format(len(errors), test_generator.samples))

    correct_predictions = np.where(predicted_classes == true_classes)[0]
    print("No. of correct predictions = {}/{}".format(len(correct_predictions), test_generator.samples))

    print("Test Accuracy = {0:.2f}%".format(len(correct_predictions)*100/test_generator.samples))
    print("Predicted in {0:.3f} minutes!".format(prediction_time/60))

def Main():
    test()                 

if __name__ == '__main__':
    Main()
