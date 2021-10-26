from flask import Flask, url_for, send_from_directory, request
import logging, os, json
from werkzeug import secure_filename
import os.path
from os import path
import os
#import cx_Oracle

#os.environ['TNS_ADMIN'] = '<Wallet Location>'
#connection = cx_Oracle.connect('ARSHAD', 'dsafdsgfdsf!!', 'instancename_medium')
#cursor = connection.cursor()

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))

#trained_model = 'model.238-0.98-0.94.hdf5'
#trained_model = 'model.238-0.98-0.94.hdf5'
trained_model = 'model.357-0.98-0.98.hdf5'

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
import base64
import shutil

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model
import os
from stat import S_IREAD
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from flask import render_template


test_batchsize = 16
image_size = 256
uploads_dir = 'uploads'


#Create new folder
def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        print(newpath)
        os.makedirs(newpath)
    return newpath

#Method takes an image and respond back the result with base 64 encoding of image
@app.route('/upload', methods=['POST'])
def api_root():
    try:
        #cursor ,connection = createOracleConnection()
        #cursor = ''
        #connection = ''
        app.logger.info(PROJECT_HOME)
        if request.method == 'POST' and request.files['image']:
            app.logger.info(request)
            params = request.values

            operator = params.get('Operator')
            patientId = params.get('PatientId')
            actual = params.get('Actual')

            print('Patient Id ',patientId)
            print('Operator',operator)

            UPLOAD_FOLDER = ('{}/uploads/' + operator + '/').format(PROJECT_HOME)
            app.logger.info(UPLOAD_FOLDER)
            img = request.files['image']
            img_name = secure_filename(img.filename)
            extensions = ['jpg','jpeg','png']
            print('Estrings',len(img_name.split('.')))
            print('Estddfrings', img_name.split('.')[len(img_name.split('.'))-1])
            res = [ele for ele in extensions if(ele == img_name.split('.')[len(img_name.split('.'))-1])]
            print('Extension', bool(res))
            if bool(res) != True:
                print('Arshad')
                data = {}
                data['null'] = ['Not a valid file only supports .jpg, .jpeg and .png']
                return json.dumps(data, indent=4) 

            create_new_folder(UPLOAD_FOLDER)
            saved_path = os.path.join(UPLOAD_FOLDER, img_name)
            app.logger.info("saving {}".format(saved_path))
            img.save(saved_path)
            filesize= os.path.getsize(saved_path)
            if filesize > 5242880:
                shutil.rmtree(UPLOAD_FOLDER)
                data = {}
                data['null'] = ['Upoaded file should be 5M or less']
                return json.dumps(data, indent=4) 
            os.chmod(saved_path, S_IREAD)
            isAnamoly = anamolyDetector(saved_path)
            if isAnamoly:
              #strHTMLError = '<html><html><head> <title></title> <style type="text/css"> body{background-image: url('cancer-cells.jpg'); background-size: cover}.aa{font-family: monospace; width: 300px; height: 320px; background-color: rgba(0,0,0,0.5); margin: 0 auto; margin-top: 50px; padding-top: 10px;padding-bottom: 10px; padding-left: 50px; padding-right: 50px; border-radius: 15px; color: white; font-weight: bolder; box-shadow: inset -4px -4px rgba(0,0,0,0.5)}</style> </head> <body><div class="aa"> <h2><center>Uploaded Image is not valid Mammography<center></h2><h1><center>Please upload a valid image</center></h1></body> <center> <input type="image" src="error.png" width="150" height="150"></center> </div></html>'
              #data = {}
              #data['null'] = ['Uploaded Image is not valid Mammography']
              return render_template("error.html")#json.dumps(data, indent=4) 
            predicted_classes = predict(operator, patientId,actual, img_name,UPLOAD_FOLDER)
            return predicted_classes
        else:
            return "Please upload an image"
    finally:
        app.logger.info("Inside finally block to delete upload folder")
        if path.exists(UPLOAD_FOLDER):
          #shutil.rmtree()
          os.system('rmdir /S /Q "{}"'.format(UPLOAD_FOLDER))

          app.logger.info("Removed Upload folder")
        else: 
          app.logger.info("Upload folder already deleted")
   
import tensorflow as tf
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    keras.backend.set_session(session)
    app.logger.info('Model loaded {}'.format(trained_model))
    new_model = load_model('trained_models/'+trained_model)
    app.logger.info('Loading complete')

def test_datagenerator(test_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(image_size, image_size),
                                                      batch_size=test_batchsize,
                                                      class_mode='categorical',
                                                      shuffle=False)

    return test_dir, test_generator





def predict(operator, patientId, actual,img_name,UPLOAD_FOLDER):
    global sess
    global graph
    app.logger.info("Prediction for Operator {}, patientId{}, Image {}".format(operator,patientId,img_name))
    test_dir1, test_generator = test_datagenerator(uploads_dir)
    prediction_start = time.clock()
    with session.graph.as_default():
        keras.backend.set_session(session)
        predictions = new_model.predict_generator(test_generator,
                                                  steps=test_generator.samples / test_generator.batch_size,
                                                  verbose=1)

    prediction_finish = time.clock()
    prediction_time = prediction_finish - prediction_start
    predicted_classes = np.argmax(predictions, axis=1)
    filenames = test_generator.filenames
    base64img = ''
    with open(UPLOAD_FOLDER+img_name, mode='rb') as file:
    	img = file.read()
    	base64img = base64.encodebytes(img).decode("utf-8")
    #shutil.rmtree(UPLOAD_FOLDER)
    data = {}
    
    predicted_class = ''
    for (a, b, c) in zip(filenames, predicted_classes, predictions):
        app.logger.info ('File Name {}, Screened Result {}, Prediction {}'.format(a, b, c))
        if b == 0:
            predicted_class = 'Screened Negative'
        if b == 1:
            predicted_class = 'Screened Positive'
        data['null'] = [img_name, predicted_class, base64img,operator]
        #Uncomment to insert into DB
        '''sql = ('insert into breast_cancer_prediction( patient_id,img_name,actual, prediction_result, operator_id) '
        'values(:patientId,:img_name,:actual,:predicted_class,:operator)')
        rs = cursor.execute(sql, [patientId,img_name, actual,predicted_class,operator])
        connection.commit()'''

    app.logger.info("Predicted in {0:.3f} minutes!".format(prediction_time / 60))
    
    #In case of HTML response
    strhtml='<html><head><style>  body{background-color:RosyBrown;}#customers {font-family: "Lucida Console", Monaco, monospace;border-collapse: collapse;border-radius: 2em;overflow: hidden;width:80%;height:45%;margin-top: 150px;margin-right: 150px;margin-left:150px;}#customers td, #customers th {border: 1px solid #ddd;padding: 8px;}#customers tr:nth-child(even){background-color: LavenderBlush;}#customers tr:hover {background-color: #ddd;}#customers th {padding-top: 12px;padding-bottom: 12px;text-align: left;background-color: DeepSkyBlue;color: white;}</style></head><body><h3></h3><p></p><table id="customers" ><tr><th>OPERATOR_NAME</th><th>PATIENT_ID</th><th>IMAGE_NAME</th><th>PREDICTED_CLASS</th><th>IMAGE</th></tr><tr><td>'+operator+'</td><td>'+patientId+'</td><td>'+img_name+'</td><td>'+predicted_class+'</td><td><img src="data:image/jpeg;base64,'+base64img+'"></td></tr></table></body></html>'



    return strhtml #json.dumps(data, indent=4)



#Method detects of uploaded image is not the breast memography
def IsImageHasAnomaly(autoencoder, filePath,threshold):  
    im = cv2.resize(cv2.imread(filePath), (420, 420))
    im = im * 1./255
    datas = np.zeros((1,  420, 420, 3))
    validation_image = np.zeros((1,  420, 420, 3),np.float32)
    validation_image[0, :, :, :] = im;   
    #print(validation_image[0].shape)
    predicted_image = autoencoder.predict(validation_image)
    #predicted_image = predicted_image.astype(np.float32)
    #validation_image = cv2.cvtColor(validation_image[0], cv2.COLOR_BGR2GRAY)
    #predicted_image = cv2.cvtColor(predicted_image[0], cv2.COLOR_BGR2GRAY)
    #validation_image = tf.image.rgb_to_grayscale(validation_image, name=None)
    #predicted_image = tf.image.rgb_to_grayscale(predicted_image, name=None)
    #print(validation_image.shape)
    #mssim,score = compare_ssim(predicted_image, validation_image,full=True,multichannel=True) 
    _mse = mse(predicted_image[0], validation_image[0]) 
    #print(_mse)
    #print(score)
    #print('_mse: {}'.format(score))
    return _mse  > threshold
    
    
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
    
def anamolyDetector(inputImage):
    img_width, img_height = 420, 420
    input_img = Input(batch_shape=(None, img_width, img_width, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    #autoencoder.summary()
    autoencoder.load_weights('trained_models/anomaly-detection.h5');    
    threshold=0.01
    isAnamoly = IsImageHasAnomaly(autoencoder, inputImage,threshold)
    
    return isAnamoly 


if __name__ == '__main__':
    #Used waitress for prodction mode
    from waitress import serve
    #serve(app,host='0.0.0.0', threads=10)
    #For development mode
    app.run(host='0.0.0.0',port='5001', debug=True, threaded=True)
    