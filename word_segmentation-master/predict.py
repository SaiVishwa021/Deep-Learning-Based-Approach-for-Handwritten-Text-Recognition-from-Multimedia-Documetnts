from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
from check import segment
from test import correct_spelling
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Lambda, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def size(filepath):
# Load the image
 img = cv2.imread(filepath)

# Desired width and height
 desired_width = 1200
 desired_height = 900

# Get the current dimensions of the image
 height, width, _ = img.shape

# Calculate the scaling factors for width and height
 width_scale = desired_width / width
 height_scale = desired_height / height

# Choose the minimum scaling factor to ensure the image fits within the desired dimensions
 min_scale = min(width_scale, height_scale)

# Calculate the new dimensions while maintaining aspect ratio
 new_width = int(width * min_scale)
 new_height = int(height * min_scale)

# Create a white canvas of the desired size
 resized_image = np.full((desired_height, desired_width, 3), 255, dtype=np.uint8)

# Calculate the position to paste the resized image to center it on the canvas
 x_offset = (desired_width - new_width) // 2
 y_offset = (desired_height - new_height) // 2

# Paste the resized image onto the canvas
 resized_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = cv2.resize(img, (new_width, new_height))
 return resized_image

def bright_image(image_path, brightness_factor=1.5):
    
    
    # Load thymage
    
    
    image=size(image_path)
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Enhance text edges using sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened_img = cv2.filter2D(blurred_img, -1, kernel)

    # Apply adaptive thresholding with a larger block size
    adaptive_threshold_img = cv2.adaptiveThreshold(
        sharpened_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )

    # Remove small noise regions using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned_img = cv2.morphologyEx(adaptive_threshold_img, cv2.MORPH_CLOSE, kernel)

    # Find contours to locate text regions
    contours, _ = cv2.findContours(cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the text regions
    mask = np.zeros_like(gray_img)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        mask[y:y+h, x:x+w] = 255
    
    text_regions = cv2.bitwise_and(image, image, mask=mask)

    brightened_image = cv2.convertScaleAbs(text_regions, alpha=brightness_factor, beta=0)


    return segment(brightened_image)

def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape

#     _, img = cv2.threshold(img,
#                            128,
#                            255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize
    img = img / 255

    return img

def Model1():
    # input with shape of height=32 and width=128
    inputs = Input(shape=(32,128,1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)

    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)
    # act_model.summary()

    return act_model,outputs,inputs



char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

act_model,outputs,inputs=Model1()

act_model.load_weights(r'C:\Users\SAI VISHWA\Desktop\word-segment-final\word_segmentation-master\sgdo-115339r-50e-89887t-9977v.hdf5')


        
def predict(filepath):
    
    
    i=bright_image(filepath)
    recognized_words1= []
    for j in range(i):
        your_image_path = 'C:/Users/SAI VISHWA/Desktop/word-segment-final/word_segmentation-master/segmented/segment'+ str(j) + ".png"
        your_image = cv2.imread(your_image_path, cv2.IMREAD_GRAYSCALE)

        # Process the image using the process_image function
        processed_image = process_image(your_image)

        # Expand dimensions to match the model's input shape
        processed_image = np.expand_dims(processed_image, axis=0)

        # Make predictions on your processed image
        prediction = act_model.predict(processed_image)

        
        
        # Use CTC decoder
        decoded = K.ctc_decode(prediction,
                               input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0]

        out = K.get_value(decoded)

        # # Extract and print the recognized text
        recognized_text = ''
        for _, x in enumerate(out):
            for p in x:
                if int(p) != -1:
                    char = char_list[int(p)]
                    if char not in (',', '.'):
                      recognized_text += char
                     
        # recognized_text=correct_spelling(recognized_text)
        recognized_words1.append(recognized_text)

        result_text = ' '.join(recognized_words1)
        result_text=correct_spelling(result_text)
    return result_text
    
# predict('C:/Users/Pavi/Downloads/word_segmentation/word_segmentation-master/test20.jpg')