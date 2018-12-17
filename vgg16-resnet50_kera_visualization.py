from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models
from keras.preprocessing import image

#Ref: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
#Ref: https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

#this function is to allow for importing modules corresponding to the CNN model
def importModel(modelName):
    model = None
    
    if modelName == 'resnet50':
        from keras.applications.resnet50 import ResNet50
        from keras.preprocessing import image
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        from keras.preprocessing import image
        model = ResNet_50('resnet50_weights.h5')
        
    else:
        from keras.applications.vgg16 import preprocess_input
        from keras.applications.vgg16 import decode_predictions
        from keras.applications.vgg16 import VGG16
        from keras.preprocessing import image
        #model = VGG16(weights='imagenet')
        model = VGG_16('vgg16_weights.h5')

    #return the chosen model
    return model

#function defining VGG16 model
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

#helper function - resizing the input image
def resizeAndPadImg(img, desiredSz):
    #desiredW = desiredShape[1]
    #desiredH = desiredShape[0]
    
    origSz = img.shape[:2] # old_size is in (height, width) format
    ratio = float(desiredSz)/max(origSz)
    #print("ratio= " + str(ratio))
    new_size = tuple([int(x*ratio) for x in origSz]) #(183, 275, 3)
    #print("new_size= " + str(new_size))
    # new_size should be in (width, height) format
    im = cv2.resize(img, (new_size[1], new_size[0])) #(149, 224)
    #print("im= " + str(im) + " shape=" + str(im.shape))
    
    delta_w = desiredSz - new_size[1] #224-224 = 0
    delta_h = desiredSz - new_size[0] #224-149 = 76
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    #print("top= " + str(top) + " bottom=" + str(bottom))
    #print("left= " + str(left) + " right=" + str(right))

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)
    #print("new_im= " + str(new_im) + " shape=" + str(new_im.shape))
    #print("new_im[40:]= " + str(new_im[40:]))
    return new_im
   
#get the image tensor from the image files
def getImgTensor(imgPath):
    img = image.load_img(imgPath, target_size=(224, 224))
    #print("getImgTensor: img: " + str(img))
    img_tensor = image.img_to_array(img)
    #print("getImgTensor: img_tensor: " + str(img_tensor))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    #print("getImgTensor: img_tensor: " + str(img_tensor))
    # Remember that the model was trained on inputs
    # that were preprocessed in the following way:
    img_tensor /= 255.

    # get image shape
    print(img_tensor.shape)
    return img_tensor
    
# plot the images
def plotImg(imgTensor):
    plt.imshow(imgTensor[0])
    plt.show()
    


#variables to tweak which layers we want to get the intermediate activations for
visual_layer_start_index = 2
total_visual_layers_needed = 4
visual_layer_step = 2

#helper function to plot intermediate activations given the activations layers, model and imageTensor
def plotActivations(model, activations, imgTensor): #, imgTensor):
    
    # layer lists
    layer_names = []
    print("model layers=" + str(model.layers))
    
    total_layer_count = len(model.layers)
    print("model layers count=" + str(total_layer_count))
    
    visual_layer_stop_idx = visual_layer_start_index

    if ((visual_layer_start_index + total_visual_layers_needed)  < total_layer_count):
        visual_layer_stop_idx = (visual_layer_start_index + total_visual_layers_needed)
    else:
        visual_layer_stop_idx = total_layer_count
    
    layer_outputs = [layer.output for layer in model.layers[visual_layer_start_index:visual_layer_stop_idx:visual_layer_step]]
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(imgTensor)
    
    #for visualLayersIdx in range(visual_layer_start_idx, total_layer_count):
        
        #visualLayersIdx
        #print("visualLayersIdx=" + str(visualLayersIdx))
    print("total_visual_layers_needed=" + str(total_visual_layers_needed))
    print("visual_layer_start_index=" + str(visual_layer_start_index))
    print("visual_layer_stop_idx=" + str(visual_layer_stop_idx))

    for layer in model.layers[visual_layer_start_index:visual_layer_stop_idx:visual_layer_step]:
        layer_names.append(layer.name)

        images_per_row = 16

    # Now let's display our feature maps
    index = 1
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        print("layer_activation-shape=" + str(layer_activation.shape))
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        print("n_features=" + str(n_features))
        print("size=" + str(size))
        print("n_cols=" + str(n_cols))
        print("display_grid=" + str(display_grid))
        
        # We'll tile each filter into this big horizontal grid
        imgIndex = 1
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                #print("channel_image.std()=" + str(channel_image.std()))
                if 0 != channel_image.std():
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        fileName = "activation_maps-layer" + str(index) + ".jpg"
        print("fileName=" + str(fileName))
        index += 1
        plt.savefig(fileName)
        
#get activations for the layers
def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units)
    
    
def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications import VGG16
from keras import backend as K

#function to visualize the activation heatmap
#write the activation heatmap to a file
def visualizeHeatMap(imgPath, model):
    
    #K.clear_session()
    K.set_learning_phase(False)
    # Note that we are including the densely-connected classifier on top
    # all previous times, we were discarding it.
    #model = VGG16('vgg16_weights.h5')#VGG16(weights='imagenet')
    
    #global graph
    #graph = tf.get_default_graph()
    
    # `img` is a PIL image of size 224x224
    if True:
        img = cv2.imread(imgPath)
    
        #print("img= " + str(img) + " shape=" + str(img.shape))
        im = resizeAndPadImg(img, desiredSz)
        #print("img= " + str(im) + " shape=" + str(im.shape))
    
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        #print("im= " + str(im) + " shape=" + str(im.shape))
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        #im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        #print("im= " + str(im) + " shape=" + str(im.shape))
    
        # Test pretrained model
        #model = VGG_16('vgg16_weights.h5')
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(optimizer=sgd, loss='categorical_crossentropy')
        #im = preprocess_input(im)
        
        print(model.summary())
        #with graph.as_default():
        preds = model.predict(im)
            
    else :
        img = image.load_img(imgPath, target_size=(224, 224))

        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = image.img_to_array(img)

        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)

        # Finally we preprocess the batch
        # (this does channel-wise color normalization)
        x = preprocess_input(x)
    
        preds = model.predict(x)
        
    print("preds=" + str( np.argmax(preds) ))
    print('Predicted:', decode_predictions(preds, top=3)[0])
    
    objectClass = np.argmax(preds[0])
    print("objectClass=" + str( objectClass )) 

    # prediction vector
    fashionObjOutput = model.output[:, objectClass]

    # The is the output feature map of the `conv2d_13` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('conv2d_13')
    print('last_conv_layer=',last_conv_layer)

    # This is the gradient of the "fashionObjOutput" class with regard to
    # the output feature map of `conv2d_13`
    grads = K.gradients(fashionObjOutput, last_conv_layer.output)[0]
    print('grads=',grads)

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    print('pooled_grads=',pooled_grads)

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `conv2d_13`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    print('iterate=',iterate)
    
    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([im])
    print('pooled_grads_value=',pooled_grads_value)
    print('conv_layer_output_value=',conv_layer_output_value)
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    #normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    
    #superimpose heatmap with the originla image
    img = cv2.imread(imgPath)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img

    # Save the image to disk
    cv2.imwrite('./fashionObjectImage_withHeatMap.jpg', superimposed_img)
    
desiredSz = 224
imgPath = 'testData/Abstract_Print_Draped_Blazer_img_00000050.jpg'

#function to visualize intermediate activations
def visualizeActivations(model):
    img = cv2.imread(imgPath)
    
    #print("img= " + str(img) + " shape=" + str(img.shape))
    im = resizeAndPadImg(img, desiredSz)
    #print("img= " + str(im) + " shape=" + str(im.shape))
    
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    #print("im= " + str(im) + " shape=" + str(im.shape))
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    #im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    #print("im= " + str(im) + " shape=" + str(im.shape))

    # Test pretrained model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    plot_model(model, to_file='vgg.png')
    print(model.summary())
    yhat = model.predict(im)
    print("np.argmax=" + str( np.argmax(yhat) ))
    
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('\n %s (%.2f%%)' % (label[1], label[2]*100))
    
    #plotImg
    imgTensor = getImgTensor(imgPath)
    print("imgTensor(shape)=" + str(imgTensor.shape))
    #plotImg(imgTensor)
    
    #Activation approach#1
    #plot activations
    # Extracts the outputs of the top 8 layers:
    layer_outputs = [layer.output for layer in model.layers[:8]]
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(imgTensor)
    #print("activations=" + str(activations))
    
    #sample activations check
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)
    plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
    plt.savefig('sampleActivatn.jpg')
    
    #Activation across layers
    plotActivations(model, activations, imgTensor)

if __name__ == "__main__":
    
    # Test pretrained model
    #global model
    K.clear_session()
    K.set_learning_phase(False)
    
    #choose the modelName to change between vgg16 and resnet
    modelName = 'vgg16'
    
    model = importModel(modelName)
    
    #Visualization#1: Intermediate Activations
    visualizeActivations(model)
    
    #Visualization#2: Heatmap Activations
    visualizeHeatMap(imgPath, model)
