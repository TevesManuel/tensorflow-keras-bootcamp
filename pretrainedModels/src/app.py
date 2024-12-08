import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy
import keras

import cv2

def resNetDemostration(filepath):
    model = keras.applications.resnet50.ResNet50()

    image = keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = numpy.expand_dims(image_array, axis=0)  # Add dimension
    image_array = keras.applications.resnet50.preprocess_input(image_array)

    predictions = model.predict(image_array)

    print("[i]ResNet50:")

    decoded_predictions = keras.applications.resnet50.decode_predictions(predictions, top=5)
    for i, (_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")

def vgg16Demostration(filepath):
    model = keras.applications.VGG16()

    image = keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = numpy.expand_dims(image_array, axis=0)  # Add dimension
    image_array = keras.applications.vgg16.preprocess_input(image_array)

    predictions = model.predict(image_array)

    print("[i]VGG16:")

    decoded_predictions = keras.applications.vgg16.decode_predictions(predictions, top=5)
    for i, (_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")

def vgg19Demostration(filepath):
    model = keras.applications.VGG19()

    image = keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = numpy.expand_dims(image_array, axis=0)  # Add dimension
    image_array = keras.applications.vgg19.preprocess_input(image_array)

    predictions = model.predict(image_array)

    print("[i]VGG19:")

    decoded_predictions = keras.applications.vgg19.decode_predictions(predictions, top=5)
    for i, (_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")

def inceptionV3Demostration(filepath):
    model = keras.applications.InceptionV3()

    image = keras.preprocessing.image.load_img(filepath, target_size=(299, 299))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = numpy.expand_dims(image_array, axis=0)
    image_array = keras.applications.inception_v3.preprocess_input(image_array)

    predictions = model.predict(image_array)

    print("[i]InceptionV3:")

    decoded_predictions = keras.applications.inception_v3.decode_predictions(predictions, top=5)
    for i, (_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")

def efficientNetV2B0Demostration(filepath):
    model = keras.applications.EfficientNetV2B0()

    image = keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = numpy.expand_dims(image_array, axis=0)
    image_array = keras.applications.efficientnet_v2.preprocess_input(image_array)

    predictions = model.predict(image_array)

    print("[i]EfficientNetV2B0:")

    decoded_predictions = keras.applications.efficientnet_v2.decode_predictions(predictions, top=5)
    for i, (_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")


def efficientNetV2LDemostration(filepath):
    model = keras.applications.EfficientNetV2L()

    image = keras.preprocessing.image.load_img(filepath, target_size=(480, 480))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = numpy.expand_dims(image_array, axis=0)
    image_array = keras.applications.efficientnet_v2.preprocess_input(image_array)

    predictions = model.predict(image_array)

    print("[i]EfficientNetV2L:")

    decoded_predictions = keras.applications.efficientnet_v2.decode_predictions(predictions, top=5)
    for i, (_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")

if __name__ == '__main__':
    running = True
    while running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo acceder a la cámara")
            exit()

        while True:
            ret, frame = cap.read()

            if ret:
                cv2.imshow("Foto", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:#Escape to exit
                    running = False
                    break

                if key == 32:#Space to screenshot
                    cv2.imwrite("./camCapture.png", frame)
                    print("Foto capturada y guardada")
                    break

        cap.release()
        cv2.destroyAllWindows()

        if running:    
            filepath = "./camCapture.png"
            resNetDemostration(filepath)
            vgg16Demostration(filepath)
            vgg19Demostration(filepath)
            inceptionV3Demostration(filepath)
            efficientNetV2B0Demostration(filepath)
            efficientNetV2LDemostration(filepath)