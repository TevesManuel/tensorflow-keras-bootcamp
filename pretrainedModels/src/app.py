import numpy
import keras

filepath = "./c.png"

image = keras.preprocessing.image.load_img(filepath, target_size=(224, 224))

### RESNET50

model = keras.applications.resnet50.ResNet50(
                                                                    include_top=True,
                                                                    weights='imagenet',
                                                                    input_tensor=None,
                                                                    input_shape=None,
                                                                    pooling=None,
                                                                    classes=1000,
                                                                    classifier_activation='softmax',
                                                                )

image_array = keras.preprocessing.image.img_to_array(image)
image_array = numpy.expand_dims(image_array, axis=0)  # Add dimension
image_array = keras.applications.resnet50.preprocess_input(image_array)

predictions = model.predict(image_array)

print("[i]ResNet50:")

decoded_predictions = keras.applications.resnet50.decode_predictions(predictions, top=5)
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

### VGG16

model = keras.applications.VGG16()

image_array = keras.preprocessing.image.img_to_array(image)
image_array = numpy.expand_dims(image_array, axis=0)  # Add dimension
image_array = keras.applications.vgg16.preprocess_input(image_array)

predictions = model.predict(image_array)

print("[i]VGG16:")

decoded_predictions = keras.applications.vgg16.decode_predictions(predictions, top=5)
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

### INCEPTION_V3

model = keras.applications.InceptionV3()

image = keras.preprocessing.image.load_img(filepath, target_size=(299, 229))
image_array = keras.preprocessing.image.img_to_array(image)
image_array = numpy.expand_dims(image_array, axis=0)
print(f"Forma del tensor final: {image_array.shape}")
image_array = keras.applications.inception_v3.preprocess_input(image_array)
print(f"Forma del tensor final: {image_array.shape}")


predictions = model.predict(image_array)

print("[i]InceptionV3:")

decoded_predictions = keras.applications.inception_v3.decode_predictions(predictions, top=5)
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")