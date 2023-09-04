import tensorflow as tf
import os
from PIL import Image
import glob
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)


def resize_images_with_glob(root_folder, target_size):         # resizing all the images from 400 pixel to 224 pixel
    output_root_folder = "resized_images/resized_images_e_h"

    for class_folder in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_folder)

        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_root_folder, class_folder)
            os.makedirs(output_class_path, exist_ok=True)

            for sub_folder in os.listdir(class_path):
                sub_folder_path = os.path.join(class_path, sub_folder)

                if os.path.isdir(sub_folder_path):
                    output_sub_folder_path = os.path.join(output_class_path, sub_folder)
                    os.makedirs(output_sub_folder_path, exist_ok=True)

                    image_files = glob.glob(os.path.join(sub_folder_path, "*.jpg"))  
                    for image_path in image_files:
                        try:
                            image = Image.open(image_path)
                            resized_image = image.resize(target_size, Image.ANTIALIAS)

                            filename = os.path.basename(image_path)
                            output_path = os.path.join(output_sub_folder_path, filename)
                            resized_image.save(output_path)

                            print(f"Resized {filename} saved successfully.")
                        except Exception as e:
                            print(f"Error resizing {image_path}: {e}")

target_size = (224, 224)  

# determining the root folder
root_folder = "partition_images/from_e_h_data"  # This place changes depending on the name of the data.


# Shrink and save all images
resize_images_with_glob(root_folder, target_size)



train_path= "resized_images/resized_images_e_h/train"
validation_path= "resized_images/resized_images_e_h/validation"
test_path= "resized_images/resized_images_e_h/test"

folders= glob.glob('resized_images/resized_images_e_h/train/*')
print(len(folders))


# Data augmentation
train_datagen=ImageDataGenerator(
    rescale= 1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)




validation_datagen=ImageDataGenerator(
    rescale= 1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)

train_set=train_datagen.flow_from_directory(train_path,
                                            target_size= (224,224),
                                            batch_size=32,
                                            class_mode='categorical')

validation_set=validation_datagen.flow_from_directory(validation_path,
                                            target_size= (224,224),
                                            batch_size=32,
                                            class_mode='categorical')

stopping= tf.keras.callbacks.EarlyStopping(monitor='val_loss',

                                min_delta=0.9,
                                patience=4,
                                verbose=0,
                                mode='auto',
                                baseline=None,
                                restore_best_weights=True)



base_model = MobileNet(weights='imagenet', include_top=False)

# Adding the custom class classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
predictions = Dense(len(folders), activation='softmax')(x)

# Create model
modelmobile = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

modelmobile.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

callback=[stopping]

epoch= 10

steps_per_epoch = train_set.samples // train_set.batch_size
validation_steps = validation_set.samples // validation_set.batch_size
# Train model
epochs = 10  # number of epoch
history_mobile = modelmobile.fit(
    train_set,
    epochs= epoch,
    steps_per_epoch= steps_per_epoch,
    validation_data= validation_set,
    validation_steps= validation_steps,
    callbacks= callback,
    verbose=1
)

test_generator = validation_datagen.flow_from_directory(   # model succes and loss evaluate
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
test_loss, test_acc = modelmobile.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print("Test accuracy :", test_acc)
print("Test loss:", test_loss)