import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def run(image_dir, img_size=(128,128), batch_size=32, epochs=5, lr=0.0001):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        image_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", subset="training")
    val_gen = datagen.flow_from_directory(
        image_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", subset="validation", shuffle=False)

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    for layer in base_model.layers: layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(train_gen.num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)

    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes

    report = classification_report(y_true, y_pred_classes, target_names=list(val_gen.class_indices.keys()), output_dict=True)
    cm = confusion_matrix(y_true, y_pred_classes)

    # Generate Grad-CAM for first few images
    gradcam_paths = []
    for i in range(min(3, len(val_gen))):  # just 3 samples
        img, label = val_gen[i][0][0], val_gen[i][1][0]
        heatmap = make_gradcam_heatmap(np.expand_dims(img, axis=0), model, base_model)
        overlay = apply_gradcam(img, heatmap)
        path = f"gradcam_{i}.png"
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        gradcam_paths.append(path)

    results = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_indices": val_gen.class_indices,
        "classification_report_labels": {
            "true_labels": y_true.tolist(),
            "pred_labels": y_pred_classes.tolist()
        }
    }
    return results, history.history, gradcam_paths

# Grad-CAM helpers
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name="block5_conv3"):
    grad_model = Model([model.inputs], [base_model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def apply_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor((img*255).astype("uint8"), cv2.COLOR_RGB2BGR), 1, heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
