# Dog and Cat Image Classification Using a Three-Layer Convolutional Neural Network

## 1. Project Overview

This project implements a **Convolutional Neural Network (CNN)** for binary image classification. Given an image, the model predicts whether it contains a cat or a dog. The implementation uses TensorFlow/Keras to build and train the CNN, a dataset hosted on Hugging Face to obtain labeled images, and Scikit-learn to evaluate classification performance.

The project is designed as an end-to-end deep-learning workflow. It covers dataset acquisition, image conversion, preprocessing, sample visualization, neural-network construction, training, validation, testing, metric reporting, sample inference, and saving both the trained model and test predictions.

The CNN contains **exactly three convolutional layers**, followed by pooling operations and fully connected classification layers. Its output layer uses a sigmoid activation to produce a score for the dog class.

The supplied Python script is intended to run in **Google Colab**. It installs the Hugging Face `datasets` library, downloads the dataset, trains the model, displays evaluation plots, and creates output files.

> **Important:** The code prints the actual test accuracy when it runs. This README does not claim a particular accuracy in advance because the measured result depends on the dataset version, runtime, and training behavior.

---

## 2. Project Objectives

The main objectives are to:

- Develop a CNN that distinguishes cats from dogs using image data.
- Understand the role of convolution, activation functions, and max pooling in image recognition.
- Load a labeled image dataset from Hugging Face.
- Convert supported image representations into consistent RGB images.
- Resize images to a uniform input shape suitable for a neural network.
- Split the available training data into training and validation subsets.
- Normalize image pixel values before learning.
- Train a model using binary cross-entropy and the Adam optimizer.
- Apply early stopping to help limit unnecessary training and restore the best validation-loss weights.
- Evaluate the model on a separate test split.
- Measure accuracy, precision, recall, and F1-score.
- Visualize a confusion matrix and training/validation curves.
- Demonstrate prediction on a test image.
- Save the trained model and a CSV file containing test predictions.

---

## 3. Problem Statement

Manual image classification can be time-consuming, particularly when a dataset contains many images. A computer-vision model can learn visual patterns from labeled examples and use those patterns to classify new images.

This project addresses a binary classification task:

**Input:** A color image containing a cat or a dog.  
**Output:** One of two classes—`cats` or `dogs`.

The model receives an image represented as a tensor of shape:

\[
128 \times 128 \times 3
\]

The first two dimensions represent image height and width, while the final dimension represents the red, green, and blue color channels.

The model's final sigmoid unit estimates the probability associated with the dog class. The code applies a threshold of 0.5:

- Probability greater than or equal to 0.5 → predicted class `dogs` (label `1`).
- Probability less than 0.5 → predicted class `cats` (label `0`).

This threshold is the project's default decision rule. It may not be optimal for every real-world application.

---

## 4. Dataset Description

### 4.1 Dataset source

The script loads the dataset using the Hugging Face Datasets library:

`louiecerv/cats_dogs_dataset`

The dataset is retrieved with:

```python
from datasets import load_dataset

dataset = load_dataset("louiecerv/cats_dogs_dataset")

train_data = dataset["train"]
test_data = dataset["test"]
```

The script expects the dataset to contain `train` and `test` splits, with each record providing an `image` field and a `label` field.

Dataset availability, schema, size, and contents may change. Check the dataset card on Hugging Face if loading fails or if the published dataset differs from the expected structure.

### 4.2 Class labels

The implementation defines:

```python
class_names = ["cats", "dogs"]
```

It assumes the dataset's labels use this mapping:

| Label | Class |
|---:|---|
| `0` | Cats |
| `1` | Dogs |

The label mapping should be verified against the dataset card or its class metadata before relying on the reported results. An incorrect mapping would reverse the interpretation of predictions and evaluation metrics.

### 4.3 Image preparation configuration

| Parameter | Value |
|---|---|
| Image size | 128 × 128 pixels |
| Color format | RGB |
| Input channels | 3 |
| Image data type after preparation | `uint8` |
| Batch size | 32 |
| Training/validation division | 80% / 20% of the original training split |
| Test data | Dataset's provided test split |
| Random seed | 42 |

The exact number of training, validation, and test images is printed by the program when it executes. It is not hard-coded in this README.

### 4.4 Why use a cat-and-dog dataset?

Cats and dogs share some visual characteristics, but their faces, ears, body shapes, fur patterns, and other image features can differ. This makes the task a useful introductory example for supervised computer vision.

The dataset is intended for learning and experimentation. Its performance should not be interpreted as proof that the model is reliable for all breeds, lighting conditions, image quality levels, backgrounds, or real-world deployment settings.

---

## 5. Technology Stack

| Technology | Role in the project |
|---|---|
| Python | Main programming language |
| Google Colab | Recommended execution environment |
| TensorFlow | Deep-learning framework |
| Keras | CNN definition, compilation, training, and inference |
| Hugging Face Datasets | Downloading and accessing the image dataset |
| Pillow (PIL) | Opening, converting, and resizing images |
| NumPy | Image arrays, labels, indexing, and numerical operations |
| Matplotlib | Image samples, confusion matrix, and learning curves |
| Scikit-learn | Classification metrics and confusion-matrix utilities |
| Pandas | Exporting test predictions to CSV |

### Important APIs

- `load_dataset()` — Loads the dataset from Hugging Face.
- `convert_to_pil()` — Converts supported image formats to RGB PIL images.
- `prepare_data()` — Resizes images and builds image and label arrays.
- `Sequential()` — Defines the CNN as a stack of layers.
- `Conv2D()` — Learns spatial image features.
- `MaxPooling2D()` — Reduces spatial dimensions.
- `Rescaling()` — Scales pixel values from 0–255 to 0–1.
- `Flatten()` — Converts feature maps to a one-dimensional vector.
- `Dense()` — Implements fully connected layers.
- `Dropout()` — Randomly drops a portion of activations during training.
- `EarlyStopping()` — Stops training when validation loss no longer improves.
- `model.fit()` — Trains the network.
- `model.evaluate()` — Evaluates the trained model.
- `model.predict()` — Produces predicted probabilities.
- `classification_report()` — Reports precision, recall, F1-score, and support.
- `confusion_matrix()` — Counts true and predicted class combinations.

---

## 6. Repository Structure

The supplied archive contains the following project file:

```text
cnn_dog_cat_colab_project/
├── cnn_cats_dogs_colab.py
└── README.md
```

The Python script generates these output files when execution reaches the corresponding save operations:

```text
dog_cat_cnn.keras
dog_cat_predictions.csv
```

The dataset itself is loaded at runtime from Hugging Face. The script does not require a manually prepared local image directory.

### Generated outputs

| File | Purpose |
|---|---|
| `dog_cat_cnn.keras` | Saved trained Keras model |
| `dog_cat_predictions.csv` | Actual test labels, predicted labels, and predicted dog probabilities |

The CSV and model are created in the process's current working directory. In Google Colab, that is commonly the notebook's working directory. Download or copy the files before the runtime is reset if you need to retain them.

---

## 7. System Requirements

### Recommended environment

- Google Colab or a compatible Python environment.
- Python version compatible with the installed TensorFlow release.
- Internet access to retrieve the Hugging Face dataset and install dependencies.
- Sufficient RAM to hold the prepared image arrays.
- CPU or GPU execution support.

A GPU is not strictly required, but it can reduce training time. The script converts the entire training and test splits into NumPy arrays before fitting the model. For large datasets, this can require substantial memory.

### Dependencies

The script imports:

- `datasets`
- `numpy`
- `matplotlib`
- `tensorflow`
- `Pillow`
- `scikit-learn`
- `pandas`

The script explicitly installs `datasets` with a notebook-style command. Other packages are expected to be available in the environment.

---

## 8. Installation and Setup

### Option A: Run in Google Colab

Google Colab is the intended environment for the supplied script.

1. Extract the ZIP archive on your computer.
2. Open Google Colab and create a new notebook.
3. Upload `cnn_cats_dogs_colab.py`, or copy its contents into a notebook cell.
4. Run the script in the notebook environment.
5. Allow the `datasets` package installation to complete.
6. Wait for the dataset to download and the image arrays to be prepared.
7. Monitor the CNN training output.
8. Review the metrics and plots.
9. Retrieve the generated `.keras` and `.csv` files before the runtime is deleted.

**Colab compatibility note:** The script contains `!pip install -q datasets`, which is notebook syntax. It works when run in a Colab/Jupyter code cell. It is not valid ordinary Python syntax when executed directly as `python cnn_cats_dogs_colab.py` in a terminal.

If you want to run the script locally, remove the notebook installation line and install the dependencies from the terminal first.

### Option B: Run locally

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install datasets numpy matplotlib tensorflow pillow scikit-learn pandas
```

Then edit `cnn_cats_dogs_colab.py` to remove or comment out the notebook-only line:

```python
!pip install -q datasets
```

Run the script:

```bash
python cnn_cats_dogs_colab.py
```

Ensure that the installed Python version and TensorFlow version are compatible. If TensorFlow installation fails, consult the official TensorFlow installation instructions for supported platforms and versions.

---

# 9. Implementation Details

## 9.1 Import libraries and set random seeds

The program imports the libraries required for data loading, image processing, neural-network training, evaluation, and visualization.

```python
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from datasets import load_dataset
```

It also imports Scikit-learn evaluation utilities and Keras layers, including `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, and `Rescaling`.

The script configures random seeds:

```python
np.random.seed(42)
tf.random.set_seed(42)
```

A seed helps make the random operations more reproducible. It does not guarantee identical model weights or metrics across every hardware device, library version, or execution environment.

The TensorFlow version is printed to help identify the runtime used for the experiment.

## 9.2 Load the dataset

The dataset is downloaded and loaded with:

```python
dataset = load_dataset("louiecerv/cats_dogs_dataset")

train_data = dataset["train"]
test_data = dataset["test"]
```

The script prints the number of examples in each split. This is an important initial check because dataset size may differ from expectations.

The implementation keeps the dataset's existing test split separate from the training and validation data. This supports a more meaningful final evaluation, provided the test split is not used to make training decisions.

## 9.3 Convert images to RGB format

Datasets can represent images in multiple ways depending on how they are stored and decoded. The project defines `convert_to_pil(image)` to normalize supported representations into a PIL RGB image.

The function handles:

- A PIL image object.
- A dictionary containing encoded image bytes.
- A dictionary containing a local image path.
- A NumPy array.
- A list that can be converted to an array.

For array inputs, values are clipped to the valid 8-bit range and converted to `uint8` when necessary. The result is converted to RGB so that all images have three color channels.

The function raises an exception for unsupported representations rather than silently accepting an unknown format. This makes unexpected dataset schema changes easier to diagnose.

## 9.4 Resize images and prepare labels

The `prepare_data(split)` function processes each record in a dataset split.

Its main steps are:

1. Read the record's image.
2. Convert it to RGB using `convert_to_pil()`.
3. Resize it to 128 × 128 pixels.
4. Convert the resized image to a NumPy array.
5. Read and convert the class label to an integer.
6. Stack the images into one array.
7. Convert the labels into a NumPy array.

The resizing operation uses Lanczos resampling:

```python
image = image.resize(
    (IMG_SIZE, IMG_SIZE),
    Image.Resampling.LANCZOS
)
```

Lanczos resampling is designed to preserve useful image detail when resizing. Resizing to a consistent shape is required because the CNN expects every image in a batch to have the same dimensions.

The resulting image array has the general shape:

\[
(N, 128, 128, 3)
\]

where \(N\) is the number of images in the split. The corresponding label array has shape:

\[
(N,)
\]

The script prints the shapes of the prepared arrays and the class counts for the training split.

### Memory consideration

The implementation stores the prepared training and test images in memory as NumPy arrays. This is straightforward for a small or medium-sized educational project, but may not scale well to very large datasets. A future version could use streaming, `tf.data`, or a generator-based pipeline to load and preprocess images in batches.

## 9.5 Visualize sample images

The script displays nine images from the prepared training data in a 3 × 3 grid. Each image is titled with its corresponding class label.

This visualization helps verify that:

- The images load successfully.
- The images have the expected color appearance.
- The labels appear to correspond to the displayed images.
- The preprocessing pipeline produces usable input samples.

Sample inspection is a useful data-quality check before training. It does not replace a full review of label correctness or dataset quality.

## 9.6 Build the CNN architecture

The model is built using the Keras `Sequential` API.

```python
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Rescaling(1.0 / 255),

    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
```

The architecture contains **exactly three `Conv2D` layers**.

### Layer-by-layer architecture

| Stage | Layer | Configuration | Purpose |
|---|---|---|---|
| 1 | Input | 128 × 128 × 3 | Accepts RGB images |
| 2 | Rescaling | Scale = 1/255 | Normalizes pixel values |
| 3 | Convolution | 32 filters, 3 × 3, ReLU | Learns low-level visual features |
| 4 | Max pooling | 2 × 2 | Reduces spatial dimensions |
| 5 | Convolution | 64 filters, 3 × 3, ReLU | Learns more complex visual patterns |
| 6 | Max pooling | 2 × 2 | Further reduces feature-map size |
| 7 | Convolution | 128 filters, 3 × 3, ReLU | Learns higher-level visual features |
| 8 | Max pooling | 2 × 2 | Reduces the final feature maps |
| 9 | Flatten | — | Converts feature maps into a vector |
| 10 | Dense | 128 neurons, ReLU | Learns combinations of extracted features |
| 11 | Dropout | Rate = 0.5 | Regularizes training by dropping activations |
| 12 | Output | 1 neuron, sigmoid | Estimates the probability of the dog class |

### Convolutional layers

A convolutional layer applies learnable filters across the image or feature maps. These filters can learn patterns such as edges, textures, shapes, and combinations of visual features.

The model increases the number of filters from 32 to 64 and then to 128. This gives later layers the capacity to represent a larger variety of patterns.

The `padding="same"` setting preserves the spatial width and height through each convolutional operation before pooling.

### Max-pooling layers

Each `MaxPooling2D((2, 2))` layer downsamples the feature maps by selecting the maximum value within local 2 × 2 regions. This reduces spatial dimensions and computational cost while retaining strong local activations.

For a 128 × 128 input, the spatial dimensions after the three pooling stages are approximately:

- After pooling 1: 64 × 64
- After pooling 2: 32 × 32
- After pooling 3: 16 × 16

The final convolutional output therefore has the shape 16 × 16 × 128 before flattening, assuming the input and configured layer dimensions remain unchanged.

### Flatten layer

`Flatten()` converts the final three-dimensional feature map into a one-dimensional vector so that it can be passed to a fully connected layer.

### Dense layer

The 128-neuron dense layer combines the features extracted by the convolutional layers to support the final classification decision.

### Dropout layer

The dropout rate is 0.5. During training, dropout randomly deactivates a proportion of activations in that layer. This acts as a regularization technique and may reduce overfitting. Dropout is not applied in the same way during ordinary inference.

### Sigmoid output layer

The final layer contains one neuron with sigmoid activation. It returns a value between 0 and 1, interpreted by the code as the model's score or estimated probability for the dog class.

## 9.7 Normalize pixel values

The model contains:

```python
Rescaling(1.0 / 255)
```

The original image values are represented as 8-bit integers from 0 to 255. Dividing by 255 maps these values to the range 0 to 1.

For a pixel value \(x\), the transformation is:

\[
x_{\text{scaled}} = \frac{x}{255}
\]

This normalization is applied inside the model, so the prepared images can remain as `uint8` arrays until they are passed to the network. Keeping preprocessing in the model also means the saved model contains this rescaling layer.

## 9.8 Compile the model

The model is compiled with:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Adam optimizer

Adam is a gradient-based optimization algorithm that adapts parameter updates using estimates of gradient moments. It is commonly used for neural-network training and provides a practical default for this project.

### Binary cross-entropy loss

Binary cross-entropy is suitable for a two-class problem with a sigmoid output. For a single sample, its standard form is:

\[
L = -[y\log(p) + (1-y)\log(1-p)]
\]

where:

- \(y\) is the true label, either 0 or 1.
- \(p\) is the predicted probability for Class 1.

The training process adjusts model weights to reduce the loss over the training examples.

### Accuracy metric

Accuracy measures the fraction of examples classified correctly. It is useful as a high-level measure, but should be interpreted alongside precision, recall, F1-score, and the confusion matrix.

The script prints `model.summary()` to show the layers, output shapes, and parameter counts.

## 9.9 Create a validation split

The script creates a validation subset from the dataset's training split.

```python
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_train))

validation_size = int(0.20 * len(X_train))

val_indices = indices[:validation_size]
train_indices = indices[validation_size:]
```

The generated indices are used to create:

- `X_train_final`, `y_train_final` — data used to fit the network.
- `X_val`, `y_val` — data used to monitor validation performance.

This reserves approximately 20% of the original training split for validation. The remaining approximately 80% is used for fitting.

The provided test split remains separate. It is used for final evaluation after training.

The split is reproducible under the fixed NumPy seed. However, it is not explicitly stratified, so class proportions in the training and validation subsets may differ slightly from the original split.

## 9.10 Train the CNN

The script configures early stopping:

```python
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

The model is then trained:

```python
EPOCHS = 15

history = model.fit(
    X_train_final,
    y_train_final,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)
```

### Training configuration

| Parameter | Value | Explanation |
|---|---:|---|
| Maximum epochs | 15 | Maximum number of passes through the training subset |
| Batch size | 32 | Number of images processed per training batch |
| Validation data | 20% of original training split | Used to monitor validation loss |
| Optimizer | Adam | Updates model weights |
| Loss | Binary cross-entropy | Measures binary prediction error |
| Early stopping patience | 3 epochs | Waits for improvement before stopping |
| Best-weight restoration | Enabled | Restores weights from the best observed validation loss |

Early stopping monitors `val_loss`. If it does not improve for three consecutive epochs, training stops early. With `restore_best_weights=True`, the model restores the weights from the epoch with the best monitored validation loss.

Early stopping can help limit unnecessary training, but it does not guarantee that overfitting is eliminated.

## 9.11 Evaluate on the test set

After training, the model is evaluated on the dataset's test split:

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)
```

The script reports test loss and test accuracy. This evaluation should be interpreted as the performance of the trained model on the supplied test split, not as a guarantee of performance on every unseen image.

For a fair assessment, avoid repeatedly adjusting model settings based on the test results. Use the validation split for training decisions and reserve the test split for final evaluation.

## 9.12 Generate predictions

The model predicts probabilities for the test images:

```python
y_prob = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()
```

The probabilities are converted into class labels:

```python
y_pred = (y_prob >= 0.5).astype(np.int32)
```

Here:

- `y_prob` contains the predicted dog-class scores.
- `y_pred` contains the predicted integer labels.

The threshold of 0.5 is a simple default. If the application requires different trade-offs between false positives and false negatives, the threshold should be selected using validation data and the intended application requirements.

## 9.13 Generate the classification report

The code prints a Scikit-learn classification report:

```python
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    )
)
```

The report contains the following metrics for each class.

### Precision

Precision measures how many predictions assigned to a class were correct.

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

### Recall

Recall measures how many true examples of a class were correctly identified.

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

### F1-score

The F1-score is the harmonic mean of precision and recall.

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
\]

When the denominator is zero, the metric requires a defined convention; the script uses `zero_division=0`.

### Support

Support is the number of true examples belonging to a class in the evaluation data.

The script also prints accuracy computed by Scikit-learn. This provides an additional check against the Keras evaluation output.

## 9.14 Display the confusion matrix

The confusion matrix is generated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)
```

It is visualized with:

```python
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    values_format="d"
)
```

The matrix compares actual labels against predicted labels.

| | Predicted cats | Predicted dogs |
|---|---:|---:|
| Actual cats | Correct cat predictions | Cats classified as dogs |
| Actual dogs | Dogs classified as cats | Correct dog predictions |

The diagonal cells represent correct classifications. The off-diagonal cells represent errors.

This view helps identify whether the classifier makes more mistakes on cats or on dogs. The actual values are printed by the script.

## 9.15 Plot training and validation curves

The script displays training and validation accuracy and loss over the epochs recorded in the `history` object.

The accuracy graph uses:

```python
history.history["accuracy"]
history.history["val_accuracy"]
```

The loss graph uses:

```python
history.history["loss"]
history.history["val_loss"]
```

These curves help diagnose model behavior:

- Increasing accuracy indicates that the model is correctly classifying more samples under the selected metric.
- Decreasing loss indicates improving predictions according to binary cross-entropy.
- A growing difference between training and validation performance may indicate overfitting.
- Poor performance on both training and validation data may indicate underfitting or an optimization problem.
- Early stopping can result in fewer than 15 recorded epochs.

The graphs should be interpreted alongside the test metrics and the dataset's characteristics.

## 9.16 Predict a sample test image

The code demonstrates inference using the first test image:

```python
sample_index = 0

sample_image = X_test[sample_index]
actual_label = int(y_test[sample_index])
```

It adds a batch dimension before prediction:

```python
probability = float(
    model.predict(
        sample_image[np.newaxis, ...],
        verbose=0
    )[0][0]
)
```

A single image has shape `(128, 128, 3)`, while the model expects a batch with shape `(batch_size, 128, 128, 3)`. Using `np.newaxis` changes the input shape to `(1, 128, 128, 3)`.

The predicted label is calculated using the 0.5 threshold. The code then computes the confidence associated with the selected class:

- If the predicted class is dog, confidence is the dog probability.
- If the predicted class is cat, confidence is one minus the dog probability.

The image is displayed with the actual class, predicted class, and computed confidence percentage.

**Interpretation note:** This confidence value is derived from the sigmoid output. It should not automatically be treated as a calibrated probability or as proof that a prediction is correct.

## 9.17 Save the trained model

The trained model is saved using:

```python
model.save("dog_cat_cnn.keras")
```

The `.keras` format stores the Keras model architecture and learned weights, along with relevant model configuration needed for loading the model.

The model includes the rescaling layer, so raw RGB pixel arrays with the expected shape can be passed to the saved model without separately applying the same normalization step.

A saved model can be loaded later with:

```python
import tensorflow as tf

model = tf.keras.models.load_model("dog_cat_cnn.keras")
```

The runtime must have a compatible TensorFlow/Keras installation. Saving the model does not automatically save the original dataset or the evaluation report.

## 9.18 Save test predictions to CSV

The script creates a Pandas DataFrame containing the true labels, predicted labels, and predicted dog probabilities:

```python
results = pd.DataFrame({
    "Actual_Label": y_test,
    "Predicted_Label": y_pred,
    "Dog_Probability": y_prob
})

results.to_csv(
    "dog_cat_predictions.csv",
    index=False
)
```

The CSV has three columns:

| Column | Description |
|---|---|
| `Actual_Label` | Ground-truth label from the test split |
| `Predicted_Label` | Model's predicted class, using the 0.5 threshold |
| `Dog_Probability` | Sigmoid output interpreted as the dog-class score |

The row order corresponds to the order of samples in `X_test` and `y_test`. The CSV does not include the original images or image filenames, so it is a prediction summary rather than a complete image-level audit trail.

## 9.19 Print the final experiment summary

At the end of the script, the program prints a summary containing:

- Project name.
- Dataset source and dataset identifier.
- Training, validation, and testing counts.
- Number of convolutional layers.
- Test accuracy.
- Saved model filename.
- Saved predictions filename.

This summary provides a compact record of the run and confirms which outputs the script attempted to save.

---

## 10. Model Configuration Summary

| Component | Configuration |
|---|---|
| Task | Binary image classification |
| Classes | Cats and dogs |
| Dataset | `louiecerv/cats_dogs_dataset` |
| Image size | 128 × 128 |
| Color channels | 3 (RGB) |
| Pixel scaling | 1/255 |
| Convolutional layers | 3 |
| Convolution filters | 32, 64, 128 |
| Kernel size | 3 × 3 |
| Convolution activation | ReLU |
| Pooling | Three 2 × 2 max-pooling layers |
| Flattening | `Flatten()` |
| Dense hidden layer | 128 neurons, ReLU |
| Dropout | 0.5 |
| Output layer | 1 neuron, sigmoid |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Metric | Accuracy |
| Maximum epochs | 15 |
| Batch size | 32 |
| Early stopping | Monitor `val_loss`, patience 3 |
| Decision threshold | 0.5 |
| Saved model | `dog_cat_cnn.keras` |
| Saved predictions | `dog_cat_predictions.csv` |

---

## 11. Expected Outputs

### Console output

During execution, the program prints:

- TensorFlow version.
- Dataset loading status.
- Training and testing sample counts.
- Prepared image and label array shapes.
- Class distribution.
- CNN architecture and parameter summary.
- Training and validation progress.
- Test loss and test accuracy.
- Classification report.
- Confusion matrix values.
- Actual and predicted labels for the sample image.
- Sample prediction confidence.
- Final experiment summary.

### Visual output

The script produces the following visualizations:

1. **Sample image grid:** Nine images with their class labels.
2. **Confusion matrix:** Correct and incorrect predictions for cats and dogs.
3. **Training and validation accuracy:** Accuracy trends across epochs.
4. **Training and validation loss:** Loss trends across epochs.
5. **Sample prediction image:** A test image with its actual label, predicted label, and computed confidence.

### Generated files

- `dog_cat_cnn.keras` — saved model.
- `dog_cat_predictions.csv` — test predictions and scores.

The exact accuracy, precision, recall, F1-score, confusion matrix counts, and training duration are determined at runtime. Use the values produced by your own execution when writing a results section or presentation.

---

## 12. Troubleshooting

### 12.1 Notebook installation syntax causes a syntax error

**Cause:** The script contains `!pip install -q datasets`, which is notebook syntax rather than standard Python syntax.

**Solution:** Run the script in a Google Colab/Jupyter code cell, or remove that line and install the dependency from the terminal before running the file locally.

### 12.2 Dataset loading fails

**Possible causes:**

- No internet connection.
- Temporary Hugging Face service or network issue.
- Dataset identifier is unavailable or has changed.
- The installed `datasets` library is incompatible with the runtime.
- The dataset's split names or schema differ from what the script expects.

**Solution:** Check internet connectivity, confirm the dataset identifier and card on Hugging Face, update or install a compatible `datasets` version, and inspect the returned dataset's column names and splits.

### 12.3 Missing image or label field

**Cause:** The script expects each record to include `image` and `label` fields.

**Solution:** Inspect `dataset["train"].column_names` and a sample record. If the dataset schema differs, adapt `prepare_data()` to the actual field names and verify the class-label mapping.

### 12.4 Unsupported image representation

**Cause:** A dataset image may be stored in a representation not handled by `convert_to_pil()`.

**Solution:** Inspect the type and structure of the image field. Extend the conversion function carefully to handle the actual representation, and verify that the result is a valid RGB image.

### 12.5 Out-of-memory error

**Cause:** The script loads the full training and test image splits into NumPy arrays. Memory use can be substantial.

**Possible solutions:**

- Use a runtime with more memory.
- Reduce the image resolution, understanding that this changes the experiment.
- Use a `tf.data` pipeline or generator to process batches instead of materializing the entire dataset.
- Cache preprocessed images to disk if appropriate.

### 12.6 TensorFlow installation error

**Cause:** The Python version, operating system, and TensorFlow version may be incompatible.

**Solution:** Check the official TensorFlow installation guidance and select a supported environment. Google Colab can reduce local setup difficulties.

### 12.7 Plots do not display

**Cause:** The execution environment may not support interactive plotting windows.

**Solution:** Run the project in Google Colab, Jupyter Notebook, or JupyterLab, or configure a suitable Matplotlib backend for the local environment.

### 12.8 Output files cannot be found

**Cause:** The files are saved relative to the current working directory.

**Solution:** Check the notebook's current directory or the terminal working directory. In Colab, download the output files before the runtime is reset if they need to be retained.

### 12.9 Model accuracy is unexpectedly low

Check the following:

1. Confirm the label mapping is `0 = cats` and `1 = dogs`.
2. Inspect sample images and their labels.
3. Verify that the dataset is loaded from the intended source.
4. Check training and validation curves.
5. Inspect the confusion matrix to identify class-specific errors.
6. Confirm that the image input shape is `(128, 128, 3)`.
7. Ensure the model is receiving the expected pixel values.
8. Compare the actual training and validation sample counts.
9. Check for duplicates, mislabeled samples, or dataset-specific issues if further investigation is required.

Do not assume that one accuracy value proves the model is working correctly. Review the complete evaluation output.

---

## 13. Limitations

The current implementation is an educational baseline and has several limitations:

- **Dataset dependence:** Results depend on the dataset's quality, distribution, and label accuracy.
- **Fixed input resolution:** Images are resized to 128 × 128, which can remove fine details.
- **Limited architecture search:** The model uses a fixed three-convolutional-layer architecture and does not systematically compare alternatives.
- **No transfer learning:** The CNN is trained from scratch rather than using a pretrained vision model.
- **Memory usage:** Entire image splits are loaded into memory before training.
- **Default threshold:** A 0.5 decision threshold may not be optimal for every application.
- **Confidence calibration:** The sigmoid output is not guaranteed to be a calibrated probability.
- **Generalization:** Test performance on this dataset does not guarantee performance on images from a different source or setting.
- **Limited output metadata:** The prediction CSV does not preserve image filenames or the image files themselves.
- **No deployment interface:** The script demonstrates model training and inference but does not provide a production API or deployed application.
- **Reproducibility:** Fixed seeds help, but different library versions and hardware can still produce different results.

---

## 14. Future Enhancements

The following additions could improve the project and make it more suitable for further experimentation.

### 14.1 Data augmentation

Apply controlled random transformations such as horizontal flips, small rotations, zoom, and contrast adjustments during training. Augmentation can help the model learn more robust features, provided transformations preserve the correct class.

### 14.2 Transfer learning

Compare the custom CNN with a pretrained architecture such as MobileNetV2, EfficientNet, or ResNet. Transfer learning can improve performance and reduce training time when suitable pretrained weights are available.

### 14.3 Save model evaluation artifacts

Save the confusion matrix, classification report, and training curves to files so that the results can be included in a project report.

### 14.4 Record image identifiers

Extend the preprocessing pipeline to retain image identifiers or source paths where available. This would make it easier to inspect individual misclassifications.

### 14.5 Build an interactive interface

Create a Streamlit interface that accepts an uploaded image and displays:

- The uploaded image.
- The predicted class.
- The dog-class score.
- The class confidence calculated from the threshold decision.
- A note that model predictions may be incorrect.

### 14.6 Hyperparameter tuning

Experiment with the number of filters, dense-layer size, dropout rate, learning rate, batch size, and epoch count. Use validation data to compare configurations and retain the test set for final evaluation.

### 14.7 Improve data pipelines

Use `tf.data` or another batch-based pipeline to reduce memory pressure and enable preprocessing and prefetching.

### 14.8 Add automated tests

Test image conversion, input shapes, label values, output dimensions, model loading, and CSV creation. Automated tests can help catch dataset or environment changes.

### 14.9 Evaluate additional metrics

Consider ROC-AUC, precision-recall curves, per-class error analysis, and threshold selection on validation data. These can provide a more complete view of classifier behavior than accuracy alone.

### 14.10 Model versioning

Record the dataset version, code revision, dependency versions, configuration, training date, and evaluation results for each experiment.

---

## 15. Ethical and Practical Considerations

This project is a simple image-classification demonstration. Before using a similar model in a real application:

- Verify that images are obtained and used in accordance with their licenses and terms.
- Check dataset documentation and attribution requirements.
- Evaluate the model on data representative of the intended use.
- Avoid treating model predictions as infallible.
- Consider the effects of poor image quality, unusual viewpoints, occlusion, and background bias.
- Clearly communicate the model's limitations to users.

The dataset and model are intended for educational experimentation and should not be considered a production-ready animal-identification system without further validation.

---

## 16. Conclusion

This project demonstrates a complete workflow for binary image classification using a Convolutional Neural Network with exactly three convolutional layers. It loads a labeled cat-and-dog dataset from Hugging Face, standardizes image dimensions and color representation, normalizes pixel values inside the model, and trains the network using Adam and binary cross-entropy.

The implementation includes a validation split and early stopping, followed by evaluation on a separate test split. Accuracy, a classification report, and a confusion matrix provide quantitative performance information, while sample images and learning curves help make the results easier to inspect. The trained model is saved in Keras format, and the test predictions are exported to CSV for further analysis.

The project provides a foundation for studying CNNs and computer vision. Further work could explore data augmentation, transfer learning, more efficient input pipelines, hyperparameter tuning, and a user-facing inference interface.

---

## 17. References and Documentation

Consult the official documentation for installation details, API behavior, and further learning:

- **TensorFlow:** https://www.tensorflow.org/
- **Keras:** https://keras.io/
- **Hugging Face Datasets:** https://huggingface.co/docs/datasets/
- **Hugging Face dataset page:** https://huggingface.co/datasets/louiecerv/cats_dogs_dataset
- **Scikit-learn metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html
- **Pillow documentation:** https://pillow.readthedocs.io/
- **NumPy documentation:** https://numpy.org/doc/
- **Matplotlib documentation:** https://matplotlib.org/stable/

---

## 18. Project Information

**Project title:** Dog and Cat Image Classification Using a Three-Layer CNN  
**Main implementation:** `cnn_cats_dogs_colab.py`  
**Dataset:** `louiecerv/cats_dogs_dataset`  
**Framework:** TensorFlow/Keras  
**Primary task:** Binary image classification  
**Saved model:** `dog_cat_cnn.keras`  
**Prediction output:** `dog_cat_predictions.csv`

Add the following details if required for an academic submission or public repository:

- **Author:** Your name
- **Institution:** Your college or organization
- **Course / Department:** Your course or department
- **Project date:** Date of submission
- **Repository:** Link to the source-code repository

## License

No project-specific license is included in the supplied archive. Before redistributing the source code or dataset, choose an appropriate license for your own work and verify the dataset's terms of use and attribution requirements.
