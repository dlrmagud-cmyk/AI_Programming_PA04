# PA04: AI-Generated vs. Real Image Classification

This project examines how training dataset size and class imbalance affect the classification of AI-generated and real images. Models were trained with [Google Teachable Machine](https://teachablemachine.withgoogle.com/), exported as Keras models, and evaluated in Python.

## 1. Research Questions

1. **Dataset size:** How does classification performance change as the number of uploaded training images increases under balanced class proportions?
2. **Class imbalance:** How does classification performance change when class proportions vary while the total number of uploaded training images remains fixed?

## 2. Materials in This Repository

- Five exported models, each with `keras_model.h5` and its corresponding `labels.txt`.
- Python code for image classification and held-out test evaluation.
- Five CSV files containing image-level predictions.
- `manifest.csv`, documenting image assignments and original filenames.
- Dataset preparation instructions.
- `requirements.txt`, specifying the evaluation dependencies.

Refer to the actual filenames and locations in this repository when setting the paths in the evaluation script. No particular folder structure is assumed by this README.

## 3. Download and Prepare the Dataset

### Step 1 — Download the original images

Download the dataset from Kaggle:

**[AI Generated Images vs Real Images — Kaggle](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)**

A Kaggle account may be required. Extract the downloaded files and identify the AI-generated and real-image classes. Class labels in this project follow the dataset annotations; image provenance was not independently verified.

The images are not redistributed in this repository because redistribution permissions have not been established. Users should obtain them from the original source and comply with the applicable dataset terms.

### Step 2 — Reconstruct the experimental image groups

Use `manifest.csv` and the accompanying dataset preparation instructions to match original filenames to the experimental filenames and subsets. **Do not create a new random split** when reproducing the reported evaluation.

The source archives used in this project contained 411 AI-generated and 396 real images. One exact duplicate was excluded from the AI class: `image34.jpg` duplicated `image33.jpg`. This left 410 AI-generated and 396 real images.

During preparation, images were converted to RGB JPEG without resizing. Filenames were standardized, and the original-to-experimental filename mapping was retained in the manifest. Use the documented preparation procedure to reproduce these transformations; using source images with different orientation, encoding, or processing may change predictions.

The common held-out test set contains:

| Actual class | Test images |
|---|---:|
| AI-generated | 80 |
| Real | 80 |
| **Total** | **160** |

Prepare the test directory as follows:

```text
01_TEST_DO_NOT_TRAIN/
├── AI-generated/
│   └── [80 assigned test images]
└── Real/
    └── [80 assigned test images]
```

These 160 images must not be uploaded for training. Every model is evaluated on the same test images. This external test set is separate from any validation split used internally by Teachable Machine.

### Step 3 — Reconstruct the training conditions

| Condition | AI-generated | Real | Total uploaded |
|---|---:|---:|---:|
| Small | 50 | 50 | 100 |
| Balanced | 100 | 100 | 200 |
| Large | 200 | 200 | 400 |
| AI Majority | 160 | 40 | 200 |
| Real Majority | 40 | 160 | 200 |

Small, Balanced, and Large address the dataset-size question. Real Majority, Balanced, and AI Majority address the class-imbalance question. Balanced is a shared reference, not a sixth independently trained model.

For each class, the balanced subsets are nested: Small is included in Balanced, and Balanced is included in Large. Use the manifest to preserve the exact membership of all conditions.

The table reports images uploaded to Teachable Machine, not necessarily the number used for gradient updates after its internal validation split.

## 4. Reproduce the Reported Evaluation

**Use the supplied exported models to reproduce the reported predictions. Retraining is not required for this step.**

### Step 1 — Prepare Python

The intended compatibility setup is Python 3.10 with TensorFlow 2.12.1, Keras 2.12.0, and NumPy 1.24.3. Install the dependencies listed in the repository:

```bash
python -m pip install -r requirements.txt
```

The evaluation code also uses Pillow. Modules such as `pathlib` and `csv` are part of Python's standard library.

Select the same environment in your editor or terminal before running the script. On Windows, TensorFlow may also require the Microsoft Visual C++ Redistributable (x64).

These version pins describe the intended compatibility environment, not a complete independently verified record of every installed package in the original execution environment.

### Step 2 — Set the evaluation paths

Open the uploaded evaluation script and edit its path settings to point to:

1. The selected model's `keras_model.h5`.
2. That model's own `labels.txt`.
3. The common test directory containing `AI-generated` and `Real`.
4. A unique output CSV path for that model.

In the evaluation code developed for this project, these settings are named:

```python
MODEL_PATH
LABELS_PATH
IMAGE_PATH
RESULT_CSV_PATH
```

If the uploaded version uses different names, follow the equivalent path settings in that file. To evaluate the entire test set, specify the test directory rather than an individual image.

Use the exact model filename. The supplied file is named `keras_model.h5`; capitalization can matter on some systems.

### Step 3 — Run the script

Run the evaluation script using the selected Python environment, either from your editor or from a terminal:

```bash
python "<evaluation-script-filename>.py"
```

Replace the placeholder with the actual filename in this repository. This command is illustrative; it is not a literal filename or a command-line interface for passing model arguments.

The script should report the test image count, overall accuracy, class-wise recall, and confusion matrix, and save the image-level predictions.

### Step 4 — Repeat for all five models

Change the model, label, and output paths for each condition. Keep the test directory unchanged. Use separate output filenames, for example:

```text
small_predictions.csv
balanced_predictions.csv
large_predictions.csv
ai_majority_predictions.csv
real_majority_predictions.csv
```

Never overwrite a model's results with those of another condition.

### Input preprocessing

The evaluation follows the Teachable Machine Keras export example:

1. Convert the image to RGB.
2. Use `ImageOps.fit` with LANCZOS resampling to resize and center-crop to 224 × 224 pixels.
3. Convert pixel values to `float32`.
4. Normalize using `image_array / 127.5 - 1.0`.
5. Predict using the exported model and select the highest-scoring class.

Read the output order from each model's own `labels.txt`. The Small model's labels are `0 ai` and `1 real`; do not assume every model has the same output ordering without checking its label file.

Model scores are not assumed to be calibrated probabilities.

## 5. Recreate Training in Teachable Machine

To repeat the training procedure rather than only evaluate the saved models:

1. Open [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Create a new image project with AI-generated and real classes.
3. Upload the exact training subsets for one condition, as listed in the manifest.
4. Apply the training settings below.
5. Train the model independently, rather than continuing from another condition's model.
6. Export the TensorFlow/Keras model and save both the model and label files.
7. Repeat for each condition and evaluate all exported models on the fixed test set.

### Training settings

| Setting | Value |
|---|---:|
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |

**Verification note:** These values were confirmed from the Small model's training screen and were the intended common settings for all conditions. Before treating them as confirmed settings for every model, check the original training records for the other four conditions. A prediction CSV does not establish the training settings.

Teachable Machine handles the underlying model training. The Python script in this repository performs inference and evaluation; it is not a Python implementation of the training procedure.

Retraining may yield different weights and results because initialization, internal splitting, and training operations can vary. The supplied exported models are the reference artifacts for reproducing the reported evaluation.

## 6. Metrics and Results

- **Accuracy:** Correct predictions divided by all test images.
- **AI recall:** Correctly predicted AI images divided by all actual AI images.
- **Real recall:** Correctly predicted real images divided by all actual real images.
- **Macro-F1:** The unweighted mean of the two class-wise F1 scores.

For confusion matrices, rows represent actual labels and columns represent predicted labels, in the order `[ai, real]`.

| Model | Accuracy (%) | AI recall (%) | Real recall (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| Small | 68.13 | 66.25 | 70.00 | 68.11 |
| Balanced | 73.13 | 75.00 | 71.25 | 73.12 |
| Large | 76.25 | 82.50 | 70.00 | 76.16 |
| AI Majority | 59.38 | 96.25 | 22.50 | 52.98 |
| Real Majority | 56.88 | 22.50 | 91.25 | 51.10 |

Values are rounded to two decimal places. Compute metrics from the underlying counts or image-level predictions, not from rounded percentages.

### Main observations

- Accuracy increased from 68.125% to 76.25% as balanced uploaded training data increased from 100 to 400 images.
- This improvement was mainly associated with increased AI recall. Real recall remained approximately unchanged.
- At a fixed total of 200 uploaded images, class imbalance reduced performance on the balanced test set.
- Recall for the underrepresented class fell to 22.50% in both imbalance directions.

## 7. Analyze the Saved Predictions Without Running the Models

Each supplied prediction CSV contains:

| Column | Description |
|---|---|
| `image_path` | Test image path relative to the test directory |
| `true_label` | Ground-truth class |
| `predicted_label` | Predicted class |
| `ai_score` | Model output for the AI class |
| `real_score` | Model output for the real class |
| `correct` | 1 for a correct prediction and 0 otherwise |

Accuracy, class-wise recall, precision, F1, and confusion matrices can be recomputed from `true_label` and `predicted_label` alone. The score columns additionally support threshold-based analyses such as ROC-AUC.

## 8. Limitations

- One training run was evaluated per experimental condition; variability across repeated training runs was not quantified.
- Comparisons are descriptive and do not establish statistical significance.
- Exact duplicates were checked, but near-duplicates were not systematically screened.
- Results concern this dataset and should not be interpreted as universal AI-image detection performance.
- At a fixed epoch count, larger training datasets also involve more optimization steps; dataset quantity and training computation are not isolated from each other.
- Exact evaluation reproduction depends on using the same exported model, test images, preprocessing, and compatible software environment.
