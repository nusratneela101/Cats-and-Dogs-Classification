# Cats and Dogs Classification (CNN)

This project is a basic Convolutional Neural Network (CNN) implementation to classify images of **cats** and **dogs** using Python, TensorFlow, and Keras in Google Colab.

# Project Overview
We aim to train a deep learning model that can accurately classify whether an image is of a **cat** or a **dog** using a custom dataset. The model is built from scratch and includes preprocessing, training, and prediction features.


# Step-by-Step Workflow

1. Install Required Libraries
   - `pip install pillow`
   - TensorFlow, NumPy, Matplotlib (pre-installed in Google Colab)

2. Upload Dataset
   - Upload 50 cat images and 50 dog images
   - Store them in `input_cat/` and `input_dog/` folders

3. Resize and Rename
   - Images are resized to 150x150 pixels
   - Renamed to `cat1.jpg`, `dog1.jpg`, etc.

4. Split Dataset
   - Used `ImageDataGenerator` with 80% training and 20% validation

5. Build CNN Model
   - `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` layers
   - Activation functions: ReLU and Sigmoid

6. Train the Model
   - Optimizer: Adam  
   - Loss: Binary Crossentropy  
   - Metric: Accuracy  
   - Callback: EarlyStopping

7. Evaluate Model
   - Accuracy and loss on validation data
   - Optional: Confusion matrix, classification report

8. Predict New Images
   - Upload any new image to test
   - Model predicts if it’s a cat or dog

9. Save Model
   - Save as `cat_dog_model.h5` for reuse
   - 
## Dependencies

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pillow (PIL)  
- Google Colab

## Results

- 95% Validation Accuracy
- Real-time predictions from uploaded images


