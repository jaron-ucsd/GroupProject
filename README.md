Traffic Sign Classification - Milestone 3


Project Overview

This milestone continues our traffic sign classification project using the GTSRB+ dataset with 205 total traffic sign classes. After our first milestone where we explored the data and planned our preprocessing steps, we have now finished the main preprocessing work and built our first model. In this phase, we completed the data preprocessing steps we planned earlier and created a convolutional neural network (CNN) to classify traffic sign images. 

Project Progress: From Planning to Implementation

Previous Milestone

In our previous milestone, we built the foundation for this project by:
- Dataset Choice: Selected the GTSRB+ Traffic Signs dataset with 162 custom classes
- Data Analysis: Looked at class distribution and found imbalanced classes
- Planning: Decided to use 150 images per training class and 20 images per test class
- Quality Check: Planned to check for broken or missing images
- Size Issues: Found that images had different sizes and needed to be made uniform

Current Milestone

Building on our previous work, we have completed:

- Size Standardization: Made all images uniform at 32 x 32 pixels 
- Quality Verification: Checked filtered valid image files
- Visualization: Created bar plots showing class distribution across all 205 classes
- Data Augmentation: Applied rotation and translation 
- Model Development: Built and trained our first CNN model 
- Model Evaluation: Achieved 88.92% validation accuracy across all traffic sign classes
- Dataset: GTSRB+ Traffic Signs with 162 custom classes
- Data Source: Kaggle dataset by Daniil Deltsov
- Image Format: Various sizes, converted to grayscale during preprocessing
- Data Split: Train/Validation/Test sets

Dataset Statistics

- Training Data: Variable images pr class (up to 150 images per class after filtering)
- Test Data: Separate test set with unlabeled images
- Image Dimensions: Various original sizes, standardized to 32 x 32 pixels during model preprocessing

Current Preprocessing Progress

1. Data Loading and Class Balance 
- Data Loading: Utilized PySpark for efficient large-scale data precoessing
- Class Balancing: Implemented the planned limitation of 150 images per class for training
- Test Set Preparation: Loaded test images (unlabeled)


2. Data Quality and Size Standardization

# Implemented image size analysis and standardization
def get_image_size(path):
    with Image.open(path) as img:
        return (img.width, img.height)

# Applied UDF to analyze all image dimensions
df_with_size = df_train.withColumn("size", udf_get_image_size(col("Path")))

- Size Analysis: Analyzed all image dimensions using PySpark UDFs
- Uniformity Achievement: Standardized all images to 32 x 32 pixels as planned
- Data Integrity: Verified image readibility 

3. Class Distribution Visualization and Balancing

train_freq = df_train.groupBy("ClassId").count().orderBy("ClassId")
plt.figure(figsize=(20, 6))
plt.bar(train_freq_pd["ClassId"], train_freq_pd["count"])
plt.title("Histogram of Class ID Frequency in Train Data")

- Visualization Completion: Generated bar plots showing class distribution across all 205 classes
- Balance Implementation: Applied the planned 150-image limit per training class
- Analysis: Confirmed balancing across train, test, and combined datasets

4. Data Augmentation Implementation

# Implemented planned augmentation strategies
transform = transforms.Compose([
    transforms.Resize((32, 32)),           # Size uniformity
    transforms.RandomRotation(10),         # Rotation as planned
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization
])

- Rotation: Applied +/- 10 degrees random rotation as planned
- Translation: Implemented random translation
- Normalization: Standardized pixel values (mean=0.5, std=0.5)
- Grayscale Conversion: Converted RGB to grayscale for efficiency

Model Framework

Convolutional Neural Network

class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=205):
        super(TrafficSignNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.flat_features = 128 * 4 * 4  # 2048 features
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


Model Components:
- Input Layer: 32 x 32 grayscale images
- Convulutional Blocks: 3 sequential blocks (1-> 32-> 64-> 128 filters)
- Batch Normalization: Applied after each convolution for training stability 
- Activation: ReLU activation functions throughout
- Fully Connected Layers:
* Flatten layer: 128 * 4 * 4 = 2048 features
* Hidden layer: 512 neurons with ReLU and Dropout(0.5)
* Output layer: 205 classes

Model Training Setup

Hyperparameters:

- Optimizer: Adam with learning rate 0.001 (optim. Adam(model.parameters(), lr=0.001)
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 10
- Learning Rate Scheduler: StepLR(reduce by 0.1 every 5 epochs)

Traning Process

- Split dataset into 80% training, 20% validation 
- Implemented early monitoring of training vs validation accuracy 
- Applied data augmentation during training to imrpove generalization

Model Evaluation Results

Training Performance

The model was trained for 10 epochs with the following training loop output:

Starting training...
Epoch [1/10], Loss: 3.4529, Train Acc: 18.73%, Val Acc: 46.99%
Epoch [2/10], Loss: 2.2027, Train Acc: 38.17%, Val Acc: 66.78%
Epoch [3/10], Loss: 1.8090, Train Acc: 47.74%, Val Acc: 74.39%
Epoch [4/10], Loss: 1.5665, Train Acc: 53.83%, Val Acc: 77.92%
Epoch [5/10], Loss: 1.3790, Train Acc: 58.98%, Val Acc: 80.90%
Epoch [6/10], Loss: 1.1414, Train Acc: 65.52%, Val Acc: 86.30%
Epoch [7/10], Loss: 1.0641, Train Acc: 68.04%, Val Acc: 87.38%
Epoch [8/10], Loss: 1.0214, Train Acc: 69.11%, Val Acc: 87.81%
Epoch [10/10], Loss: 0.9341, Train Acc: 71.68%, Val Acc: 88.92%

Validation Results

Final Validation Accuracy: 88.92%

Classification Report Summary:
- Overall Accuracy: 89% (19,486 validation samples)
- Macro Average: Precision: 0.89, Recall: 0.85, F1-Score: 0.86
- Weighted Average: Precision: 0.89, Recall: 0.89, F1-Score: 0.89

Per-Class Performance Results
- Best Performing Classes: Classes 6, 15, 170 (F1-scores >= 0.99)
- Challenging Classes: Classes 47, 106, 160, 193, 196 (F1-scores =< 0.60)
- Most classes (>80%) achieved F1-scores above 0.80

Test Set Evaluation

test_dataset = TrafficSignDataset(test_rows, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Generate predictions for test set
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)


Fitting Analysis

Current Model Position:

Based on the training results: 

- Final Training Accuracy: 71.68%
- Final Validation Accuracy: 88.92%
- Training Loss Progression: Steady decrease from 3.45 to 0.93

Analysis

The model appears to be in the underfitting region because:

- Validation accuracy (88.92%) significantly exceeds training accuracy (71.68%), which is unusual and suggests that the model could learn the training  data better
- Consistent learning pattern: Both training and validation accuracies improved throughout training without signs of overfitting
- Loss progression: Training loss decreased consistently, indicating the model was still learning
- Performance gap: The 17% gap where validation outperforms training suggests the model has room for improvement on the training set

Learning Curve Interpretation

The training progression shows learning with:

1. Rapid initial improvement (Epochs 1-3): Validation accuracy jumped from 47% to 74%
2. Optimization (Epochs 4-8): Gradual improvement reaching 87-88%
3. Continued learning potential: No plateau observed by epoch 10

Next Steps and Future Models

Planned Improvements

1. Framework Modifications:
- Experiment with deeper networks (ResNet, DenseNet architectures)
- Add skip connections for better gradient flow
- Implement attention mechanisms for better feature focus

2. Regularizationn Techniques:
- Increase dropout rates if overfitting is detected
- Add L2 weight regularization
- Implement early stopping based on validation loss

3. Data Enhancement:
- Expand data augmentation strategies
- Balance class distribution through oversampling/undersampling

4. Hyperparameter Optimization
- Grid search for optimal learning rates
- Experiment with different optimizers (SGD, AdamW)
- Tune batch sizes and learning rate schedules


--------------------------------------------------------------------------------------------------------------
# Questions

Where does your model fit in the fitting graph? 
The model fits well within the fitting graph, showing a clear upward trend in both training and validation accuracy over the 10 epochs. While training accuracy starts lower, it steadily improves, and the validation accuracy remains consistently higher, suggesting that the model is learning meaningful patterns without overfitting. This positioning on the fitting graph indicates a model that is underfitting slightly but has strong potential with further training or fine-tuning.

What are the next models you are thinking of and why?
For the next steps, exploring more advanced models such as convolutional neural networks (CNNs) (if the data is image-based) or transformer-based architectures (for text data) would be a logical progression. These models are capable of capturing more complex features and relationships in the data. Additionally, experimenting with pretrained models or implementing transfer learning could provide a performance boost, especially for classes with fewer examples. These approaches can help build on the current model’s strengths while addressing areas where improvement is needed.

Conclusion

What is the conclusion of your 1st model?

Our CNN model was successful as a first attempt. We got a 88.92% accuracy on validation data. The training accuracy (71.68%) was lower than validation accuracy, meaning the model can learn more. Our model handled all classes which means it worked with all 205 traffic sign types. The model kept improving after 10 epochs  which means it was still learning. What worked well was that the complete data processing from PySpark to PyTorch worked smoothly. The 3-layer CNN design was effective for learning traffic sign features. The data augmentation (rotation and translation) helped the model. 

What can be done to possibly improve it?

We could train longer. We could continue past 10 epochs, since it was still improving. We could also better train the model. We could adjust settings to help the model learn the training data better. We could also work on classes that performed poorly (like classes 47, 106, 160). Another thing that we could do to imrpove the model is to try better models. We could possibly use ResNet or other advanced models or we could start with models that are already trained on images. We could also include brightness and zoom effects as we originally planned to do. 


Link:

https://www.kaggle.com/datasets/daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes/data

# Environment Setup:

These were the default settings we used to setup our Juypter Session for our environment on Expanse SDSC.

SLURM Settings:

Account: "TG-CIS240277"

Partition: "shared"

Time Limit: Based on how long you want your Juypter Session to be running, usually we run "120" minutes

Number of cores: 2

Memory required per node (GB): 4

GPUs: 0

Singularity Image File Location: we used the default given, "~/esolares/spark_py_latest_jupyter_dsc232r.sif"

Environment modules to be loaded: we loaded "singularitypro" to our Juypter Session

For Working Directiory, we used "home".

Type: "JuypterLab"


