# GroupProject

# Traffic Sign Dataset: https://www.kaggle.com/datasets/daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes/data

# Introduction

We chose to work with a traffic sign dataset because traffic sign recognition is a critical component of modern driver assistance systems and autonomous vehicles. It's a fascinating challenge that combines computer vision and real-world problem solving. What makes this project especially cool is its focus on imperfect, real-world conditions, such as blurry, tilted, faded, or partially blocked signs, rather than just clean, textbook examples. These kinds of edge cases are exactly where current systems can struggle the most.	

By training and evaluating models on these more realistic scenarios, we can better understand how model performance is affected by visual noise and distortion. The broader impact of developing a robust predictive model in this context is significant: improving traffic sign recognition can enhance road safety, reduce accidents, and support the safe deployment of autonomous vehicles. A model that performs well under difficult conditions isn't just a technical achievement, but it's also a step toward making AI more trustworthy and practical in everyday life.

Methods

Data Exploration:

For our group project, we had several ways of exploring our data. The dataset we chose came with a folder with test images, sub folders of different traffic signs for training, and a test_data.csv file with the correct classification for each test image. Our first step in data exploration was reading in the test_data.csv to verify that the amount of rows in the file matches with the number of images in the test folder. Through using PySpark to read in the test_data.csv file, we were able to verify that there are indeed 205 ClassID, 205 different traffic signs, and 53,454 rows in the csv file. We were also curious, whether test images were split evenly among the ClassIDs, and through using groupBy, we were able to see that the ClassIDs do not have the same amount of test images. The test_data.csv file contains the ground truth of the correct labels to each of the test images.:

path = kagglehub.dataset_download(“daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes")

![image](https://github.com/user-attachments/assets/3089043b-1c75-4e08-9cdd-1bae251a1642)


Dataset analysis showed that training and test images were organized in separate folders by class. Training data was loaded by going through each class folder:

```data = [] # Iterating over each subdirectory in "Train" (each subdir is a different class) 
for subdir in sorted(os.listdir(train_root)): class_dir = os.path.join(train_root, subdir) if os.path.isdir(class_dir): class_id = subdir # e.g. "0", "1", "10", ... # Iterate over all files in that subdirectory for filename in os.listdir(class_dir): full_path = os.path.join(class_dir, filename) if os.path.isfile(full_path): data.append((full_path, class_id))```

Image dimension analysis was performed using UDF functions:

```def get_image_size(path): 
	with Image.open(path) as img: 
		return (img.width, img.height)```

```udf_get_image_size = udf(get_image_size, schema_image_size) df_with_size = df_train.withColumn("size", udf_get_image_size(col("Path")))```

We were also curious about the distribution of ClassID and Frequency of images in the test and train datasets, and we were able to visualize the comparisons through a histogram. This allowed us to see which ClassID had the maximum and minimum amount of images, and the average of images in each ClassID.

![image](https://github.com/user-attachments/assets/d089fe87-8eab-4f3c-9dbf-d2d074ab6f1d)





