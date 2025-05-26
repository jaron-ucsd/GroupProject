# GroupProject

# Traffic Sign Dataset: https://www.kaggle.com/datasets/daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes/data

Our group project is based on the traffic signs dataset, in which we will be training a model and evaluating the model's performace on different traffic signs from the test data. 

Pre-Processing Stage:
The first steps of pre-processing our data would be understanding the dataset first.

As for visualization, we would plot a bar graph to visualize how many images are in each class. 
This will allow us to see if the classes are balanced or not, and if so we can set a maximum image per class.
Example: Setting the train data classes to have 150 images each to train on and setting the test data classes to have 20 images each.

Then we would load in the Train and Test data, along with the Test.csv to read the acutual labels of the test data. 

Next step would be checking to see if any data in the dataset is corrupt, unreadable, and or have missing labels or images.

Then we can resize all the images to the same size for uniformity, since some images may have different sizes.

We can also fix and repair some images by data augmentation (rotation, increasing/decreasing brightness, zoom/unzoom) 

Since this dataset has already split the train and test data set, we can read them in seperately without creating a train/test split.

# Environment Setup
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

Every other setting were left untouched, and no values. 
