# Potabilities - Water Quality Evaluation

<p align="center">
<img height="75%" src="https://user-images.githubusercontent.com/32643842/129628650-77cdec1b-03ae-4180-9b51-b93b4ead14ab.png" />
</p>

## Business Case
Potabilites is a start-up manufacturing a groundbreaking new device that will be able to quickly and cheaply test the potability of water anywhere in the world. 

As the engineers at Potabilities work on the physical testing device, our team was hired to develop the model this device will use to determine whether the water sample is potable according to the World Health Organization and US Environmental Protection Agency standards.

## Method
Our data was obtained from the Water Quality Dataset from Kaggle, available here. It contained 3,270 samples from water bodies all over the world. It contains nine features that the WHO and USEPA consider important in evaluating the potability of water. 

First we explored the distribution of our target variable and discovered that we had a class imbalance of 1,998 non-potable samples and 1,278 that were potable. 

<p align="center">
<img width="75%" src="https://user-images.githubusercontent.com/32643842/129629121-f704af34-f69e-4bd1-a570-fa14a43b82f1.jpg" />
</p>

Considering the class imbalance we decided to check the separability of our target data. By generating a scaled scatterplot we were able to see a clear separability which showed us that we did not need to worry about balancing our data with Smote or other techniques. 

Next we checked for any Null values in our data and discovered that 1,434 entries contained Null values. Since this is a fairly large chunk of our data we decided that we should try to fill in these gaps if possible. 

Having many different models at our disposal we decided to create a Modeling Environment so we can easily run tests with different configurations of our data on different models at once. We divided it into three components:
- Data Instantiations
- Model Instantiations
- Testing Area

To see the function we created to easily run any instantiated model and return the model as an object as well as the model's performance results see the file dm_models.ipynb in the Notes folder.

Considering all of our data exploration we decided that it would be good to test our models on 4 different data sets:
- Raw Data with filled in Null values using a Nearest Neighbors Classifier model which would fill in any NaNs based on their 5 closest neighbors.
- Data with all of our Null values dropped.
- Raw Data with filled in Null values using a Nearest Neighbors Classifier model which would fill in any NaNs based on their 5 closest neighbors while - dropping outlier data.

After extensive testing using 4 versions of our data we narrowed our ideal models to the top 3 performers:
- Gradient Boosting Classifier
- Random Forest Classifier
- Bagging Classifier

## Summary
Our top model turned out to be a Random Forest Classifier which gave us between a 69% and 71% accuracy on our test data. Our model precision for non-potable water is slightly higher than for potable, which we are encouraged by since we would absolutely prefer to lean on the side of caution with our life and death recommendations.

## Next Steps
Continuing to improve our model will require further testing by manipulating our Gridsearch parameters. We would also like to try some other methods of filling our NaN values, for example with the standard average. 

The most surefire way of improving our model will be through the collection of new, complete data which means that the device Potabilites is building will actually become more accurate as it is used on new water samples. As more data becomes available we would like to see if splitting the data by region can have an effect on the performance of our model.


## Repo Structure

├── build                   # Phase3_Project_Final_Notebook_Dimitry.ipynb

├── Notes                   # Drafts of Jupyter Notebooks for Project and dm_models notebook

├── data                    # Project Data and Images

└── README.md               # Overview of Project
