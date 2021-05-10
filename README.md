# Neural_Network_Charity_Analysis

## Overview

The purpose of the analysis in this repository is to create a neural network model that can assist in determining whether applicants will be successful if funded by Alphabet Soup. The model's accuracy is a key performance criteria of the model's success.

## Resources

Languages & Libraries used in this repository are:

* Python
  * [scikit-learn library](https://scikit-learn.org/stable/index.html)
  * [Pandas](https://pandas.pydata.org/docs/)
  * [TensorFlow for Python](https://www.tensorflow.org/api_docs/python/tf)

From the [dataset provided by Alphabet Soup](/Resources/charity_data.csv)

## Results

### Preprocessing

The target for the model is the `IS_SUCCESSFUL` flag.

Features for the model would include:

* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* NAME-An identifier column that is used to filter out 'first time' applicants

Removed from the dataset is `EIN` column used for identification purposes.

### Compilation, Training & Evaluation of the model

Model performance compared to the initial neural network was optimized by increasing the number of neural layers, increasing the number of neurons, decreasing the epochs to account for the increased neurons to prevent overfitting, and keeping `NAME` data but creating a `First_Time_Applicant` tag to replace company names where the name only shows up once as seen by the `value_counts` method.

Three layers were selected for the optmizied neural network. The input layer consists of 836 input features. The first hidden layer consists of an equal amount of neurons. The second consists of half that number of neurons and third layer consists of one third of the number of input features of neurons. The number of neurons chosen was determined by best practice of having total neurons be between two to three times the number of input features with understanding that past the first layer, diminshing returns occurs. Additionally, under the consideration that the original dataframe consisted of 9 labels pre-encoding, the neuron to feature ratio can also be more lax since a majority of the features added were from `NAME` in comparison to the original network creation that did not include names.

Original Neural Network (sans `NAME`)

![original](/Resources/originalnn.png)

Optimized Neural Network (include `NAME`)

![nncreation](/Resources/nncreation.png)

The target performance of greater than 75% was achieved clocking in at 0.7953352928161621% accuracy

![accuracy](/Resources/finalaccuracy.png)

## Summary

With the increase in accuracy given addition of the `NAME` identifier and binning, it would appear that first time applicants may skew the accuracy, likely in that Alphabet Soup would be willing to give first time applicants a try and first time applicants being more likely to follow through and be successful to provide a good impression on Alphabet Soup. This in combination with Special Considerations albeit minimal, adds to the model.

Given that first time applicants seems to be such a factor, then a different model to predict success could be the random forest classifier using the same dataset since a first time applicant would lean to a particular side within that classifier.