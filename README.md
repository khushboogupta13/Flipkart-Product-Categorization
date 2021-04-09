# Flipkart Product Categorization

This project aims at predicting the category of a product that is usually available on e-commerce sites like Flipkart. The Machine Learning and Deep Learning models have been trained on a Flipkart e-commerce dataset that can be accessed [here](https://docs.google.com/spreadsheets/d/1on6ApK7jNXSy20Bfw_NjKMTmeICTHyBvIK-3am9irFA/edit?usp=sharing).

## Codebase Structure

1. [Notebooks](https://github.com/khushboogupta13/Flipkart-Product-Categorization/tree/main/Notebooks): This is the folder containing all the Jupyter Notebooks that have been used for Exploratory Data Analysis, training and testing of the Machine Learning and Deep Learning models.
2. [requirements.txt](): This file contains all the dependencies of the project that are needed to reproduce the development environment.
3. [Dataset](https://github.com/khushboogupta13/Flipkart-Product-Categorization/tree/main/Dataset): This folder contains all the datasets (imbalanced and balanced) in CSV format. 

## STEP 1: Exploratory Data Analysis and Data PreProcessing

- An in depth analysis of the dataset was done with the help of **Word Clouds, Bar Graphs, TSNE Visualizations**, etc to get an idea about the most frequent unigrams in the Product Description, distribution of products and brands across the different Product Categories, analysis of the length of the description, etc. 
- For Data Cleaning, **Contraction Mapping, removal of custom stopwords, URLs, Tokenization and Lemmatization** was done.
- Because of the clear imbalance in the dataset, balancing techniques like **Oversampling** and **Undersampling** were performed on the dataset as well. These were then saved in the form of a CSV file.





