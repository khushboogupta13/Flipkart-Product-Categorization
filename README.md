# Flipkart Product Categorization

This project aims at predicting the category of a product that is usually available on e-commerce sites like Flipkart. The Machine Learning and Deep Learning models have been trained on a Flipkart e-commerce dataset that can be accessed [here](https://docs.google.com/spreadsheets/d/1on6ApK7jNXSy20Bfw_NjKMTmeICTHyBvIK-3am9irFA/edit?usp=sharing).

## Codebase Structure

1. [Notebooks](https://github.com/khushboogupta13/Flipkart-Product-Categorization/tree/main/Notebooks): This is the folder containing all the Jupyter Notebooks that have been used for Exploratory Data Analysis, training and testing of the Machine Learning and Deep Learning models.
2. [requirements.txt](): This file contains all the dependencies of the project that are needed to reproduce the development environment.
3. [Dataset](https://github.com/khushboogupta13/Flipkart-Product-Categorization/tree/main/Dataset): This folder contains all the datasets (imbalanced and balanced) in CSV format. 

## Approach

### STEP 1: Exploratory Data Analysis and Data Preprocessing

- An in depth analysis of the dataset was done with the help of **Word Clouds, Bar Graphs, TSNE Visualizations**, etc to get an idea about the most frequent unigrams in the Product Description, distribution of products and brands across the different Product Categories, analysis of the length of the description, etc. 
- For Data Cleaning, **Contraction Mapping, removal of custom stopwords, URLs, Tokenization and Lemmatization** was done.
- Because of the clear imbalance in the dataset, balancing techniques like **Oversampling** and **Undersampling** were performed on the dataset as well. These were then saved in the form of a CSV file.

### STEP 2: Machine Learning Models for Product Categorization

### STEP 3: Deep Learning Models for Product Categorization

## Future Work
- Feature extraction can be performed on the **Product Category Tree** column in order to find a more detailed class to which a product can belong. 
- Using other advanced data balancing techniques like **SMOTE**, etc.
- Training and evaluating the Deep Learning model on datasets other than the undersampled one. These models could then be tested on a variety of e-commerce data available online to understand the scalability of the model when it comes to dealing with real-world data. 
- Using **Named Entity Recognition** techniques to figure out brands that make products belonging to a specific category.  

## References
- [ Vasvani et. al. Attention is all you need. Nips 2017](https://arxiv.org/pdf/1706.03762)
- [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers#saveevalcheckpoints)
- [Tranformer Models by HuggingFace](https://huggingface.co/transformers/pretrained_models.html)
- [Multiclass text classification with Deep Learning](https://www.google.com/url?q=https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613&sa=D&source=editors&ust=1618012348136000&usg=AOvVaw1ofOCyteqD6PZaxopj9qc8)
- [Multiclass text classification using LSTM](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17)






