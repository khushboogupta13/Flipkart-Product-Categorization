# Flipkart Product Categorization

This project aims at predicting the category of a product that is usually available on e-commerce sites like Flipkart. The Machine Learning and Deep Learning models have been trained on a Flipkart e-commerce dataset that can be accessed [here](https://docs.google.com/spreadsheets/d/1on6ApK7jNXSy20Bfw_NjKMTmeICTHyBvIK-3am9irFA/edit?usp=sharing).

## Codebase Structure

1. [Notebooks](https://github.com/khushboogupta13/Flipkart-Product-Categorization/tree/main/Notebooks): This is the folder containing all the Jupyter Notebooks that have been used for Exploratory Data Analysis, training and testing of the Machine Learning and Deep Learning models.
2. [requirements.txt](https://github.com/khushboogupta13/Flipkart-Product-Categorization/blob/main/requirements.txt): This file contains all the dependencies of the project that are needed to reproduce the development environment.
3. [Dataset](https://github.com/khushboogupta13/Flipkart-Product-Categorization/tree/main/Dataset): This folder contains all the datasets (imbalanced and balanced) in CSV format. 
4. [Report](): This folder contains the Report which summarises all the observations and conclusions made while working on the project. 

## Approach

The following tasks were undertaken for the **Multiclass Classification** of e-commerce products based on their **description**:

  1. The dataset and several of its hidden parameters were visualised (using libraries like `seaborn`, `matplotlib`, `yellowbrick`, etc). This then helped in data cleaning as several words from the Word Cloud were removed from the corpus as they did not contribute much in terms of Product Classification.
  2. It was decided to move forward by only using the **root of the Product Category Tree** as the Primary label/category for classification.
  3. Data cleaning, preprocessing and resampling was then performed to balance out the given dataset. 
  4. After a detailed analysis of the dataset through visualisation and other parameters, it was decided to categorise the products in the following 13 categories and remove the noise (other miscellaneous categories having less than 10 products):
    
    a) Clothing
    b) Jewellery
    c) Sports & Fitness
    d) Electronics
    e) Babycare
    f) Home Furnishing & Kitchen
    g) Personal Accessories
    h) Automotive
    i) Pet Supplies
    j) Tools & Hardware
    k) Ebooks
    l) Toys & School Supplies
    m) Footwear
    
  6. Then, the following Machine Learning algorithms (using `scikit-learn` libraries) were applied on the dataset:
    
    a) Logistic Regression (Binary and Multiclass variants)
    b) Linear Support Vector Machine
    c) Multinomial Naive Bayes
    d) Decision Tree
    d) Random Forest Classifier
    e) K Nearest Neighbours

   5. Even though good accuracy was achieved using the ML models, the following Deep Learning Models (using `PyTorch` framework) were also implemented on the dataset:
    
    1. Transformer based models like:
    a) Bidirectional Encoder Representations from Transformers (BERT)
    b) RoBERTa
    c) DistilBERT
    d) XLNet
    2. Recurrent Neural Network based Long-Short Term Memory(LSTM)


### STEP 1: Exploratory Data Analysis and Data Preprocessing

- An in depth analysis of the dataset was done with the help of **Word Clouds, Bar Graphs, TSNE Visualizations**, etc to get an idea about the most frequent unigrams in the Product Description, distribution of products and brands across the different Product Categories, analysis of the length of the description, etc. 
- For Data Cleaning, **Contraction Mapping, removal of custom stopwords, URLs, Tokenization and Lemmatization** was done.
- Because of the clear imbalance in the dataset, balancing techniques like **Oversampling** and **Undersampling** were performed on the dataset as well. These were then saved in the form of a CSV file.

### STEP 2: Machine Learning Models for Product Categorization

- The above mentioned 6 ML algorithms were applied on the imbalanced, oversampling balanced and undersampling balanced datasets. Noise was removed from each of these datasets and these datasets had already been cleaned and preprocessed in the previous notebook. 
- Several evaluation metrics like **Classification Report, Confusion Matix, Accuracy Score, ROC Curves and AUC Scores** were used for the comparison of the models. The Validation score of the ML algorithms when applied on the dataset are tabulated below:

| ML Algorithm       | Validation Accuracy on Imbalanced Dataset          | Validation Accuracy on Balanced Dataset (Oversampling)                          | Validation Accuracy on Balanced Dataset (Undersampling)          |
| ---                | ---             | ---                                   | ---             | 
| Logistic Regression (Binary)               | 0.9654           | 0.9756                        | 0.9486               | 
| Logistic Regression (Multiclass)     | 0.9735            | 0.9893                     | 0.9654               | 
| Naive Bayes     | 0.9096             | 0.9602        | 0.9054               | 
| __Linear SVM__            | __0.9799__            | __0.9958__              | __0.9749__               |
| Decision Trees            | 0.70170            | 0.6883                     | 0.7561               | 
| Random Classifier             | 0.9209          | 0.9367          | 0.9235                | 
| K Nearest Neighbours        | 0.9564           | 0.98       | 0.9453               | 

- From the above table, we can clearly see that **Linear Support Vector Machine** algorithm performed the best across all the three datasets.  

### STEP 3: Deep Learning Models for Product Categorization

-  The Deep Learning Models were only trained and evaluated on the dataset that was baalnced using the Undersampling technique.
- After a detailed study of all the Transformer based Deep Learning algorithms like BERT, RoBERTa, DistilBERT, XLNet and Recurrent Neural Network based LSTM, it was decided that **BERT (uncased, base, with all the layers freezed except the last one)** worked the best on our dataset by giving an **f1-score of 0.98**. 

![Confusion Matrix](https://i.imgur.com/SLskxzo.png)

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






