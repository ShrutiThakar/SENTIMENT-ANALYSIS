# SENTIMENT-ANALYSIS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SHRUTI ASHOK THAKAR

*INTERN ID*: CT08DF1127

*DOMAIN*: DATA ANALYSIS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION*: This code builds and evaluates a logistic regression model for sentiment analysis using a Twitter dataset. It walks through a complete natural language processing (NLP) workflow, including data loading, text preprocessing, feature extraction using TF-IDF, model training, and performance evaluation using classification metrics and a confusion matrix. This end-to-end pipeline is implemented in a Jupyter Notebook or Google Colab environment, making it interactive, efficient, and easy to visualize.

The process begins with the installation of the required libraries. The command !pip install nltk seaborn installs NLTK (Natural Language Toolkit), which is a powerful Python library for text processing and linguistic analysis, and Seaborn, a visualization library built on top of Matplotlib that provides attractive statistical plots. These libraries play a crucial role in handling text cleaning and presenting evaluation results effectively.

Following installation, the necessary libraries are imported. The core ones include pandas for data manipulation, re and string for regular expressions and string handling, matplotlib.pyplot and seaborn for data visualization, and several components from sklearn including data splitting, text vectorization, model building, and evaluation metrics. From nltk, the script imports stopwords to filter out common words that don’t contribute much meaning and WordNetLemmatizer for lemmatization, which reduces words to their base or root form. The script also ensures that the required corpora for stopwords and wordnet are downloaded using nltk.download().

The dataset is loaded using the wget command, which fetches a CSV file from GitHub containing labeled tweets. This dataset includes two columns: label and tweet. The column names are renamed for clarity—label is changed to sentiment and tweet to text. The sentiment labels are mapped to binary format where 1 indicates a positive or relevant sentiment and 0 indicates negative or irrelevant sentiment. This conversion is crucial for binary classification.

The core NLP step involves text preprocessing, which is encapsulated in the clean() function. This function performs multiple important tasks:

Converts all characters to lowercase for uniformity.

Removes URLs, special characters, and digits using regular expressions.

Splits the cleaned string into individual words (tokens).

Removes stopwords such as "the", "is", and "in", which carry little analytical value.

Lemmatizes each word, converting it to its root form to reduce dimensionality.
This function is applied to the entire dataset, and a new column clean is added containing the preprocessed tweets.

Next, the cleaned text data is transformed into numerical features using TF-IDF vectorization (TfidfVectorizer). This technique converts textual data into a matrix of numerical values that reflect the importance of each word relative to all other documents. The parameter max_features=5000 limits the number of features to the top 5,000 most important terms, ensuring computational efficiency.

The data is then split into training and testing sets using train_test_split, with 80% used for training and 20% for testing. A Logistic Regression model—a simple yet effective linear model for binary classification—is then trained on the training data.

After training, the model makes predictions on the test set, and its performance is evaluated using several metrics. The classification_report() displays precision, recall, F1-score, and support for both classes (positive and negative sentiment), which helps to understand how well the model is distinguishing between the two. Additionally, a confusion matrix is computed and visualized using a Seaborn heatmap. This matrix provides a direct view of the number of true positives, true negatives, false positives, and false negatives, allowing one to assess where the model is making correct predictions and where it is failing.

In conclusion, this code represents a practical and comprehensive workflow for performing sentiment analysis on social media text. It demonstrates all key stages of an NLP project: cleaning raw text data, extracting meaningful features, training a machine learning model, and evaluating it with visual and quantitative tools. By using standard libraries and real-world data, this script serves as a powerful foundation for more advanced text classification tasks in fields such as opinion mining, customer feedback analysis, and social media monitoring.
