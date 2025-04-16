📰 Fake News Detection using NLP & Machine Learning
This project is aimed at detecting fake news articles using Natural Language Processing (NLP) techniques and a machine learning classifier. It utilizes TF-IDF vectorization for feature extraction and the Multinomial Naive Bayes algorithm for classification.

🚀 How It Works (Step-by-Step)
1. Import Required Libraries
All essential libraries such as pandas, nltk, sklearn, and numpy are imported to handle data processing, NLP, model training, and evaluation tasks.

2. Fix NLTK Issues
Downloads necessary NLTK resources such as:

Stopwords: to remove commonly used words that don’t add value.

Punkt tokenizer: for breaking sentences into words.

3. Load Dataset
The dataset is loaded from a CSV file with proper encoding handling (ISO-8859-1).
The script automatically tries to detect:

The text/content column

The label/category column

4. Text Preprocessing
A custom function is applied to clean and transform the text:

Converts text to lowercase

Removes numbers and punctuation

Tokenizes the text using regular expressions

Removes English stopwords

Returns cleaned tokens as a joined string

5. Convert Labels
Converts textual labels (e.g., "fake", "real") into numerical format using categorical encoding.

This is important for training the machine learning model.

6. Drop Empty Texts
Any rows with empty or NaN after preprocessing are removed to ensure valid data is passed to the model.

7. Train-Test Split
The cleaned data is split into training and testing sets (80% train, 20% test) using train_test_split.

8. Text Vectorization using TF-IDF
The TfidfVectorizer converts text into numerical vectors:

Minimum document frequency is set to 1

A maximum of 5000 features is retained

9. Train Naïve Bayes Model
A Multinomial Naive Bayes classifier is trained using the TF-IDF vectors from the training set.

10. Model Evaluation
The model is evaluated using:

Accuracy Score

Classification Report (Precision, Recall, F1-score for each class)

11. Live Prediction Example
The script includes example predictions on sample texts:

Predicts whether the given news is Fake or Real after preprocessing and vectorizing the input text.

📌 Sample Output
bash
Copy
Edit
✅ Model Accuracy: 0.9365

🔹 Classification Report:
              precision    recall  f1-score   support
      Fake       0.94       0.93      0.93       200
      Real       0.93       0.94      0.94       200

🛑 Fake News Detected!
✅ Real News!
📁 Dataset
Make sure the dataset (data.csv) is uploaded to your Google Drive (or local machine) and contains:

A column with news text/content

A column with labels indicating fake/real news

✅ Requirements
You can install required Python libraries using:

bash
Copy
Edit
pip install pandas numpy scikit-learn nltk
