Project Overview:
The script utilizes machine learning techniques to preprocess email data, train an SVM model, and evaluate its performance on a test dataset. The workflow includes:

Data Loading: The dataset is loaded from a CSV file containing email content and labels ("spam" or "ham").

Data Preprocessing: Missing values are handled, and the email labels are encoded into numeric values (0 for "ham" and 1 for "spam"). Text data is vectorized using CountVectorizer to convert the email content into a numerical representation.

Model Training: A Support Vector Machine (SVM) with a linear kernel is trained using the preprocessed dataset to classify emails.

Model Evaluation: The model’s performance is assessed using classification_report and confusion_matrix, and the confusion matrix is visualized using a heatmap for better interpretation.
