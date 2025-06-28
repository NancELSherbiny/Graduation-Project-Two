# results of Translating CoDraw dataset
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the CSV file
df = pd.read_csv('references.csv')  # Replace with your file name

# Load the pre-trained model and tokenizer
model_name = "t5-small"  # You can use "t5-base" for better results
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to paraphrase a sentence
def paraphrase_sentence(sentence):
    input_text = "paraphrase: " + sentence
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1, num_beams=5)
    paraphrased_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_sentence

# Apply paraphrasing to each sentence in the DataFrame
df['Paraphrased_Sentence'] = df['references'].apply(paraphrase_sentence)

# Save the results to a new CSV file
df.to_csv('paraphrased_arabic_sentences.csv', index=False)

print("Paraphrased sentences saved to 'paraphrased_arabic_sentences.csv'")

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
#!pip install nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab') # Download the punkt_tab data package

# Load CSV files
references_df = pd.read_csv("references.csv")
predictions_df = pd.read_csv("predictions.csv")

# Extract references from both columns
references = []
for _, row in references_df.iterrows():
    # Get the values from 'references_1' and 'references_2', handle NaN
    ref1 = str(row['references_1']) if pd.notna(row['references_1']) else ""
    ref2 = str(row['references_2']) if pd.notna(row['references_2']) else ""

    # Add both references as a list for the current sentence
    references.append([ref1, ref2])

# Get predictions
predictions = predictions_df['predictions'].tolist()
predictions = [str(pred) for pred in predictions] # Convert to strings

# Validate that both lists have the same number of sentences
if len(references) != len(predictions):
    raise ValueError("The number of references and predictions must be the same.")

# Calculate METEOR score for each reference-prediction pair
meteor_scores = [
    meteor_score([word_tokenize(ref1), word_tokenize(ref2)], word_tokenize(pred))
    for [ref1, ref2], pred in zip(references, predictions)
]

# Calculate the average METEOR score
average_meteor = sum(meteor_scores) / len(meteor_scores)

# Output the results
print("Individual METEOR scores for each sentence:")
for i, score in enumerate(meteor_scores):
    print(f"Sentence {i+1}: {score:.4f}")

print(f"\nAverage METEOR score: {average_meteor:.4f}")

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from pyarabic.araby import tokenize  # Import the Arabic tokenizer
import re

# Install NLTK and download necessary data (if not already done)
import nltk
nltk.download('wordnet')

# Normalize Arabic text to improve consistency
def normalize_arabic(text):
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)     # Normalize different forms of Alif
    text = re.sub(r'ى', 'ي', text)         # Normalize Ya
    text = re.sub(r'ؤ', 'و', text)         # Normalize Waw
    text = re.sub(r'ئ', 'ي', text)         # Normalize Hamza on Ya
    return text

# Load CSV files
# Replace 'references.csv' and 'predictions.csv' with your actual file paths
references_df = pd.read_csv("references.csv")
predictions_df = pd.read_csv("predictions.csv")

# Assuming the CSVs have columns named 'reference1', 'reference2', and 'prediction'
references = [
    [normalize_arabic(ref1), normalize_arabic(ref2)]
    for ref1, ref2 in zip(references_df['references_1'], references_df['references_2'])
]
predictions = [normalize_arabic(pred) for pred in predictions_df['predictions'].tolist()]

# Tokenize using pyarabic's tokenizer
references = [[tokenize(ref1), tokenize(ref2)] for ref1, ref2 in references]  # Wrap each pair in a list
predictions = [tokenize(pred) for pred in predictions]

# Validate that both lists have the same number of sentences
if len(references) != len(predictions):
    raise ValueError("The number of references and predictions must be the same.")

# Calculate METEOR score for each reference-prediction pair
meteor_scores = [
    meteor_score(refs, pred)
    for refs, pred in zip(references, predictions)
]

# Calculate the average METEOR score
average_meteor = sum(meteor_scores) / len(meteor_scores)

# Output the results
print("Individual METEOR scores for each sentence:")
for i, score in enumerate(meteor_scores):
    print(f"Sentence {i+1}: {score:.4f}")

print(f"\nAverage METEOR score: {average_meteor:.4f}")

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from pyarabic.araby import tokenize  # Import the Arabic tokenizer
import re

# Install NLTK and download necessary data (if not already done)
#!pip install nltk
import nltk
nltk.download('wordnet')

# Normalize Arabic text to improve consistency
def normalize_arabic(text):
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)     # Normalize different forms of Alif
    text = re.sub(r'ى', 'ي', text)         # Normalize Ya
    text = re.sub(r'ؤ', 'و', text)         # Normalize Waw
    text = re.sub(r'ئ', 'ي', text)         # Normalize Hamza on Ya
    return text

# Load CSV files
# Replace 'references.csv' and 'predictions.csv' with your actual file paths
references_df = pd.read_csv("references.csv")
predictions_df = pd.read_csv("predictions.csv")

# Extract references from both columns, handling NaN values
references = []
for _, row in references_df.iterrows():
    ref1 = normalize_arabic(str(row['references_1'])) if pd.notna(row['references_1']) else ""
    ref2 = normalize_arabic(str(row['references_2'])) if pd.notna(row['references_2']) else ""
    references.append([ref1, ref2])

predictions = predictions_df['predictions'].tolist()
predictions = [normalize_arabic(str(pred)) for pred in predictions]

# Tokenize using pyarabic's tokenizer
references_tokenized = [[tokenize(ref1), tokenize(ref2)] for ref1, ref2 in references]
predictions_tokenized = [tokenize(pred) for pred in predictions]

# Validate that both lists have the same number of sentences
if len(references_tokenized) != len(predictions_tokenized):
    raise ValueError("The number of references and predictions must be the same.")

# Calculate METEOR score for each reference-prediction pair
meteor_scores = [
    meteor_score(refs, pred)
    for refs, pred in zip(references_tokenized, predictions_tokenized)
]

# Calculate the average METEOR score
average_meteor = sum(meteor_scores) / len(meteor_scores)

# Output the results
print("Individual METEOR scores for each sentence:")
for i, score in enumerate(meteor_scores):
    print(f"Sentence {i+1}: {score:.4f}")

print(f"\nAverage METEOR score: {average_meteor:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure with two subplots: one for the histogram and one for the box plot
fig, (ax_hist, ax_box) = plt.subplots(
    nrows=2, ncols=1,
    gridspec_kw={"height_ratios": [4, 1]},
    figsize=(8, 6),
    sharex=True
)

# Plot the histogram
sns.histplot(meteor_scores, bins=10, kde=False, ax=ax_hist, color="skyblue")
ax_hist.set_ylabel("Frequency")
ax_hist.set_title("Distribution of METEOR Scores")

# Plot the box plot
sns.boxplot(x=meteor_scores, ax=ax_box, color="lightgreen")
ax_box.set_xlabel("METEOR Score")

# Remove the y-axis label for the box plot
ax_box.set_yticks([])

# Adjust the spacing between the subplots
plt.tight_layout()

# Display the plots
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

'
# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create the strip plot
plt.figure(figsize=(10, 1))
sns.stripplot(x=meteor_scores, color="blue", jitter=True)
plt.xlabel("METEOR Score")
plt.title("Strip Plot of METEOR Scores")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns



# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x=meteor_scores, inner="point", color="lightblue")
plt.xlabel("METEOR Score")
plt.title("Violin Plot of METEOR Scores")
plt.show()

import matplotlib.pyplot as plt

# Generate a list of sentence indices
sentences = list(range(1, len(meteor_scores) + 1))

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(sentences, meteor_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Sentence Number')
plt.ylabel('METEOR Score')
plt.title('METEOR Scores for Each Sentence')
plt.grid(True)
plt.xticks(sentences)  # Set x-ticks to be each sentence number
plt.tight_layout()
plt.show()

import numpy as np

# Calculate additional statistics
median_score = np.median(meteor_scores)
mode_score = np.argmax(np.bincount([int(score * 100) for score in meteor_scores])) / 100  # Approximate mode
variance_score = np.var(meteor_scores)
std_dev_score = np.std(meteor_scores)

print(f"Median METEOR score: {median_score:.4f}")
print(f"Mode METEOR score: {mode_score:.4f}")
print(f"Variance of METEOR scores: {variance_score:.4f}")
print(f"Standard deviation of METEOR scores: {std_dev_score:.4f}")

# Categorize scores
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['Poor (0-0.2)', 'Moderate (0.2-0.4)', 'Good (0.4-0.6)', 'Very Good (0.6-0.8)', 'Excellent (0.8-1.0)']
categorized_scores = pd.cut(meteor_scores, bins=bins, labels=labels)

# Count the number of sentences in each category
category_counts = categorized_scores.value_counts().sort_index()

# Plot the distribution of categories
plt.figure(figsize=(8, 5))
sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
plt.title("Distribution of METEOR Score Categories")
plt.xlabel("METEOR Score Category")
plt.ylabel("Number of Sentences")
plt.xticks(rotation=45)
plt.show()

# Find sentences with the highest and lowest scores
max_score_index = np.argmax(meteor_scores)
min_score_index = np.argmin(meteor_scores)

print(f"Sentence with the highest METEOR score ({meteor_scores[max_score_index]:.4f}):")
print(f"Prediction: {predictions[max_score_index]}")
print(f"Reference: {references[max_score_index]}")

print(f"\nSentence with the lowest METEOR score ({meteor_scores[min_score_index]:.4f}):")
print(f"Prediction: {predictions[min_score_index]}")
print(f"Reference: {references[min_score_index]}")

# Example: Correlation between sentence length and METEOR score
sentence_lengths = [len(pred) for pred in predictions]

# Calculate correlation
correlation = np.corrcoef(sentence_lengths, meteor_scores)[0, 1]
print(f"Correlation between sentence length and METEOR score: {correlation:.4f}")

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=sentence_lengths, y=meteor_scores, color="blue")
plt.title("Sentence Length vs. METEOR Score")
plt.xlabel("Sentence Length (Number of Tokens)")
plt.ylabel("METEOR Score")
plt.show()

# Cumulative distribution plot
plt.figure(figsize=(8, 5))
sns.ecdfplot(meteor_scores, color="purple")
plt.title("Cumulative Distribution of METEOR Scores")
plt.xlabel("METEOR Score")
plt.ylabel("Proportion of Sentences")
plt.grid(True)
plt.show()

# Heatmap of METEOR scores
plt.figure(figsize=(10, 6))
sns.heatmap(np.array(meteor_scores).reshape(1, -1), cmap="YlGnBu", cbar=True)
plt.title("Heatmap of METEOR Scores")
plt.xlabel("Sentence Number")
plt.ylabel("METEOR Score")
plt.show()

import pandas as pd

# Create a summary table
summary_data = {
    "Metric": ["Average", "Median", "Standard Deviation", "Minimum", "Maximum"],
    "Value": [
        np.mean(meteor_scores),
        np.median(meteor_scores),
        np.std(meteor_scores),
        np.min(meteor_scores),
        np.max(meteor_scores)
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df)

# Save visualizations
plt.savefig("meteor_score_distribution.png")
summary_df.to_csv("meteor_score_summary.csv", index=False)

import plotly.express as px

# Interactive line plot
fig = px.line(x=range(1, len(meteor_scores) + 1), y=meteor_scores, labels={"x": "Sentence Number", "y": "METEOR Score"})
fig.update_layout(title="Interactive METEOR Scores for Each Sentence")
fig.show()
