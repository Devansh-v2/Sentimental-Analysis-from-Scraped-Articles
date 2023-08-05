import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import os
import warnings

nltk.download('punkt')
warnings.filterwarnings("ignore")

# Converting Excel file to DataFrame
df = pd.read_excel('Input.xlsx')

# Importing stopwords and word lists
stp_wrds = 'StopWords_All.txt'
Pos_wrds = 'positive-words.txt'
Neg_wrds = 'negative-words.txt'

# Extracting URLs from the DataFrame
urls = df['URL']
url_no = []
temp = []

# Creating a folder to store articles
folder_name = "Articles"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Extracting Title and Text from urls
for url in urls:

    # Sending a GET request to the URL
    page = requests.get(url, headers={"User-Agent": "XY"})
    url_no.append(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    # Extracting the title of the article
    title = soup.find("title").get_text()
    title = title.replace('| Blackcoffer Insights', '')

    # Checking if the title is "Page not found"
    if title == 'Page not found':

        # Droping the corresponding row from the DataFrame
        rows_to_drop = df[df['URL'] == url].index
        df.drop(index=rows_to_drop, inplace=True)
    else:

        # Extracting the text content of the article
        text_element = soup.find(attrs={'class': 'td-post-content'})
        if text_element is not None:
            text = text_element.get_text()

    temp.append(text)

    # Generating a file name as Title for the article
    file_title = re.sub(r'[^\w\s.-]', '', title)
    file_name = f"{folder_name}/{file_title}.txt"

    # Writeing the article text to a file
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write("Article Title: " + title + "\n")
        file.write("Article Text: " + text + '\n')

# Reading positive word list from file
with open(Pos_wrds, 'r') as posfile:
    Pos_wrds_content = posfile.read().lower()
positiveWordList = Pos_wrds_content.split('\n')

# Reading negative word list from file
with open(Neg_wrds, 'r') as negfile:
    Neg_wrds_content = negfile.read().lower()
NegativeWordList = Neg_wrds_content.split('\n')

# Reading stopwords list from file
with open(stp_wrds, 'r') as stpfile:
    Stop_wrds_content = stpfile.read().lower()
stopWordList = Stop_wrds_content.split('\n')

# Tokenizer function to filter stopwords
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words

# Function to calculate positive score
def positive_score(text):
    posword = 0
    tokenphrase = tokenizer(text)
    for word in tokenphrase:
        if word in positiveWordList:
            posword += 1
    retpos = posword
    return retpos

# Function to calculate negative score
def negative_score(text):
    negword = 0
    tokenphrase = tokenizer(text)
    for word in tokenphrase:
        if word in NegativeWordList:
            negword += 1
    retneg = negword
    return retneg

# Function to calculate polarity score
def polarity_score(positive_score, negative_score):
    return round((positive_score - negative_score) / ((positive_score + negative_score) + 0.000001), 3)

# Function to calculate subjectivity score
def subjectivity_score(positive_score, negative_score, filtered_words):
    return round((positive_score + negative_score) / (len(filtered_words) + 0.000001), 3)

# Function to calculate average sentence length
def average_sentence_length(text):
    word_count = len(tokenizer(text))
    sentence_count = len(sent_tokenize(text))
    if sentence_count > 0:
        average_sentence_length = word_count / sentence_count
    else:
        average_sentence_length = 0
    return round(average_sentence_length)

# Function to calculate percentage of complex words
def percentage_complex_word(text):
    tokens = tokenizer(text)
    complexWord = 0
    complex_word_percentage = 0
    for word in tokens:
        vowels = 0
        if word.endswith(('es', 'ed')):
            pass
        else:
            for w in word:
                if w == 'a' or w == 'e' or w == 'i' or w == 'o' or w == 'u':
                    vowels += 1
            if vowels > 2:
                complexWord += 1
    if len(tokens) != 0:
        complex_word_percentage = complexWord / len(tokens) * 100
    return round(complex_word_percentage, 2)

# Function to calculate Fog Index
def fog_index(averageSentenceLength, percentageComplexWord):
    fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
    return round(fogIndex, 2)

# Function to calculate average number of words per sentence
def average_words_per_sentence(text):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    total_words = 0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        total_words += len(words)
    average_words_per_sentence = total_words / total_sentences
    return round(average_words_per_sentence)

# Function to calculate complex word count
def complex_word_count(text):
    tokens = tokenizer(text)
    complexWord = 0
    for word in tokens:
        vowels = 0
        if word.endswith(('es', 'ed')):
            pass
        else:
            for w in word:
                if w == 'a' or w == 'e' or w == 'i' or w == 'o' or w == 'u':
                    vowels += 1
            if vowels > 2:
                complexWord += 1
    return complexWord

# Function to calculate total word count
def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)

# Function to count syllables in a word
def count_syllables(word):
    vowels = "aeiou"
    exceptions = ["es", "ed"]
    count = 0
    word = word.lower()
    if word.endswith(tuple(exceptions)):
        return count
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    return count

# Function to count syllables in a text
def count_syllables_in_text(text):
    words = text.split()
    total_syllables = 0
    total_words = len(words)
    for word in words:
        syllable_count = count_syllables(word)
        total_syllables += syllable_count
    return round(total_syllables / total_words)

# Function to count personal pronouns in a text
def count_personal_pronouns(text):
    pronouns = r'\b(I|we|my|ours|us)\b'
    excluded_word = r'\bUS\b'
    # Find all matches of personal pronouns
    matches = re.findall(pronouns, text, flags=re.IGNORECASE)
    # Exclude matches where the word is "US"
    matches = [match for match in matches if not re.search(excluded_word, match, flags=re.IGNORECASE)]
    # Count the remaining matches
    pronoun_count = len(matches)
    return pronoun_count

# Function to calculate average word length
def average_word_length(text):
    words = text.split()
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    if total_words == 0:
        return 0
    average_length = total_characters / total_words
    return round(average_length)

# Creating a DataFrame to store the results
df1 = pd.DataFrame({'temp': temp})
df1['URL ID'] = range(len(df1))
df1['URL'] = url_no
df1["Positive Score"] = df1["temp"].apply(positive_score)
df1["Negative Score"] = df1["temp"].apply(negative_score)
df1["Polarity Score"] = np.vectorize(polarity_score)(df1['Positive Score'], df1['Negative Score'])
df1["Subjectivity Score"] = np.vectorize(subjectivity_score)(df1['Positive Score'], df1['Negative Score'], df1["temp"].apply(tokenizer))
df1["Average Sentence Length"] = df1["temp"].apply(average_sentence_length)
df1["Percentage of Complex Word"] = df1["temp"].apply(percentage_complex_word)
df1["Fog Index"] = np.vectorize(fog_index)(df1['Average Sentence Length'], df1['Percentage of Complex Word'])
df1["Average Number of Words per Sentence"] = df1["temp"].apply(average_words_per_sentence)
df1["Complex Word Count"] = df1["temp"].apply(complex_word_count)
df1["Word Count"] = df1["temp"].apply(total_word_count)
df1["Syllable per Word"] = df1["temp"].apply(count_syllables_in_text)
df1["Personal Pronoun"] = df1["temp"].apply(count_personal_pronouns)
df1["Average Word Length"] = df1["temp"].apply(average_word_length)

# Droping the temporary column from the DataFrame
df1 = df1.drop('temp', axis=1)

# Saving the DataFrame to an Excel file
df1.to_excel('Output Data Structure.xlsx', encoding='utf-8')
