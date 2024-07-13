import streamlit as st
import pandas as pd

# Load your Excel files (replace with your actual files)
file1 = 'data/ontario_debates_june_2023.xlsx'
file2 = 'data/on_ced_2014.xlsx'
file3 = 'data/on_ced_2018.xlsx'
file4 = 'data/ontario_sentences_june_2023.xlsx'

# Display samples of each dataframe
st.title("CED Ontario Data Sample", anchor=False)


c = st.columns(2)
with c[0]:
    st.write('Elections Results, 2014, 2018')
    with open(file1, 'rb') as f:
        st.download_button(label="Download 2014 Election Data", data=f, file_name='on_ced_2014.xlsx')
    with open(file3, 'rb') as f:
        st.download_button(label="Download 2018 Election Data", data=f, file_name='on_ced_2014.xlsx')

with c[1]:
    st.write('Hansard Transcripts, November 2023')
    with open(file2, 'rb') as f:
        st.download_button(label="Download Speeches", data=f, file_name='ontario_debates_november_2024.xlsx')
    with open(file2, 'rb') as f:
        st.download_button(label="Download Sentences", data=f, file_name='ontario_sentences_november_2024.xlsx')

st.write('Ensure you are downloading these files in the same directory as the python script being run')


# Introduction to Data Analysis
st.header("Introduction to Simple Data Analysis")

with st.expander('Getting Started with Python'):
    st.markdown("""
    ## Getting Started with Data Analysis

    ### Option 1: Using Google Colab
    Google Colab is a free online platform that allows you to write and execute Python code in your browser.

    1. Go to [Google Colab](https://colab.research.google.com/).
    2. Click on **New Notebook**.
    3. Create new folder called 'data'
    4. Upload data into data folder
    5. You can now start coding in Python.
    [example filter election data notebook](https://colab.research.google.com/drive/1CYv_BXlI0tHIXmI89ktDOH9K41vGNaNB?usp=sharing)""")

    st.image('data/file_upload.png')

    st.markdown("""
    ### Option 2: Using Anaconda Locally
    Anaconda is a free and open-source distribution of Python and R programming languages for data science and machine learning.

    1. Download and install Anaconda from [Anaconda.com](https://www.anaconda.com/products/individual).
    2. Open **Anaconda Navigator** and launch **Jupyter Notebook**.
    3. You can now start coding in Python.

    ### Option 3: Search on Google or ChatGPT for other methods
    """)

with st.expander('Using ChatGPT/ Gemini for coding help'):
    st.markdown("""
    ## Using ChatGPT for Data Science Help

    You can use ChatGPT to assist you with data science coding issues or to generate analysis code. Here are some example prompts you can use:

    Example Prompts

    - "How do I load an Excel file into a pandas DataFrame?"

    - "Can you show me how to read a CSV file in Python?"

    - "Fix this code xxxx"

    - "Modify this code xxxx to aggregate votes by the party column"

    - "How do I train a linear regression model using scikit-learn?"

    - "Can you provide an example of how to use K-means clustering in Python?"

    Sample Prompt and Response
    Prompt:

    "How do I handle missing values in a pandas DataFrame?"

    Response:
    """)
    st.code("""
    import pandas as pd

    # Load data
    df = pd.read_excel('your_data.xlsx')

    # Drop rows with any missing values
    df_cleaned = df.dropna()

    # Fill missing values with a specific value
    df_filled = df.fillna(0)

    # Fill missing values with the mean of the column
    df_filled_mean = df.fillna(df.mean())

    print(df_cleaned.head())
    print(df_filled.head())
    print(df_filled_mean.head()""")

    st.markdown(
    """
    By using prompts like these, you can get quick assistance and code examples to help with your data science tasks.
    """)

st.header("Analysis Example", anchor=False)
st.write("Example of filtering the data to calculate incumbent success")
st.code("""
import pandas as pd #libary used to load data into a 'dataframe' for manipulation
import numpy as np #library used to manipulate numbers
file_name = 'data/on_ced_2014.xlsx'#file name/directory relative to file location of script being run
data = pd.read_excel(file_name)
print(list(data))
#column names are ['ID', 'Constituency', 'firstname', 'middlename', 'lastname', 'Party', 'Elected', 'Incumbent', 'Valid Votes']

#filter to just elected candidate results
victors = data[data['Elected'] == 1]

#count number of incumbent victors
incumbent_victors = len( victors[victors['Incumbent'] == 1] )

#filter to just un-elected candidate results
losers = data[data['Elected'] != 1]

#count number of un-elected incumbents
incumbent_losers = len( losers[losers['Incumbent'] == 1] )

#display the percentage of incumbent election success
incumbent_success = incumbent_victors / (incumbent_victors + incumbent_losers)
print(f'{np.round(incumbent_success, 2)*100} %')
""")

st.header("Use other people's research and code")
st.write('Osnabruegge, Moritz, Elliott Ash and Massimo Morelli. 2021. "Cross-Domain Topic Classification for Political Texts". Political Analysis. ')

st.image('data/data_availability_statement.png', 'https://codeocean.com/capsule/0078777/tree/v1')

with open('data/Cross-Domain Topic Classification for Political Texts.pdf', 'rb') as f:
    st.download_button(label="Download Cross-Domain Topic PDF", data=f, file_name='Cross-Domain Topic Classification for Political Texts.pdf')


st.write('The code below requires you download the BOTH models (tfidf_8.pkl, logistic_model_8.pkl ) below into folder data/models/')
st.markdown('You can download these models here: [Cross-Domain Codebase](https://codeocean.com/capsule/0078777/tree/v1)')
st.markdown('[Analysis Colab Notebook](https://colab.research.google.com/drive/1ANDuQnLOwBZgGrlCQOP3DTm4BGX6v0X9?usp=sharing)')
st.code("""
#define import libraries and define function to plot prediced class counts
import os
import pandas as pd
import numpy as np

#Function to create a bar plot of the counts of unique values in the
def count_bar_plot(plot_column, title):
    #dataframe column named what ever string is passed as plot_column
    #plot_column: str of name of column

    #get list of unique column variables
    unique_classes = data[plot_column].unique()

    # Define colors
    colors = plt.cm.get_cmap('tab10', len(unique_classes))

    # Calculate counts for each class
    counts = data[plot_column].value_counts().sort_index()

    # Plot each class with a unique color
    for i, unique_class in enumerate(unique_classes):
        count = counts[unique_class]
        plt.bar(unique_class, count, color=colors(i), alpha=0.7, label=unique_class)

    #remove x axis
    plt.gca().set_xticklabels([])
    plt.title(title)
    # Add a legend
    plt.ylabel('Sentence Count')
    plt.legend(title='Predicted Class')


#LOAD EXCEL DATA INTO PANDAS DATAFRAME
data = pd.read_excel('data/ontario_sentences_june_2023.xlsx')

# Take a subset of the data (first 1000 rows)
data = data.iloc[:1000, :]

# Extract the 'text' column for analysis
X = data['text'].values

# Load the pre-trained TF-IDF vectorizer
vectorizer1 = pd.read_pickle('data/models/tfidf_8.pkl')

# Load the pre-trained logistic regression model
policy_prob = pd.read_pickle('data/models/logistic_model_8.pkl')

# Transform the text data to TF-IDF features
Xtfidf = vectorizer1.transform(X)

# Predict the top topic for each text entry
data['top_topic'] = policy_prob.predict(Xtfidf)

# Predict the probability for each class
policy_probs = policy_prob.predict_proba(Xtfidf)

# Add a column for each class's probability
for i, topic in enumerate(policy_prob.classes_):
    data[topic] = policy_probs[:,i]

#assign the topic with the highest predicted probability as the predicted topic
data['predicted_topic'] = policy_prob.classes_[np.argmax(policy_probs, axis=1)]

#send dataframe column name
title = "Predicated Sentence Topic Content in June 2023 Parliament Sample"
count_bar_plot('predicted_topic', title)

""")
st.image('Topic Content in June 2023 Parliament Sample.png', 'Chart created with above code')
