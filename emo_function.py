import re
import pickle

import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.preprocessing import MinMaxScaler

import mplcyberpunk as mlp
plt.style.use("cyberpunk")

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------Data Information ---------------------------------------------------------------------------------


def INFO(df):
    """
    Generates an information table summarizing the columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame to generate information table for.

    Returns:
        pandas.DataFrame: Information table summarizing the columns of the DataFrame.
    """

    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    info = []

    for col in df.columns:
        count_rows = len(df[col])
        NAN_values = df[col].isna().sum()
        percent = (NAN_values / count_rows) * 100
        data_type = type(df[col][0])
        col_type = df[col].dtype
        if col_type not in [int, float]:
            column_type = "Categorical"
            Max = "Not Applicable"
            Min = "Not Applicable"
        else:
            column_type = "Numerical"
            Max = max(df[col])
            Min = min(df[col])
        try:
            n_uniques = df[col].nunique()
            ratio = count_rows / n_uniques
        except:
            n_uniques = "Not Applicable"
            ratio = "Not Applicable"
        info.append([col, data_type, column_type, count_rows, NAN_values, percent, n_uniques, ratio, Max, Min])

    col_info_df = pd.DataFrame(info, columns=['Column', 'Data Type', 'Column Type', 'count_rows', 'Missing', 'Percent Missing', 'Number of Uniques', 'Ratio of Uniques', 'Max', 'Min'])

    return col_info_df

  
#-------------------------------------------------------------------Emotion plotting ---------------------------------------------------------------------------------


def classify_emotions(v_valence, v_arousal, v_dominance):
    """
    Classify emotions based on valence, arousal, and dominance values.

    Parameters:
        v_valence (float): Valence value of the input point.
        v_arousal (float): Arousal value of the input point.
        v_dominance (float): Dominance value of the input point.

    Returns:
        None. Prints the closest emotions with their intensities.
    """
    # Load the dataset from a CSV file
    dataset = pd.read_csv("https://raw.githubusercontent.com/karthik0899/EmoSense/main/VAD_values.csv")
    point = np.array([v_valence, v_arousal, v_dominance])
    
    # Extract the means and standard deviations for each emotion
    emotion_names = dataset['Emotion'].unique()
    means = np.zeros((len(emotion_names), 3))
    stds = np.zeros((len(emotion_names), 3))
    for i, emotion in enumerate(emotion_names):
        sub_df = dataset[dataset['Emotion'] == emotion]
        means[i] = sub_df[['V_MEAN', 'A_MEAN', 'D_MEAN']].values
        stds[i] = sub_df[['V_SD', 'A_SD', 'D_SD']].values

    # Calculate the distances between the point and the centers of each ellipsoid
    distances = []
    for i in range(len(means)):
        center = means[i]
        distance = np.linalg.norm(point - center)
        distances.append(distance)

    # Sort the emotions by distance and select the top 5
    sorted_idx = np.argsort(distances)
    top5_idx = sorted_idx[:5]
    top5_emotions = emotion_names[top5_idx]

    # Calculate the intensities of the top 5 emotions in terms of a percentage
    max_distance = np.max(distances)
    intensities = []
    for i in top5_idx:
        intensities.append((1 - distances[i] / max_distance) * 100)

    # Print the result
    print("The point [Valence=", v_valence, ", Arousal=", v_arousal, ", Dominance=", v_dominance, "] closely resembles the following emotions with the following intensities:")
    for i, emotion in enumerate(top5_emotions):
        print(emotion, ":", intensities[i], "%")

  #-------------------------------------------------------------------Data rescaling ---------------------------------------------------------------------------------

    
def range_scaler(array, assumed_max_input=5, assumed_min_input=1):
    """
    Scale the values in the input array to the range of -1 to 1.

    Parameters:
        array (ndarray): Input array to be scaled.
        assumed_max_input (float): Assumed maximum value of the input array. Default is 5.
        assumed_min_input (float): Assumed minimum value of the input array. Default is 1.

    Returns:
        ndarray: Scaled array with values ranging from -1 to 1.
    """
    array_std = (array - assumed_min_input) / (assumed_max_input - assumed_min_input)
    array_scaled = array_std * (1 - (-1)) + (-1)
    return array_scaled


#-------------------------------------------------------------------Data Preprocessing ---------------------------------------------------------------------------------


def preprocess_dataframe(df):
    
    """
    Preprocesses a DataFrame by performing various data cleaning and filtering operations.

    Parameters:
        df (pandas.DataFrame): DataFrame containing a column named 'text' representing the text data.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """

    df = df[df['text'].str.split().apply(len) >= 4].copy()  # Drop rows with less than 4 words

    # Convert all words to lowercase and remove HTML tags
    df['text'] = df['text'].str.lower().str.replace('<br /><br />', ' ')

    # Replace multiple spaces with a single space
    df['text'] = df['text'].str.replace('\s+', ' ', regex=True)

    # Drop rows with only numerical values
    df = df[~df['text'].str.isnumeric()].copy()
    
    df['word_counts'] = df['text'].apply(lambda x: len(x.split()))

    # Add special character count column to the DataFrame
    df['special_chars_count'] = df['text'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))
    df = df[~((df["V"] == 3.0) & (df["A"] == 3.0) & (df["D"] == 3.0))]
    df = df.reset_index(drop = True)
    
    return df

def get_3_0(df):
    """
    Filters a DataFrame to retrieve rows where V, A, and D values are all equal to 3.0.

    Args:
        df (pandas.DataFrame): The input DataFrame to filter.

    Returns:
        pandas.DataFrame: A new DataFrame containing only rows where V, A, and D are all 3.0.
    """
    # Filter rows where V, A, and D values are all equal to 3.0
    VAD_3_0 = df[(df.V == 3.0) & (df.A == 3.0) & (df.D == 3.0)]
    
    # Reset the index of the filtered DataFrame
    VAD_3_0.reset_index(drop=True, inplace=True)
    
    # Return the filtered DataFrame
    return VAD_3_0

def filter_3_0(df):
    """
    Filters a DataFrame to remove rows where V, A, and D values are all equal to 3.0.

    Args:
        df (pandas.DataFrame): The input DataFrame to filter.

    Returns:
        pandas.DataFrame: A new DataFrame with rows removed where V, A, and D are all 3.0.
    """
    # Filter rows where V, A, or D values are not equal to 3.0
    VAD_filtered = df[(df['V'] != 3.0) | (df['A'] != 3.0) | (df['D'] != 3.0)]
    
    # Reset the index of the filtered DataFrame
    VAD_filtered.reset_index(drop=True, inplace=True)
    
    # Return the filtered DataFrame
    return VAD_filtered

def add_bins_to_dataset(dataset, inplace=False):
    """
    Adds binning columns to the dataset based on the presence of 'V', 'A', and 'D' columns.

    This function takes a dataset and adds binning columns for the 'V', 'A', and 'D' columns. 
    Binning is a process of dividing continuous values into discrete intervals (bins). The binning 
    columns are added to the dataset to represent the binned values of the original columns.

    Args:
        dataset (pandas.DataFrame): The dataset to modify.
        inplace (bool, optional): If True, modifies the dataset in-place. If False, creates a new copy 
            of the dataset. Defaults to False.

    Returns:
        pandas.DataFrame: The modified dataset with the added binning columns.

    Notes:
        The binning process uses the following interval ranges:
        - Bin 0: (1.0, 2.0]
        - Bin 1: (2.0, 3.0]
        - Bin 2: (3.0, 4.0]
        - Bin 3: (4.0, 5.0]

    """

    if not inplace:
        dataset = dataset.copy()

    # Define the columns to check for
    columns = ['V', 'A', 'D']

    # Check for missing columns
    missing_columns = [col for col in columns if col not in dataset.columns]

    for col in missing_columns:
        print(f"Column '{col}' is missing 😥.")

    # Define the bin edges
    bins = [1, 2, 3, 4, 5]

    # Apply binning to the available columns
    if 'V' in dataset.columns:
        dataset['V_bins'] = pd.cut(dataset['V'], bins=bins, labels=False, include_lowest=True)
    if 'A' in dataset.columns:
        dataset['A_bins'] = pd.cut(dataset['A'], bins=bins, labels=False, include_lowest=True)
    if 'D' in dataset.columns:
        dataset['D_bins'] = pd.cut(dataset['D'], bins=bins, labels=False, include_lowest=True)

    # Return the modified dataset
    return dataset

def original_split(df):
    """
    Splits a dataset into three subsets: train, dev, and test.

    This function takes a dataset and splits it into three subsets based on the 'split' column. The 'split' column
    is expected to contain the values 'train', 'dev', or 'test' to indicate the subset membership of each row.

    Args:
        df (pandas.DataFrame): The dataset to split.

    Returns:
        tuple: A tuple containing the train, dev, and test subsets as separate pandas.DataFrames.

    Notes:
        This splitting approach is based on the suggestion by the original EmoBank team.

    """
    train_df = df[df['split'] == 'train'].copy()
    dev_df = df[df['split'] == 'dev'].copy()
    test_df = df[df['split'] == 'test'].copy()
    return train_df, dev_df, test_df

#-------------------------------------------------------------------Data EDA & Analysis ---------------------------------------------------------------------------------



def plot_VAD_histograms(df):
    """
    Plots histograms of Valence, Arousal, and Dominance values in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns 'V', 'A', and 'D' representing Valence, Arousal, and Dominance, respectively.

    Returns:
        None
    """
    
    v_column = None
    a_column = None
    d_column = None
    
    if 'V' in df.columns:
        v_column = 'V'
    elif 'V_MEAN' in df.columns:
        v_column = 'V_MEAN'
    else:
        raise ValueError("No column 'V' or 'V_MEAN' found in the DataFrame.")
    
    if 'A' in df.columns:
        a_column = 'A'
    elif 'A_MEAN' in df.columns:
        a_column = 'A_MEAN'
    else:
        raise ValueError("No column 'A' or 'A_MEAN' found in the DataFrame.")
    
    if 'D' in df.columns:
        d_column = 'D'
    elif 'D_MEAN' in df.columns:
        d_column = 'D_MEAN'
    else:
        raise ValueError("No column 'D' or 'D_MEAN' found in the DataFrame.")
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs[0].hist(df[v_column], color='red')
    axs[0].set_xlabel('Valence')
    axs[0].set_ylabel('Frequency')
    axs[1].hist(df[a_column], color='green')
    axs[1].set_xlabel('Arousal')
    axs[2].hist(df[d_column], color='blue')
    axs[2].set_xlabel('Dominance')
    fig.suptitle('Frequency of VAD Values')
    
    
    return plt.show()




def plot_VAD_boxplot(data):
    """
    Plots a boxplot of the VAD (Valence, Arousal, Dominance) values in the given dataframe.

    Parameters:
    data (pandas.DataFrame): A pandas DataFrame containing columns for V, A, and D values.

    Returns:
    None
    """
    v_column = None
    a_column = None
    d_column = None
    
    if 'V' in data.columns:
        v_column = 'V'
    elif 'V_MEAN' in data.columns:
        v_column = 'V_MEAN'
    else:
        raise ValueError("No column 'V' or 'V_MEAN' found in the DataFrame.")
    
    if 'A' in data.columns:
        a_column = 'A'
    elif 'A_MEAN' in data.columns:
        a_column = 'A_MEAN'
    else:
        raise ValueError("No column 'A' or 'A_MEAN' found in the DataFrame.")
    
    if 'D' in data.columns:
        d_column = 'D'
    elif 'D_MEAN' in data.columns:
        d_column = 'D_MEAN'
    else:
        raise ValueError("No column 'D' or 'D_MEAN' found in the DataFrame.")
    
    plt.figure(figsize=(15,6))
    sns.boxplot(data=data[[v_column, a_column, d_column]], palette='Set3')
    plt.title('Distribution of VAD Values')
    plt.legend(labels=['Valence (V)', 'Arousal (A)', 'Dominance (D)'])

    return plt.show()




def plot_vad_scatter(df):
    """
    Plots a scatter plot of Valence (V) against Arousal (A) with Dominance (D) as the color hue.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns 'V', 'V_MEAN', 'A', 'A_MEAN', 'D', or 'D_MEAN' representing Valence, Arousal, and Dominance, respectively.

    Returns:
        None
    """
    x_column = None
    y_column = None
    hue_column = None

    if 'V' in df.columns:
        x_column = 'V'
    elif 'V_MEAN' in df.columns:
        x_column = 'V_MEAN'
    else:
        raise ValueError("No column 'V' or 'V_MEAN' found in the DataFrame.")

    if 'A' in df.columns:
        y_column = 'A'
    elif 'A_MEAN' in df.columns:
        y_column = 'A_MEAN'
    else:
        raise ValueError("No column 'A' or 'A_MEAN' found in the DataFrame.")

    if 'D' in df.columns:
        df['Dominance_Category'] = np.where(df['D'] < 3.0, 'Low',
                                            np.where(df['D'] == 3.0, 'Medium', 'High'))
        hue_column = 'Dominance_Category'
    elif 'D_MEAN' in df.columns:
        df['Dominance_Category'] = np.where(df['D_MEAN'] < 3.0, 'Low',
                                            np.where(df['D_MEAN'] == 3.0, 'Medium', 'High'))
        hue_column = 'Dominance_Category'
    else:
        raise ValueError("No column 'D' or 'D_MEAN' found in the DataFrame.")

    plt.figure(figsize=(15, 6))
    scatterplot = sns.scatterplot(x=x_column, y=y_column, hue=hue_column, palette=["green", "yellow", "red"], data=df)
    plt.title("Scatter Plot of VAD Values")
    plt.xlabel("Valence (V)")
    plt.ylabel("Arousal (A)")

    # Custom legend with bucketed values
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Low (D < 3.0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='Medium (D = 3.0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='High (D > 3.0)')
    ]
    scatterplot.legend(handles=legend_elements, title='Dominance (D)', loc='best')

    return plt.show()

    

def plot_emotion_distributions(df):
    """
    Plots the distribution of Valence, Arousal, and Dominance in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns 'V', 'A', and 'D' representing Valence, Arousal, and Dominance, respectively.

    Returns:
        None
    """

    v_column = None
    a_column = None
    d_column = None
    
    if 'V' in df.columns:
        v_column = 'V'
    elif 'V_MEAN' in df.columns:
        v_column = 'V_MEAN'
    else:
        raise ValueError("No column 'V' or 'V_MEAN' found in the DataFrame.")
    
    if 'A' in df.columns:
        a_column = 'A'
    elif 'A_MEAN' in df.columns:
        a_column = 'A_MEAN'
    else:
        raise ValueError("No column 'A' or 'A_MEAN' found in the DataFrame.")
    
    if 'D' in df.columns:
        d_column = 'D'
    elif 'D_MEAN' in df.columns:
        d_column = 'D_MEAN'
    else:
        raise ValueError("No column 'D' or 'D_MEAN' found in the DataFrame.")
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    sns.distplot(df[v_column], ax=axs[0], color='red')
    axs[0].set_xlabel('Valence')
    axs[0].set_ylabel('Density')

    sns.distplot(df[a_column], ax=axs[1], color='green')
    axs[1].set_xlabel('Arousal')
    axs[1].set_ylabel('Density')

    sns.distplot(df[d_column], ax=axs[2], color='blue')
    axs[2].set_xlabel('Dominance')
    axs[2].set_ylabel('Density')
    fig.suptitle('VAD Distributions', fontsize=16)

    return plt.show()

    
    
def plot_VAD_pairplot(df):
    """
    Plots a pair plot of Valence, Arousal, and Dominance in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns 'V', 'A', and 'D' representing Valence, Arousal, and Dominance, respectively.

    Returns:
        None
    """

    v_column = None
    a_column = None
    d_column = None
    
    if 'V' in df.columns:
        v_column = 'V'
    elif 'V_MEAN' in df.columns:
        v_column = 'V_MEAN'
    else:
        raise ValueError("No column 'V' or 'V_MEAN' found in the DataFrame.")
    
    if 'A' in df.columns:
        a_column = 'A'
    elif 'A_MEAN' in df.columns:
        a_column = 'A_MEAN'
    else:
        raise ValueError("No column 'A' or 'A_MEAN' found in the DataFrame.")
    
    if 'D' in df.columns:
        d_column = 'D'
    elif 'D_MEAN' in df.columns:
        d_column = 'D_MEAN'
    else:
        raise ValueError("No column 'D' or 'D_MEAN' found in the DataFrame.")
    
    sns.pairplot(df, vars=[v_column, a_column, d_column], height=4)

    return plt.show()


    
def create_heatmap(df):
    """
    Creates a correlation heatmap between the features in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns 'V', 'A', 'D', 'text'.

    Returns:
        None
    """

    v_column = None
    a_column = None
    d_column = None

    if 'V' in df.columns:
        v_column = 'V'
    elif 'V_MEAN' in df.columns:
        v_column = 'V_MEAN'
    else:
        raise ValueError("No column 'V' or 'V_MEAN' found in the DataFrame.")

    if 'A' in df.columns:
        a_column = 'A'
    elif 'A_MEAN' in df.columns:
        a_column = 'A_MEAN'
    else:
        raise ValueError("No column 'A' or 'A_MEAN' found in the DataFrame.")

    if 'D' in df.columns:
        d_column = 'D'
    elif 'D_MEAN' in df.columns:
        d_column = 'D_MEAN'
    else:
        raise ValueError("No column 'D' or 'D_MEAN' found in the DataFrame.")

    # Create a copy of the DataFrame without modifying the original DataFrame
    modified_df = df.copy()

    if 'text' in df.columns:
#         # Add word count column to the DataFrame
#         modified_df['word_counts'] = modified_df['text'].apply(lambda x: len(x.split()))

#         # Add special character count column to the DataFrame
#         modified_df['special_chars_count'] = modified_df['text'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))

        # Select the columns to include in the correlation matrix
        
        columns = [v_column, a_column, d_column, 'word_counts', 'special_chars_count']
    else:
        columns = [v_column, a_column, d_column]
    
    corr_matrix = modified_df[columns].corr()

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)

    # Set the title
    ax.set_title('Correlation Heat Map between the Features')

    return plt.show()



def plot_top_words_and_special_chars(df, num_words=30, num_chars=30):
    """
    Plots the top most occurring words and special characters and their frequencies in a DataFrame.
    Parameters:
        df (pandas.DataFrame): DataFrame containing a column named 'text' representing the text data.
        num_words (int): Number of top words to plot (default: 30).
        num_chars (int): Number of top special characters to plot (default: 30).
    Returns:
        None
    """

    # Remove stopwords from text data
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # Plot the top most occurring words and their frequencies
    top_words = df['text'].str.split(expand=True).stack().value_counts()[:num_words]
    plt.figure(figsize=(15, 8))
    sns.barplot(x=top_words.index, y=top_words.values, alpha=0.8)
    plt.title('Top {} Most Occurring Words'.format(num_words))
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Words', fontsize=12)
    plt.xticks(rotation=45)

    # Plot the top most occurring special characters and their frequencies
    top_special_chars = df['text'].str.findall(r'[^\w\s]').explode().value_counts()[:num_chars]
    plt.figure(figsize=(15, 8))
    sns.barplot(x=top_special_chars.index, y=top_special_chars.values, alpha=0.8)
    plt.title('Top {} Most Occurring Special Characters'.format(num_chars))
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Special Characters', fontsize=12)
    return plt.show()



def plot_VAD_vs_special_chars_scatter(df):
    """
    Plots scatter plots of Valence, Arousal, and Dominance against the count of special characters in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns 'special_chars_count', 'V', 'A', and 'D' representing the count of special characters, Valence, Arousal, and Dominance, respectively.

    Returns:
        None
    """

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    axs[0].scatter(df['special_chars_count'], df['V'], color='red', label='Valence')
    axs[0].set_xlabel('Count of Special Characters')
    axs[0].set_ylabel('Valence')
    axs[0].legend()

    axs[1].scatter(df['special_chars_count'], df['A'], color='green', label='Arousal')
    axs[1].set_xlabel('Count of Special Characters')
    axs[1].set_ylabel('Arousal')
    axs[1].legend()

    axs[2].scatter(df['special_chars_count'], df['D'], color='blue', label='Dominance')
    axs[2].set_xlabel('Count of Special Characters')
    axs[2].set_ylabel('Dominance')
    axs[2].legend()

    return plt.show()


    
def plot_VAD_vs_WordCounts(df):
    """
    Plots scatter plots of Valence, Arousal, and Dominance against word counts in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns 'word_counts', 'V', 'A', and 'D' representing word counts, Valence, Arousal, and Dominance, respectively.

    Returns:
        None
    """

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    axs[0].scatter(df['word_counts'], df['V'], color='red', label='Valence')
    axs[0].set_xlabel('Word Counts')
    axs[0].set_ylabel('Valence')
    axs[0].legend()

    axs[1].scatter(df['word_counts'], df['A'], color='green', label='Arousal')
    axs[1].set_xlabel('Word Counts')
    axs[1].set_ylabel('Arousal')
    axs[1].legend()

    axs[2].scatter(df['word_counts'], df['D'], color='blue', label='Dominance')
    axs[2].set_xlabel('Word Counts')
    axs[2].set_ylabel('Dominance')
    axs[2].legend()


    return plt.show()


    
def plot_word_counts_frequencies(df):
    """
    Plots the histogram of word counts frequencies in a DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing a column named 'word_counts' representing the word counts.

    Returns:
        None
    """

    plt.figure(figsize=(15, 6))
    sns.histplot(df['word_counts'], kde=False)
    plt.title("Word Counts Vs Frequency")
    plt.xlabel("Word Counts")
    plt.ylabel("Frequency")
    plt.show()


    


def plot_word_cloud(data_frame, column_name, width=1500, height=1000, max_words=200):
    """
    Plots a word cloud of the most frequent words in a given column of a Pandas DataFrame.

    Parameters:
    data_frame (pandas.DataFrame): The DataFrame containing the text column.
    column_name (str): The name of the text column to be analyzed.
    width (int, optional): Width of the word cloud plot. Default is 1500.
    height (int, optional): Height of the word cloud plot. Default is 1000.
    max_words (int, optional): Maximum number of words to include in the word cloud plot. Default is 200.
    max_word_lenght will be 3
    """

    text = ' '.join(data_frame[column_name])
    wordcloud = WordCloud(width=width, height=height, background_color='white', max_words=max_words,min_word_length = 3).generate(text)
    plt.figure(figsize=(12,6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return plt.show()

def bins_distribution(df):
    """
    Plots histograms and max count labels for specified columns in the given DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.

    Returns:
        matplotlib.figure.Figure: The resulting figure containing the histograms.

    Note:
        The function checks if the specified columns ('V_bins', 'A_bins', 'D_bins') are present in the DataFrame.
        If a column is not present, it will print a message indicating that the column is not available.
        The figure will be returned only for available columns, and there won't be any blank plots.

    """
    # Determine available columns
    available_columns = []
    for column in ['V_bins', 'A_bins', 'D_bins']:
        if column in df.columns:
            available_columns.append(column)
        else:
            print(f"Column '{column}' is not available 😥.")

    # Count the number of available columns
    num_plots = len(available_columns)

    if num_plots == 0:
        print("No available columns to plot 😭.")
        return None

    # Determine the number of rows and columns for subplots based on available columns
    nrows = 1
    ncols = num_plots

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4))

    # Define the x-labels
    x_labels = ['1-2', '2-3', '3-4', '4-5']

    # Iterate over available columns and plot histograms
    for i, column in enumerate(available_columns):
        counts = df[column].value_counts().sort_index()

        # Set a fixed range for the x-axis
        x_range = np.arange(len(x_labels))

        # Add zero counts for missing bins
        counts = counts.reindex(x_range, fill_value=0)

        ax = axes[i] if num_plots > 1 else axes  # Handle subplot indexing if only one subplot
        ax.bar(counts.index, counts, align='center', edgecolor='black')

        for j, count in enumerate(counts):
            ax.text(j, count, f'Count: {count} \n {np.round((count/len(df[column]))*100,2)}%', ha='center', va='bottom', color='red')

        ax.set_xticks(x_range)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Bins')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{column} Distribution', pad=25)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Return the figure
    return plt.show()
