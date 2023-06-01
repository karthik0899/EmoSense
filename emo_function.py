# ----------------------------------------------------------------------------------Nessesary Libraries------------------------------------------------------------------
import re
import string
import pickle
import os
import numpy as np
import pandas as pd
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.preprocessing import MinMaxScaler
# os.system("pip install mplcyberpunk")
# import mplcyberpunk as mlp
# plt.style.use("cyberpunk")

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import mplcyberpunk as mlp
plt.style.use('cyberpunk')

df_emos = pd.read_csv("https://raw.githubusercontent.com/karthik0899/EmoSense/main/VAD_values.csv")
#-------------------------------------------------------------------Data Information ---------------------------------------------------------------------------------


import numpy as np
import pandas as pd

def INFO(df):
    """
    Generates an information table summarizing the columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame for which the information table will be generated.

    Returns:
        pandas.DataFrame: An information table summarizing the columns of the DataFrame.
        
    Description:
        The INFO function takes a pandas DataFrame as input and generates an information table that provides a summary of each column in the DataFrame. The information table includes details such as column name, data type, column type (categorical or numerical), total number of rows, count of missing values, percentage of missing values, number of unique values, ratio of rows to unique values (if applicable), maximum value (for numerical columns), and minimum value (for numerical columns).

    Example:
        >>> import pandas as pd
        >>> data = {'Name': ['John', 'Jane', 'Mike'],
        ...         'Age': [30, 25, 35],
        ...         'Gender': ['Male', 'Female', 'Male']}
        >>> df = pd.DataFrame(data)
        >>> INFO(df)
            Column Data Type  Column Type  count_rows  Missing  Percent Missing  Number of Uniques  Ratio of Uniques    Max  Min
        0    Name    object  Categorical           3        0              0.0                  3               1.0   John  NaN
        1     Age     int64    Numerical           3        0              0.0                  3               1.0     35   25
        2  Gender    object  Categorical           3        0              0.0                  2                1    NaN  NaN
    """
    
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

# import plotly.graph_objects as go
# def plot_VDA(v,a,d):
#     # Sample data
#     V = v
#     A = a
#     D = d

#     # Assigning colors based on positive/negative values
#     colors = ['rgb(255, 102, 102)', 'rgb(255, 204, 102)', 'rgb(102, 178, 255)']

#     # Creating the bar plot
#     fig = go.Figure(data=go.Bar(
#         x=[V, A, D],
#         y=[1, 2, 3],
#         orientation='h',
#         marker=dict(
#             color=colors,
#             line=dict(
#                 color='rgba(58, 71, 80, 1)',
#                 width=1.5
#             )
#         ),
#         width=0.4
#     ))

#     # Adding a vertical dotted line at x=0
#     fig.add_shape(
#         type="line",
#         x0=0,
#         y0=0,
#         x1=0,
#         y1=4,
#         line=dict(
#             color="rgb(80, 80, 80)",
#             width=1,
#             dash="dot"
#         )
#     )

#     # Adding a background gradient
#     fig.update_layout(
#         title="Bar Plot",
#         xaxis=dict(
#             range=[-1, 1],  # Adjust the x-axis range as needed
#             zeroline=True,
#             zerolinecolor='rgb(80, 80, 80)',
#             zerolinewidth=1,
#             showgrid=True,  # Show grid
#             gridwidth=1,  # Set grid width
#             gridcolor='LightGrey',  # Set grid color
#             dtick=0.1,  # Set distance between grid lines
#             showticklabels=True,
#             ticks='outside',
#             tickcolor='rgb(80, 80, 80)',
#             tickwidth=1,
#             ticklen=10,
#             tickfont=dict(
#                 size=14,
#                 color='rgb(80, 80, 80)'
#             )
#         ),
#         yaxis=dict(
#             tickvals=[1, 2, 3],
#             ticktext=['V', 'A', 'D'],
#             showgrid=True,  # Show grid
#             gridwidth=1,  # Set grid width
#             gridcolor='LightGrey',  # Set grid color
#             tickfont=dict(
#                 size=14,
#                 color='rgb(80, 80, 80)'
#             )
#         ),
#         showlegend=False,
#         plot_bgcolor='rgba(248, 249, 250, 1)',
#         paper_bgcolor='rgba(248, 249, 250, 1)',
#         margin=dict(
#             l=40,
#             r=20,
#             t=40,
#             b=20
#         )
#     )

#     # Adding annotations for values above the bars
#     annotations = [
#         dict(
#             x=x,
#             y=y + 0.3,  # Adjust the y-coordinate to position the value above the bar
#             text=str(round(x, 3)),
#             xanchor='center',
#             yanchor='bottom',
#             showarrow=False,
#             font=dict(
#                 size=16,
#                 color='rgb(80, 80, 80)'
#             )
#         )
#         for x, y in zip([V, A, D], [1, 2, 3])
#     ]
    
#     fig.update_layout(
#     width=800,
#     height= 250
#         )
#     fig.update_layout(annotations=annotations)

#     # Adding a title and axis labels
#     fig.update_layout(
#         title={
#             'text': 'Valence, Arousal, and Dominance Values',
#             'x': 0.5,
#             'y': 0.95,
#             'xanchor': 'center',
#             'yanchor': 'top',
#             'font': dict(
#                 size=24,
#                 color='rgb(80, 80, 80)'
#             )
#         },
#         xaxis_title='Values',
#         yaxis_title=None,
#         font=dict(
#             family='Arial',
#             size=16,
#             color='rgb(80, 80, 80)'
#         )
#     )

#     # Displaying the plot
#     fig.show(autosize=False)




# def classify_emotions(v_valence, v_arousal, v_dominance):
#     """
#     Classify emotions based on valence, arousal, and dominance values.

#     Parameters:
#         v_valence (float): Valence value of the input point.
#         v_arousal (float): Arousal value of the input point.
#         v_dominance (float): Dominance value of the input point.

#     Returns:
#         None. Prints the closest emotions with their intensities.
#     """
#     # Load the dataset from a CSV file
#     dataset = pd.read_csv("https://raw.githubusercontent.com/karthik0899/EmoSense/main/VAD_values.csv")
#     point = np.array([v_valence, v_arousal, v_dominance])
    
#     # Extract the means and standard deviations for each emotion
#     emotion_names = dataset['Emotion'].unique()
#     means = np.zeros((len(emotion_names), 3))
#     stds = np.zeros((len(emotion_names), 3))
#     for i, emotion in enumerate(emotion_names):
#         sub_df = dataset[dataset['Emotion'] == emotion]
#         means[i] = sub_df[['V_MEAN', 'A_MEAN', 'D_MEAN']].values
#         stds[i] = sub_df[['V_SD', 'A_SD', 'D_SD']].values

#     # Calculate the distances between the point and the centers of each ellipsoid
#     distances = []
#     for i in range(len(means)):
#         center = means[i]
#         distance = np.linalg.norm(point - center)
#         distances.append(distance)

#     # Sort the emotions by distance and select the top 5
#     sorted_idx = np.argsort(distances)
#     top5_idx = sorted_idx[:5]
#     top5_emotions = emotion_names[top5_idx]

#     # Calculate the intensities of the top 5 emotions in terms of a percentage
#     max_distance = np.max(distances)
#     intensities = []
#     for i in top5_idx:
#         intensities.append((1 - distances[i] / max_distance) * 100)

#     plot_VDA(v_valence, v_arousal, v_dominance)
#     # Print the result
#     print("The point [Valence=", v_valence, ", Arousal=", v_arousal, ", Dominance=", v_dominance, "] closely resembles the following emotions with the following intensities:")
#     for i, emotion in enumerate(top5_emotions):
#         print(emotion, ":", intensities[i], "%")


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

def range_scaler_temp(value, assumed_max_input=5, assumed_min_input=1):
    """
    Scale the input value to the range of -1 to 1.

    Parameters:
        value (float): Input value to be scaled.
        assumed_max_input (float): Assumed maximum value of the input array. Default is 5.
        assumed_min_input (float): Assumed minimum value of the input array. Default is 1.

    Returns:
        float: Scaled value ranging from -1 to 1.
    """
    value_std = (value - assumed_min_input) / (assumed_max_input - assumed_min_input)
    value_scaled = value_std * (1 - (-1)) + (-1)
    return value_scaled



#-------------------------------------------------------------------Data Preprocessing ---------------------------------------------------------------------------------


import re
import spacy
import pandas as pd

def preprocess_dataframe(df):
    """
    Preprocesses the text column of a dataframe by removing angle brackets and data within them,
    removing links, removing special characters except punctuation marks, and masking names.
    
    Args:
        df (pd.DataFrame): The input dataframe with a text column to be preprocessed.
        
    Returns:
        pd.DataFrame: The preprocessed dataframe with the text column updated.
    """
    
    def mask_names(sentence):
        # Process the sentence with spaCy
        doc = nlp(sentence)

        # Iterate over named entities in the sentence
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                # Replace the name with [NAME]
                sentence = sentence.replace(ent.text, '[NAME]')

        return sentence
    
    # Remove angle brackets and data within < > angle brackets
    df['filtered'] = df['text'].apply(lambda x: re.sub('<[^>]+?>', '', x))
    
    # Remove links
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['filtered'] = df['filtered'].apply(lambda x: re.sub(pattern, '', x))
    
    # Remove special characters except punctuation marks
    pattern = r"[^\w\s?!.,]+"
    df['filtered'] = df['filtered'].apply(lambda x: re.sub(pattern, "", x))
    
    # Load the pre-trained English language model
    nlp = spacy.load('en_core_web_sm')
    
    # Mask names
    df['filtered'] = df['filtered'].apply(lambda x: mask_names(x))
    df = df[df['filtered'] != '']
    df['word_counts'] = df['filtered'].apply(lambda x: len(x.split()))     # Add word count column to the DataFrame

    # Add special character count column to the DataFrame
    df['special_chars_count'] = df['filtered'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))

    return df


def remove_punc(text):
    """
    Removes punctuation marks, single quotes, double quotes, and special quotation marks from the given text.

    Args:
        text (str): The input text to remove punctuation from.

    Returns:
        str: The text with punctuation marks removed.

    Example:
        >>> text = "Hello, World!"
        >>> remove_punc(text)
        'Hello World'
    """
    # Remove punctuation marks except square brackets
    punctuations = string.punctuation.replace("[", "").replace("]", "")
    text = text.translate(str.maketrans("", "", punctuations))

    # Remove single quotes
    text = text.replace("'", "")

    # Remove double quotes
    text = text.replace('"', "")

    # Remove special quotation marks
    text = text.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")

    return text

import string
import contractions

def greedy_preprocess(text):
    """
    Preprocesses text by expanding contractions, removing punctuation marks (except square brackets),
    single quotes, double quotes, special quotation marks, and extra whitespace.
    
    Args:
        text (str): The input text to be preprocessed.
        
    Returns:
        str: The preprocessed text.
    """
    # Expand contractions
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    text = ' '.join(expanded_words)
    
    # Remove punctuation marks except square brackets
    punctuations = string.punctuation.replace("[", "").replace("]", "")
    text = text.translate(str.maketrans("", "", punctuations))
    
    # Remove single quotes
    text = text.replace("'", "")
    
    # Remove double quotes
    text = text.replace('"', "")
    
    # Remove special quotation marks
    text = text.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    
    # Remove extra whitespace
    text = text.replace("  ", " ")
    
    return text

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


def add_emotion_features(dataset, scaled=False, pred=False, inplace=False):
    """
    Adds emotion features to a dataset based on valence, arousal, and dominance values.

    Parameters:
        dataset (pandas.DataFrame): The dataset to which emotion features will be added.
        scaled (bool, optional): Whether the dataset contains scaled (normalized) valence, arousal, and dominance columns. Defaults to False.
            - if scaled is True, original is False
        pred (bool, optional): Whether the dataset contains predicted valence, arousal, and dominance columns. Defaults to False.
        inplace (bool, optional): Whether to modify the dataset in-place or return a modified copy. Defaults to False.

    Returns:
        pandas.DataFrame: The dataset with emotion features added.

    Description:
        The add_emotion_features function adds emotion features to a dataset based on valence (V), arousal (A), and dominance (D) values. The function accepts the dataset as input and modifies it by adding one or more new columns containing the emotion features.

        The function provides flexibility with the following parameters:
        - scaled: If set to True, the function assumes that the dataset contains scaled (normalized) valence, arousal, and dominance columns, and adds the emotion features with the name 'emotion_features_scaled'. If set to False (default), the function assumes the dataset contains true valence, arousal, and dominance values and adds the emotion features with the name 'emotion_features_true'.
        - pred: If set to True, the function assumes the dataset contains predicted valence, arousal, and dominance columns with names suffixed by '_pred' (e.g., 'V_pred', 'A_pred', 'D_pred'). It adds the emotion features with the name 'emotion_features_pred'. If set to False (default), the function does not consider predicted columns.
        - inplace: If set to True, the function modifies the dataset in-place. If set to False (default), the function creates a copy of the dataset and modifies the copy, leaving the original dataset unchanged.

        The function checks the presence of the required columns in the dataset based on the parameter settings. If any required column is missing, it prints a warning message indicating the missing column(s).

        The emotion features are added as new columns in the dataset. Each feature is represented as a tuple of valence, arousal, and dominance values, corresponding to each row in the dataset.

    Example:
        >>> import pandas as pd
        >>> data = {'V': [0.5, 0.8, 0.3], 'A': [0.7, 0.6, 0.9], 'D': [0.2, 0.4, 0.1]}
        >>> df = pd.DataFrame(data)
        >>> add_emotion_features(df, scaled=True)
           V  A  D    emotion_features_scaled
        0  0.5  0.7  0.2  (0.5, 0.7, 0.2)
        1  0.8  0.6  0.4  (0.8, 0.6, 0.4)
        2  0.3  0.9  0.1  (0.3, 0.9, 0.1)
    """

    print(f'scaled(if True/1 then Original is False/0 ): {scaled}  pred: {pred}  inplace: {inplace}')
    if not inplace:
        dataset = dataset.copy()
    
    if scaled:
        v_col = 'V_SCALED'
        a_col = 'A_SCALED'
        d_col = 'D_SCALED'
        name = 'emotion_features_scaled'
        if all(col in dataset.columns for col in [v_col, a_col, d_col]):
            emotion_features = list(zip(dataset[v_col], dataset[a_col], dataset[d_col]))
            dataset[name] = pd.Series(emotion_features)
        else:
            missing_columns = [col for col in [v_col, a_col, d_col] if col not in dataset.columns]
            print(f"The following columns are missing in the dataset: {', '.join(missing_columns)}")
    else:
        v_col = 'V'
        a_col = 'A'
        d_col = 'D'
        name = 'emotion_features_true'
        if all(col in dataset.columns for col in [v_col, a_col, d_col]):
            emotion_features = list(zip(dataset[v_col], dataset[a_col], dataset[d_col]))
            dataset[name] = pd.Series(emotion_features)
        else:
            missing_columns = [col for col in [v_col, a_col, d_col] if col not in dataset.columns]
            print(f"The following columns are missing in the dataset: {', '.join(missing_columns)}")
    
    v_col = 'V'
    a_col = 'A'
    d_col = 'D'
    if pred:
        v_col += '_pred'
        a_col += '_pred'
        d_col += '_pred'
        name = 'emotion_features_pred'
        if all(col in dataset.columns for col in [v_col, a_col, d_col]):
            emotion_features = list(zip(dataset[v_col], dataset[a_col], dataset[d_col]))
            dataset[name] = pd.Series(emotion_features)
        else:
            missing_columns = [col for col in [v_col, a_col, d_col] if col not in dataset.columns]
            print(f"The following columns are missing in the dataset: {', '.join(missing_columns)}")
    
    return dataset
#     if inplace:
#         return None
#     else:
#         return dataset
    
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


    


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_word_cloud(data_frame, column_name, width=1500, height=1000, max_words=200):
    """
    Plots a word cloud of the most frequent words in a given column of a Pandas DataFrame.
    Parameters:
    data_frame (pandas.DataFrame): The DataFrame containing the text column.
    column_name (str): The name of the text column to be analyzed.
    width (int, optional): Width of the word cloud plot. Default is 1500.
    height (int, optional): Height of the word cloud plot. Default is 1000.
    max_words (int, optional): Maximum number of words to include in the word cloud plot. Default is 200.
    max_word_length (int, optional): Maximum word length to include in the word cloud plot. Default is 3.
    """

    text = ' '.join(data_frame[column_name])
    
    # Filter out specific words, e.g., "NAME"
    # Consider converting text to lower case or upper case to ensure accurate comparison
    exclude_words = ["name"] # lower case to match the converted text
    text = ' '.join(word for word in text.lower().split() if word not in exclude_words) # Convert text to lower case
    
    wordcloud = WordCloud(width=width, height=height, background_color='white', max_words=max_words, min_word_length=3).generate(text)
    plt.figure(figsize=(12, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


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

def plot_top_bottom_emotions(dataset, top_n=10, bottom_n=10):
    """
    Plot horizontal bar charts for the top N and bottom N emotions in the dataset.

    Parameters:
        - dataset (DataFrame): The dataset containing the 'emotions' column.
        - top_n (int, optional): Number of top emotions to display. Default is 10.
        - bottom_n (int, optional): Number of bottom emotions to display. Default is 10.

    Returns:
        None

    """
    # Count the occurrences of each emotion
    emotion_counts = dataset['emotions'].value_counts()

    # Sort the counts in descending order
    sorted_counts = emotion_counts.sort_values(ascending=False)

    # Get the top N and bottom N emotions
    top_emotions = sorted_counts.head(top_n)
    bottom_emotions = sorted_counts.tail(bottom_n)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the top N emotions
    axes[0].barh(top_emotions.index, top_emotions.values)
    axes[0].set_title(f'Top {top_n} Emotions')

    # Plot the bottom N emotions
    axes[1].barh(bottom_emotions.index, bottom_emotions.values)
    axes[1].set_title(f'Bottom {bottom_n} Emotions')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    return plt.show()

def plot_top_words_and_special_chars(df, num_words=30, num_chars=30):
    """
    Plots the top most occurring words and special characters and their frequencies in a DataFrame.
    Parameters:
        df (pandas.DataFrame): DataFrame containing a column named 'filtered' representing the text data.
        num_words (int): Number of top words to plot (default: 30).
        num_chars (int): Number of top special characters to plot (default: 30).
    Returns:
        None
    """

    # Remove stopwords from text data
    stop_words = set(stopwords.words('english'))
    df['filtered'] = df['filtered'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # Remove '--', '[NAME]', '[NAME],' and '[NAME].' from text data
    df['filtered'] = df['filtered'].str.replace('--', ' ')
    df['filtered'] = df['filtered'].str.replace('[NAME]', ' ')
    df['filtered'] = df['filtered'].str.replace('[NAME],', ' ')
    df['filtered'] = df['filtered'].str.replace('[NAME].', ' ')

    # Plot the top most occurring words and their frequencies
    top_words = df['filtered'].str.split(expand=True).stack().value_counts()[:num_words]
    plt.figure(figsize=(15, 8))
    sns.barplot(x=top_words.index, y=top_words.values, alpha=0.8)
    plt.title('Top {} Most Occurring Words'.format(num_words))
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Words', fontsize=12)
    plt.xticks(rotation=45)
    # Remove special characters and punctuations from the text
#     df['filtered'] = df['filtered'].str.replace(r'\W', ' ')

#     # Plot the top most occurring words and their frequencies
#     top_words = df['filtered'].str.split(expand=True).stack().value_counts()[:num_words]
#     plt.figure(figsize=(15, 8))
#     sns.barplot(x=top_words.index, y=top_words.values, alpha=0.8)
#     plt.title('Top {} Most Occurring Words'.format(num_words))
#     plt.ylabel('Frequency', fontsize=12)
#     plt.xlabel('Words', fontsize=12)
#     plt.xticks(rotation=45)
    # Plot the top most occurring special characters and their frequencies
    top_special_chars = df['filtered'].str.findall(r'[^\w\s]').explode().value_counts()[:num_chars]
    plt.figure(figsize=(15, 8))
    sns.barplot(x=top_special_chars.index, y=top_special_chars.values, alpha=0.8)
    plt.title('Top {} Most Occurring Special Characters'.format(num_chars))
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Special Characters', fontsize=12)
    return plt.show()

#--------------------------------------------------------------------- Inference-------------------------------------------------------------------------------------

def closest_emotions(vad_tuple,data=df_emos):
    """
    Classify emotions based on valence, arousal, and dominance values.

    Parameters:
        vad_tuple (tuple): Tuple containing valence, arousal, and dominance values.

    Returns:
        str: The closest emotion based on the input VAD values.
    """
    v_valence, v_arousal, v_dominance = vad_tuple
    # Load the dataset from a CSV file
    dataset = data.copy()

    # Calculate the Euclidean distances between the point and the centers of each emotion
    dataset['Distance'] = np.sqrt((dataset['V_MEAN'] - v_valence) ** 2 + (dataset['A_MEAN'] - v_arousal) ** 2 + (dataset['D_MEAN'] - v_dominance) ** 2)

    # Sort the emotions by distance and select the closest one
    closest_emotion = dataset.loc[dataset['Distance'].idxmin(), 'Emotion']
    
    return closest_emotion
# -------------------------------------------------- Model A testing-----------------------------------

def tokenize_function(examples):
     return tokenizer(examples["text"], padding="max_length", truncation=True) 
    
    
def pipeline_prediction(text):
    df=pd.DataFrame({'text':[text]})
    dataset = Dataset.from_pandas(df,preserve_index=False) 
    tokenized_datasets = dataset.map(tokenize_function)
    raw_pred, _, _ = trainer.predict(tokenized_datasets,) 
    return(raw_pred[0][0])

def create_usability_column(data, pipeline_prediction, range_scaler_temp):
    A_pred = {}
    df_emos = pd.read_csv("EmoSense/VAD_values.csv")
    mean_A = df_emos['A_SD'].mean()
    for text in data['text']:
        A_pred[text] = pipeline_prediction(text)

    data['A_pred'] = data['text'].map(A_pred)
    data["A_SCALED"] = data['A'].apply(range_scaler_temp)
    data["pred_A_SCALED"] = data['A_pred'].apply(range_scaler_temp)
    data['diff'] = abs(data['A_SCALED'] - data['pred_A_SCALED'])

    def check_usability(value):
        if value < mean_A / 3:
            return 'usable'
        else:
            return 'not usable'

    data["usability"] = data['diff'].apply(check_usability)

    return data
