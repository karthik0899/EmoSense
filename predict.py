import os
import zipfile
import scipy
import transformers
import numpy as np
import pandas as pd
from datasets import Dataset,load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from accelerate import PartialState
import plotly.graph_objects as go
import plotly.io as pio



os.environ['KAGGLE_CONFIG_DIR'] = 'EmoSense'
os.system("kaggle datasets download -d karthikrathod/emosense-models")

# Specify the path to the zip file
zip_path = 'emosense-models.zip'

# Specify the directory where you want to extract the contents of the zip file
extract_dir = 'emosense-models'

# Open the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents of the zip file to the specified directory
    zip_ref.extractall(extract_dir)

df_emos = pd.read_csv("https://raw.githubusercontent.com/karthik0899/EmoSense/main/VAD_values.csv")

 
def get_path(x):
  
  # Specify the filename or file path
  filename = x # Change this to your desired file name or path

  # Get the absolute path of the file
  file_path = os.path.abspath(filename)

  # Print the file path
  return file_path

model_V = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_V'))
tokenizer_V = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_V'))
trainer_V = Trainer(model=model_V)

model_A = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/model_A'))
tokenizer_A = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_A'))
trainer_A = Trainer(model=model_A)

model_D = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_D'))
tokenizer_D = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_D'))
trainer_D = Trainer(model=model_D)

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


def predict_vad(sentence:str):

    dataset = Dataset.from_pandas(pd.DataFrame({'text':[sentence]}),preserve_index=False) # converting input text to dataset

    #====================================MODEL_V===============================================
    # model_V = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_V'))
    # tokenizer_V = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_V'))
    # trainer_V = Trainer(model=model_V)
    def tokenize_function_V(examples):
      return tokenizer_V(examples["text"], truncation=True) # Tokenization function to tokenize the input text
    def pipeline_prediction_V(dataset):

      tokenized_datasets_V = dataset.map(tokenize_function_V)
      raw_pred, _, _ = trainer_V.predict(tokenized_datasets_V) # predicting the specific varaible using respective model and tokenizer
      return(raw_pred[0][0])
    #=======================================MODEL_A===============================================
    # model_A = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/model_A'))
    # tokenizer_A = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_A'))
    # trainer_A = Trainer(model=model_A)
    def tokenize_function_A(examples):
      return tokenizer_A(examples["text"], truncation=True) 
    def pipeline_prediction_A(dataset):

      tokenized_datasets_A = dataset.map(tokenize_function_A)
      raw_pred, A, B = trainer_A.predict(tokenized_datasets_A) 
      return(raw_pred[0][0])
    #======================================MODEL_D================================================
    # model_D = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_D'))
    # tokenizer_D = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_D'))
    # trainer_A = Trainer(model=model_A)

    def tokenize_function_D(examples):
      return tokenizer_D(examples["text"], truncation=True) 
    def pipeline_prediction_D(dataset):


      tokenized_datasets_D = dataset.map(tokenize_function_D)
      raw_pred, _, _ = trainer_D.predict(tokenized_datasets_D) 
      return(raw_pred[0][0])
    Vhat = np.round((range_scaler_temp(np.round(pipeline_prediction_V(dataset),3))),2)
    Ahat = np.round((range_scaler_temp(np.round(pipeline_prediction_A(dataset),3))),2)
    Dhat = np.round((range_scaler_temp(np.round(pipeline_prediction_D(dataset),3))),2)
    return Vhat,Ahat,Dhat

def plot_VDA(v,a,d):
    # Sample data
    V = v
    A = a
    D = d

    # Assigning colors based on positive/negative values
    colors = ['rgb(255, 102, 102)', 'rgb(255, 204, 102)', 'rgb(102, 178, 255)']

    # Creating the bar plot
    fig = go.Figure(data=go.Bar(
        x=[V, A, D],
        y=[1, 2, 3],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(
                color='rgba(58, 71, 80, 1)',
                width=1.5
            )
        ),
        width=0.4
    ))

    # Adding a vertical dotted line at x=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=0,
        y1=4,
        line=dict(
            color="rgb(80, 80, 80)",
            width=1,
            dash="dot"
        )
    )

    # Adding a background gradient
    fig.update_layout(
        title="Bar Plot",
        xaxis=dict(
            range=[-1, 1],  # Adjust the x-axis range as needed
            zeroline=True,
            zerolinecolor='rgb(80, 80, 80)',
            zerolinewidth=1,
            showgrid=True,  # Show grid
            gridwidth=1,  # Set grid width
            gridcolor='LightGrey',  # Set grid color
            dtick=0.1,  # Set distance between grid lines
            showticklabels=True,
            ticks='outside',
            tickcolor='rgb(80, 80, 80)',
            tickwidth=1,
            ticklen=10,
            tickfont=dict(
                size=14,
                color='rgb(80, 80, 80)'
            )
        ),
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=['V', 'A', 'D'],
            showgrid=True,  # Show grid
            gridwidth=1,  # Set grid width
            gridcolor='LightGrey',  # Set grid color
            tickfont=dict(
                size=14,
                color='rgb(80, 80, 80)'
            )
        ),
        showlegend=False,
        plot_bgcolor='rgba(248, 249, 250, 1)',
        paper_bgcolor='rgba(248, 249, 250, 1)',
        margin=dict(
            l=40,
            r=20,
            t=40,
            b=20
        )
    )

    # Adding annotations for values above the bars
    annotations = [
        dict(
            x=x,
            y=y + 0.3,  # Adjust the y-coordinate to position the value above the bar
            text=str(round(x, 3)),
            xanchor='center',
            yanchor='bottom',
            showarrow=False,
            font=dict(
                size=16,
                color='rgb(80, 80, 80)'
            )
        )
        for x, y in zip([V, A, D], [1, 2, 3])
    ]
    
    fig.update_layout(
    width=800,
    height= 250
        )
    fig.update_layout(annotations=annotations)

    # Adding a title and axis labels
    fig.update_layout(
        title={
            'text': 'Valence, Arousal, and Dominance Values',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=24,
                color='rgb(80, 80, 80)'
            )
        },
        xaxis_title='Values',
        yaxis_title=None,
        font=dict(
            family='Arial',
            size=16,
            color='rgb(80, 80, 80)'
        )
    )

    # Displaying the plot
    fig.show(autosize=False)
  


def plot_emotions(df_emos, pred_vda, top5_emos):
    # Filter the dataframe to only include the top 5 emotions
    df_top5 = df_emos[df_emos['Emotion'].isin(top5_emos)]

    # Calculate distances and find the closest emotion
    distances = []
    for i in range(len(df_top5)):
        dist = scipy.spatial.distance.euclidean([pred_vda[0], pred_vda[1], pred_vda[2]], 
                                  [df_top5.iloc[i]['V_MEAN'], df_top5.iloc[i]['D_MEAN'], df_top5.iloc[i]['A_MEAN']])
        distances.append(dist)
    closest_index = np.argmin(distances)

    # Create scatter plot for top 5 emotions
    fig = go.Figure()

    # Add predicted VDA point
    fig.add_trace(go.Scatter3d(
        x=[pred_vda[0]],
        y=[pred_vda[1]],
        z=[pred_vda[2]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',  # set color to bright red
        ),
        name='Predicted VDA'
    ))

    # Add top 5 emotions
    for i in range(len(df_top5)):
        # Color mapping from A_SD (0 to 1) to colors (red to blue)
        color = 'rgb({}, 0, {})'.format(int((1 - df_top5.iloc[i]['A_SD']) * 255),
                                         int(df_top5.iloc[i]['A_SD'] * 255))
        # Opacity mapping from dominance (-1 to 1) to opacity (0 to 1)
        opacity = (df_top5.iloc[i]['D_MEAN'] + 1) / 2

        fig.add_trace(go.Scatter3d(
            x=[df_top5.iloc[i]['V_MEAN']],
            y=[df_top5.iloc[i]['D_MEAN']],
            z=[df_top5.iloc[i]['A_MEAN']],
            mode='markers',
            marker=dict(
                size=np.mean([df_top5.iloc[i]['V_SD'], df_top5.iloc[i]['D_SD'], df_top5.iloc[i]['A_SD']]) * 100,  # average SD for marker size
                color=color,
                sizemode='diameter',
                opacity=opacity  # make spheres translucent based on dominance
            ),
            text=df_top5.iloc[i]['Emotion'],
            name=df_top5.iloc[i]['Emotion']
        ))

        # Add lines from predicted VDA point to emotion means
        fig.add_trace(go.Scatter3d(
            x=[pred_vda[0], df_top5.iloc[i]['V_MEAN']],
            y=[pred_vda[1], df_top5.iloc[i]['D_MEAN']],
            z=[pred_vda[2], df_top5.iloc[i]['A_MEAN']],
            mode='lines',
            line=dict(
                color='orange' if i == closest_index else 'black',  # set color to orange if this is the closest emotion
                width=2
            ),
            hovertext=f'Distance: {distances[i]:.2f}'
        ))

    # Set the title and axis labels
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Valence', range=[-1,1]),  # set range to [-1,1]
            yaxis=dict(title='Dominance', range=[-1,1]),  # set range to [-1,1]
            zaxis=dict(title='Arousal', range=[-1,1]),  # set range to [-1,1]
            bgcolor='rgba(255, 255, 255, 0)'  # set transparent background
        ),
        title_text='3D Scatter Plot of Emotions',
        paper_bgcolor='rgba(0,0,0,0)',  # set transparent paper_bgcolor
        plot_bgcolor='rgba(0,0,0,0)',  # set transparent plot_bgcolor
        autosize=True,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ),
    )

    # Show the plot
    fig.show()

    # Save the plot
    pio.write_html(fig, file='Emotions.html', auto_open=True)




def classify_emotions(X):
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
    v_valence, v_arousal, v_dominance = X
    point = np.array([v_valence, v_arousal, v_dominance])
    
    # Extract the means and standard deviations for each emotion
    emotion_names = dataset['Emotion'].unique()
    means = np.zeros((len(emotion_names), 3))
    stds = np.zeros((len(emotion_names), 3))
    for i, emotion in enumerate(emotion_names):
        sub_df = dataset[dataset['Emotion'] == emotion]
        means[i] = sub_df[['V_MEAN', 'A_MEAN', 'D_MEAN']].values
        stds[i] = sub_df[['V_SD', 'A_SD', 'D_SD']].values

    # Calculate the Mahalanobis distances between the point and the centers of each ellipsoid
    distances = []
    for i in range(len(means)):
        inv_cov_matrix = np.linalg.inv(np.diag(stds[i]**2))  # invert the covariance matrix
        diff = point - means[i]  # calculate the difference between the point and the mean
        distance = np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))
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
    plot_VDA(v_valence, v_arousal, v_dominance) 
    # Print the result
    print("The point [Valence=", v_valence, ", Arousal=", v_arousal, ", Dominance=", v_dominance, "] closely resembles the following emotions with the following intensities:")
    for i, emotion in enumerate(top5_emotions):
        print(emotion, ":", intensities[i], "%")
    
    plot_emotions(df_emos,[v_valence, v_arousal, v_dominance],top5_emotions)
