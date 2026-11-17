# WISDM - Human Activity Recognition

## Project Overview
This project focuses on Human Activity Recognition (HAR) using the WISDM dataset. The goal is to classify human physical activities based on smartphone accelerometer sensor data.

## Dataset
The project uses the WISDM dataset, which contains accelerometer readings collected from participants performing different physical activities.

The activities include:
- Walking
- Jogging
- Upstairs
- Downstairs
- Sitting
- Standing

## Project Files
- `PT1_solution.ipynb`: Main Jupyter Notebook containing data preprocessing, feature extraction, model training, and evaluation.
- `dashboard.py`: Interactive Streamlit dashboard for data exploration and model evaluation.
- `pt1_utils.py`: Helper functions used throughout the project.
- `requirements.txt`: Required Python libraries.
- `data/`: Dataset directory.
- `README.md`: Project documentation.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

Run the following command in the project directory:

```bash
streamlit run dashboard.py
```

## Online Dashboard

Access the interactive dashboard here:

https://wisdm-human-activity-recognition-qvfewa5qtcgskgy2jfhhfb.streamlit.app/

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebook

## Model Evaluation
The project includes model evaluation using:
- Accuracy
- Macro F1-score
- Precision and Recall
- Classification Report
- Confusion Matrix

