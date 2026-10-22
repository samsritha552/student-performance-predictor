# Student Performance Predictor 📚

## About the Project
The Student Performance Predictor is a beginner-friendly Machine Learning project that predicts a student's final grade based on study time, past grades, failures, and absences.

The project uses Python and Scikit-learn to train a Linear Regression model and Streamlit to create an interactive web application.

## Features
- Predicts a student's final grade.
- Uses student academic and attendance information.
- Displays model evaluation metrics.
- Shows actual vs. predicted grades in a graph.
- Provides an interactive web dashboard.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Dataset
This project uses the Student Performance dataset from the UCI Machine Learning Repository.

Dataset source: https://archive.ics.uci.edu/dataset/320/student+performance

## How to Run the Project

1. Download or clone this repository.
2. Install the required libraries:

   
   python -m pip install -r requirements.txt
   

3. Run the Streamlit application:

   
   python -m streamlit run app.py
   

4. Open the local URL displayed in your terminal.

## Model Evaluation
The current model uses Linear Regression.

- Mean Absolute Error (MAE): approximately 1.34
- R² Score: approximately 0.78

These results are from the current train-test split and may vary if the data or model settings change.

## Disclaimer
This project is for educational purposes. Predictions are estimates and should not be used to make important decisions about students.
