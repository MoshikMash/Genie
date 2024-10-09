import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("test_predictions_with_dates_and_features.csv")


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)  # 'cbar=False' removes the color bar/legend
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    st.pyplot(plt.gcf())


def main():
    st.title('Confusion Matrix by Facility and Specialty')

    st.write("""
        The predictions shown are based on the data up to the **2024-07-01** date.
        The test evaluation presented here is for predictions made **from 2024-07-01** onwards.

        The confusion matrix displays **how many times** the model's predictions match the actual outcomes. Each cell in the matrix represents the **number of cases** where the model made a correct or incorrect prediction:

        | True Label ↓ / Predicted Label → | Predicted No (0) | Predicted Yes (1) |
        |----------------------------------|------------------|-------------------|
        | **Actual No (0)**                | True Negative (TN) | False Positive (FP) |
        | **Actual Yes (1)**               | False Negative (FN) | True Positive (TP) |

        - **True Positive (TP)**: The model correctly predicted that a position **would open** within the specified weeks.
        - **True Negative (TN)**: The model correctly predicted that a position **would not open** within the specified weeks.
        - **False Positive (FP)**: The model predicted that a position **would open**, but it did not.
        - **False Negative (FN)**: The model predicted that a position **would not open**, but one did.

        Each number in the matrix represents **how many times** that particular outcome occurred.
    """)

    # Load the data
    df = load_data()

    # Step 1: Facility Combo Box
    facility_options = df['facility'].unique()
    selected_facility = st.selectbox('Select Facility', facility_options)

    # Step 2: Specialty Combo Box (filtered by selected facility)
    filtered_specialty_options = df[df['facility'] == selected_facility]['specialty'].unique()
    selected_specialty = st.selectbox('Select Specialty', filtered_specialty_options)

    # Filter the dataframe based on selections
    filtered_df = df[(df['facility'] == selected_facility) & (df['specialty'] == selected_specialty)]

    # Select the true and predicted labels for confusion matrix
    y_true = filtered_df['position_opened_next_x_weeks']
    y_pred = filtered_df['xgboost_predicted_label']  # Example for one model; change as needed

    # Plot the confusion matrix
    if not filtered_df.empty:
        plot_confusion_matrix(y_true, y_pred,
                              f'Confusion Matrix for {selected_facility} - {selected_specialty} (Predictions from 2024-07-01)')
    else:
        st.write("No data available for the selected facility and specialty combination.")


if __name__ == "__main__":
    main()
