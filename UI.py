import streamlit as st
import Dataset_1_logic as dc
from Pages import *
import pandas as pd
from clustering_pages import *

# ---------------------------------------------------------------------------------------------------------
st.set_option('deprecation.showPyplotGlobalUse', False)

# ---------------------------------------------------------------------------------------------------------

# Placeholder functions for data manipulation
def load_dataset(dataset_choice):
    dataset_number = int(dataset_choice.split(" ")[1])
    dataset = dc.import_dataset(dataset_number)
    return dataset
def describe_data(data):
    return data.describe()
def plot_boxplot(data, column):
    return dc.create_box_plot(data, column)
def plot_scatterplot(data, x_column, y_column):
    return dc.create_scatter_plot(data, x_column, y_column)
def plot_histogram(data, column):
    return dc.create_histogram(data, column)
def Replace_missing_values(methode):
    return dc.replace_missing_values(methode)
def Replace_outliers(dataset ,methode):
    return dc.Replace_outliers(methode)
 

# ---------------------------------------------------------------------------------------------------------
def toggle_button(button_name):
    if button_name not in st.session_state:
        st.session_state[button_name] = 1
        return st.session_state[button_name]
    else:
        del st.session_state[button_name]
# ---------------------------------------------------------------------------------------------------------
def toggle_other_buttons(button_name):
    if button_name == "visualisation":
        st.session_state["visualisation"] = 1
        st.session_state["process"] = 0
        st.session_state["description"]=0 
    elif button_name == "description":
        st.session_state["description"] = 1
        st.session_state["process"]=0
        st.session_state["visualisation"]=0
    elif button_name == "process":
        st.session_state["process"] = 1
        st.session_state["visualisation"] =0
        st.session_state["description"]=0    
         
# ---------------------------------------------------------------------------------------------------------
def toggle_other_buttons_2(button_name):
    if button_name == "description_2":
       st.session_state["description_2"] = 1
       st.session_state["Pretraitement_2"]=0
       st.session_state["Visualisation_2"]=0
    elif button_name == "Pretraitement_2":
        st.session_state["Pretraitement_2"] = 1
        st.session_state["Visualisation_2"] =0
        st.session_state["description_2"]=0
    elif button_name == "Visualisation_2":
        st.session_state["Visualisation_2"] = 1
        st.session_state["Pretraitement_2"] = 0
        st.session_state["description_2"]=0     

# ---------------------------------------------------------------------------------------------------------
def toggle_other_buttons_3(button_name): 
    if button_name == "Description_3":
       st.session_state["Description_3"] = 1
       st.session_state["Discretisation_3"]=0
       st.session_state["Apriori_3"]=0
       st.session_state["Regles_associations_3"]=0
       st.session_state["Predictions_3"]=0
    elif button_name == "Discretisation_3":
        st.session_state["Discretisation_3"] = 1
        st.session_state["Description_3"] =0
        st.session_state["Apriori_3"]=0
        st.session_state["Regles_associations_3"]=0
        st.session_state["Predictions_3"]=0
    elif button_name == "Apriori_3":
        st.session_state["Apriori_3"] = 1
        st.session_state["Description_3"] =0
        st.session_state["Discretisation_3"]=0
        st.session_state["Regles_associations_3"]=0
        st.session_state["Predictions_3"]=0
    elif button_name == "Regles_associations_3":
        st.session_state["Regles_associations_3"] = 1
        st.session_state["Description_3"] =0
        st.session_state["Discretisation_3"]=0
        st.session_state["Apriori_3"]=0
        st.session_state["Predictions_3"]=0
    elif button_name == "Predictions_3":
        st.session_state["Predictions_3"] = 1
        st.session_state["Description_3"] =0
        st.session_state["Discretisation_3"]=0
        st.session_state["Apriori_3"]=0
        st.session_state["Regles_associations_3"]=0
      
# ---------------------------------------------------------------------------------------------------------
def general_overview():
    return_welcome =  st.sidebar.button("Return home" ,use_container_width=True)

    st.title("General Overview")
    # Main button to return to the welcome page
    if return_welcome:
        st.session_state.page = "welcome"
    # Page for Dataset Manipulation
    dataset_choice = st.sidebar.selectbox("Select Dataset", ["Dataset 1", "Dataset 2", "Dataset 3"])
    st.title(f"Dataset Visualisation - {dataset_choice}")

    # Load selected dataset
    dataset = load_dataset(dataset_choice)

    # Display dataset
    st.subheader(f"Selected Dataset : {dataset_choice}")
    st.table(dataset.head(10))

    if dataset_choice == "Dataset 1":
        if "Dataset 1" not in st.session_state:
            st.session_state['Dataset 1'] = 1
        task_option = st.sidebar.selectbox("Select Task", ["Data description", "Plot Boxplot", "Plot Scatterplot", "Plot Histogram"])
    elif dataset_choice == "Dataset 2":
        if "Dataset 2" not in st.session_state:
            st.session_state['Dataset 2'] = 1
        task_option = st.sidebar.selectbox("Select Task", ["Data description", "Plot Boxplot", "Plot Scatterplot", "Plot Histogram"])
    elif dataset_choice == "Dataset 3":
        if "Dataset 3" not in st.session_state:
            st.session_state['Dataset 3'] = 1
        task_option = st.sidebar.selectbox("Select Task", ["Data description","Plot Scatterplot", "Plot Histogram"])

    if task_option == "Data description":

        description_choice = st.sidebar.selectbox("Select description type", ["Data description", "Description table", "Description measures"])
        if description_choice == "Data description":
            st.subheader("Dataset Description")
            st.table(describe_data(dataset))
        elif description_choice == "Description table":
            st.subheader("Dataset Description")
            st.table(dc.Description_table(dataset))
        elif description_choice == "Description measures":
            st.subheader("Dataset Description")
            st.table(dc.Calculate_Measure(dataset))

    elif task_option == "Plot Boxplot":
        st.subheader("Boxplot")
        columns = dataset.columns
        # exclude non-numeric columns
        numeric_columns = dataset._get_numeric_data().columns
        column_for_boxplot = st.selectbox("Select Column for Boxplot", numeric_columns )
        plot_boxplot(dataset, column_for_boxplot)
        st.pyplot()

    elif task_option == "Plot Scatterplot":
        st.subheader("Scatter Plot")
        numeric_columns = dataset._get_numeric_data().columns

        x_column = st.selectbox("Select X-axis Column", numeric_columns)
        y_column = st.selectbox("Select Y-axis Column", numeric_columns)
        plot_scatterplot(dataset, x_column, y_column)
        st.pyplot()

    elif task_option == "Plot Histogram":
        st.subheader("Histogram")
        numeric_columns = dataset._get_numeric_data().columns

        column_for_histogram = st.selectbox("Select Column for Histogram", numeric_columns)
        plot_histogram(dataset, column_for_histogram)
        st.pyplot()
 
# ------------------------------------------------------------ -----------------------------

# Welcome Page
def welcome_page():
    st.title("Welcome to Dataset Manipulation Interface")
    st.subheader("Select your desired Task ")
    
    # Buttons on the welcome page with unique keys
    if st.button("General Overview",use_container_width=True):
        st.session_state.page = "general_overview"
    if st.button("Dataset Manipulation 1", key="dataset_manipulation_1" ,use_container_width=True):
        st.session_state.page = "dataset_manipulation_1"
    if st.button("Dataset Manipulation 2", key="dataset_manipulation_2" ,use_container_width=True):
        st.session_state.page = "dataset_manipulation_2"
    if st.button("Dataset Manipulation 3", key="dataset_manipulation_3" ,use_container_width=True):
        st.session_state.page = "dataset_manipulation_3"
    if st.button("Supervised Analysis", key="supervised_analysis" ,use_container_width=True):
        st.session_state.page = "supervised_analysis"
    if st.button("Unsupervised Analysis", key="unsupervised_analysis" ,use_container_width=True):
        st.session_state.page = "unsupervised_analysis"

# ------------------------------------------- Main app logic  ----------------------------------------------

# Main app logic
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "general_overview":
    general_overview()
elif st.session_state.page == "dataset_manipulation_1":
    dataset_manipulation_dataset1()
elif st.session_state.page == "dataset_manipulation_2":
    dataset_manipulation_dataset2()
elif st.session_state.page == "dataset_manipulation_3":
    dataset_manipulation_dataset3()
elif st.session_state.page == "supervised_analysis":
    supervised_analysis()
elif st.session_state.page == "unsupervised_analysis":
    if "dataset_unsupervised" not in st.session_state:
        st.session_state['dataset_unsupervised'] = 1
        load_dataset_unsupervised()
    unsupervised_clustering()