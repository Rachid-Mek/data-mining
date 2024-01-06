import pandas as pd
import streamlit as st
from main import execute_knn , execute_Dt, execute_Rf, run_dbscan, run_kmeans
from Prep_dataset1 import Preprocessing, Preprocessing_1

#----------------------------------Supervised Analysis-------------------------------------#
def toggle_other_buttons_4(button_name):
    if button_name == "knn":
        st.session_state["knn"] = 1
        st.session_state["decision_tree"] = 0
        st.session_state["random_forest"]=0 
        st.session_state["comparaison_algo"]=0
    elif button_name == "decision_tree":
        st.session_state["decision_tree"] = 1
        st.session_state["knn"]=0
        st.session_state["random_forest"]=0
        st.session_state["comparaison_algo"]=0
    elif button_name == "random_forest":
        st.session_state["random_forest"] = 1
        st.session_state["knn"] =0
        st.session_state["decision_tree"]=0
        st.session_state["comparaison_algo"]=0
    elif button_name == "comparaison_algo":
        st.session_state["comparaison_algo"] = 1
        st.session_state["knn"] =0
        st.session_state["decision_tree"]=0
        st.session_state["random_forest"]=0


def dataset_options():
    dataset = Preprocessing_1()
    options = {}

    # Create three columns for each row
    col1, col2, col3 = st.columns(3)

    for i, column in enumerate(['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']):
        # Use the appropriate column based on the index
        if i % 3 == 0:
            current_col = col1
        elif i % 3 == 1:
            current_col = col2
        else:
            current_col = col3

        # Add selectbox to the current column
        options[column] = current_col.selectbox(column, [''] + list(dataset[column].unique()), key=column)

    return options

@st.cache_data(show_spinner=False)
def cached_execute_knn(k, distance_function):
    return execute_knn(k,distance_function)

@st.cache_data(show_spinner=False)
def cached_execute_Dt(min_samples_split, max_depth):
    return execute_Dt(min_samples_split, max_depth)

@st.cache_data(show_spinner=False)
def cached_execute_Rf(n_trees, min_samples_split, max_depth, n_features):
    return execute_Rf(n_trees, min_samples_split, max_depth, n_features)

@st.cache_data(show_spinner=False) 
def load_dataset_unsupervised():
    st.session_state["dataset_0"],_ = Preprocessing(0.7 , Norm=0)
    st.session_state["dataset_1"],_ = Preprocessing(0.7 , Norm=1)
    st.session_state["dataset_2"],_ = Preprocessing(0.7 , Norm=2)


def supervised_analysis():
    st.title("Supervised Analysis Page")
    st.sidebar.title("Supervised Analysis")
    # Add buttons to perform data manipulation in the sidebar
    if "knn" not in st.session_state:
        st.session_state["knn"] = 0
    if "decision_tree" not in st.session_state:
        st.session_state["decision_tree"] = 0
    if "random_forest" not in st.session_state:
        st.session_state["random_forest"] = 0
    if "comparaison_algo" not in st.session_state:
        st.session_state["comparaison_algo"] = 0

    k_nearest_neighbors = st.sidebar.button("K-Nearest Neighbors",key="k_nearest_neighbors", use_container_width=True, on_click=toggle_other_buttons_4 , args=["knn"])
    decision_tree = st.sidebar.button("Decision Tree",key="decision tree", use_container_width=True, on_click=toggle_other_buttons_4 , args=["decision_tree"])
    random_forest = st.sidebar.button("Random Forest",key="random forest", use_container_width=True, on_click=toggle_other_buttons_4 , args=["random_forest"])
    comparaison = st.sidebar.button("Comparaison",key="comparaison", use_container_width=True, on_click=toggle_other_buttons_4 , args=["comparaison_algo"])
    return_home = st.sidebar.button("Return Home",use_container_width=True)

    if return_home:
        st.session_state.page = "welcome"

#--------------------------------------------KNN PAGE -----------------------------------------------#
    if k_nearest_neighbors or st.session_state["knn"]:
        st.subheader(f"Working On - k-Nearest Neighbors")
        # add input text to get the number of k to execute the knn algorithm
        k = st.slider("Select the number of k", 1, 30, 9)
        distance = st.selectbox("Select distance function", ["Euclidean", "Manhattan", "Cosine"])
        # execute the knn algorithm
        plt, conf_mat, df_metrics, knn_classifier = cached_execute_knn(k, distance_function=distance)
        # drop box to choose if display the confusion matrix as plt or as a table
        display_conf_mat = st.selectbox("Display confusion matrix as", ["Table", "Plot"])
        if display_conf_mat == "Table":
            st.subheader("Confusion Matrix")
            st.table(conf_mat)
        elif display_conf_mat == "Plot":
            st.subheader("Confusion Matrix")
            st.pyplot(plt)
        # display the metrics in a table
        st.subheader("Metrics")
        st.table(df_metrics)
        #-----------------------------------------predictions part-----------------------------------------#
        st.markdown("""---""")
        st.subheader(f'Predictions using k-Nearest Neighbors classifier')
        selected_values = dataset_options()
        predict_button = st.button("Predict", use_container_width=True)
        if predict_button:
            if any(value == '' for value in selected_values.values()):
                st.error("Please fill in all the fields")
            else:
                prediction = knn_classifier.predict([list(selected_values.values())])
                st.success(f"The predicted class is : {int(prediction[0])}")
            
    elif decision_tree or st.session_state["decision_tree"]:
        st.subheader(f"Working On - decision_tree")
        # minimum number of samples to split an internal node
        min_samples_split = st.slider("Select the minimum number of samples to split an internal node", 2, 20, 10)
        # maximum depth of the tree
        max_depth = st.slider("Select the maximum depth of the tree", 20, 150, 50)        
        plt, conf_mat, df_metrics, dt = cached_execute_Dt(min_samples_split, max_depth)
        display_conf_mat = st.selectbox("Display confusion matrix as", ["Table", "Plot"])
        if display_conf_mat == "Table":
            st.subheader("Confusion Matrix")
            st.table(conf_mat)
        elif display_conf_mat == "Plot":
            st.subheader("Confusion Matrix")
            st.pyplot(plt)
        st.subheader("Metrics")
        st.table(df_metrics)
        #-----------------------------------------predictions part-----------------------------------------#
        st.markdown("""---""")
        st.subheader(f'Predictions using decision Tree classifier')
        selected_values = dataset_options()
        predict = st.button("Predict",use_container_width=True)
        if predict:
            if any(value == '' for value in selected_values.values()):
                st.error("Please fill in all the fields")
            else:
                prediction = dt.predict([list(selected_values.values())])
                st.success(f"The predicted class is : {int(prediction[0])}")

    elif random_forest or st.session_state["random_forest"]:
        st.subheader(f"Working On - Random Forest")
        # number of trees
        n_trees = st.slider("Select the number of trees", 5, 20, 10)
        # minimum number of samples to split an internal node
        min_samples_split = st.slider("Select the minimum number of samples to split an internal node", 2, 20, 10)
        # maximum depth of the tree
        max_depth = st.slider("Select the maximum depth of the tree", 20, 150, 50)

        # execute the Random Forest algorithm
        plt, conf_mat, df_metrics , rf = cached_execute_Rf(n_trees, min_samples_split, max_depth, n_features=None)
        display_conf_mat = st.selectbox("Display confusion matrix as", ["Table", "Plot"])
        if display_conf_mat == "Table":
            st.subheader("Confusion Matrix")
            st.table(conf_mat)
        elif display_conf_mat == "Plot":
            st.subheader("Confusion Matrix")
            st.pyplot(plt)
        st.subheader("Metrics")
        st.table(df_metrics)
        #-----------------------------------------predictions part-----------------------------------------#
        st.markdown("""---""")
        st.subheader(f'Predictions using random forest classifier')
        selected_value = dataset_options()
        # button to execute the algorithm and get the prediction
        predict = st.button("Predict",use_container_width=True)

        if predict:
            # if any of the fields is empty, display an error message
            if any(value == '' for value in selected_value.values()):
                st.error("Please fill in all the fields")
            else:
                # get the prediction
                prediction = rf.predict([list(selected_value.values())])
                st.success(f"The predicted class is : {int(prediction[0])}")

    elif comparaison or st.session_state["comparaison_algo"]:
        st.subheader(f"Comparison between algorithms")
        # number of k
        k = st.slider("Select the number of k", 1, 20, 5)
        # distance function
        distance = st.selectbox("Select distance function", ["Euclidean", "Manhattan", "Cosine"])
        # number of trees
        n_trees = st.slider("Select the number of trees", 5, 25, 10)
        # minimum number of samples to split an internal node
        min_samples_split = st.slider("Select the minimum number of samples to split an internal node", 1, 20, 5)
        # maximum depth of the tree
        max_depth = st.slider("Select the maximum depth of the tree", 5, 20, 10)

        # execute all and getting the metrics as a dataframe
        metric = execute_all(k , min_samples_split, max_depth, n_trees, distance_function=distance, n_features=None)
        # display the metrics
        st.table(metric)


def execute_all(k , min_samples_split, max_depth, n_trees, distance_function='Euclidean', n_features=None):
    '''this function will return a dataframe containing all the metrics of the 3 algorithms'''
    fig, conf_mat, df_metrics_knn , knn_classifier = cached_execute_knn(k,distance_function)
    fig, conf_mat, df_metrics_dt , decision_tree = execute_Dt(min_samples_split, max_depth, n_features)
    fig, conf_mat, df_metrics_rf , random_forest = cached_execute_Rf(n_trees, min_samples_split, max_depth, n_features)

    # get all the metrics in one dataframe
    df_metrics = pd.concat([df_metrics_knn, df_metrics_dt, df_metrics_rf], axis=0)
    df_metrics.index = ['KNN', 'Decision Tree', 'Random Forest']
    return df_metrics


#----------------------------------Unsupervised Analysis-------------------------------------#
def toggle_other_buttons_5(button_name):
    if button_name == "kmeans":
        st.session_state["kmeans"] = 1
        st.session_state["dbscan"]=0 
 
    elif button_name == "dbscan":
        st.session_state["dbscan"] = 1
        st.session_state["kmeans"] =0

def unsupervised_clustering():
    st.title("Unsupervised clustering Page")
    st.sidebar.title("Unsupervised   clustering")
    # Add buttons to perform data manipulation in the sidebar
    if "kmeans" not in st.session_state:
        st.session_state["kmeans"] = 0
    if "dbscan" not in st.session_state:
        st.session_state["dbscan"] = 0

    kmeans = st.sidebar.button("K-Means",key="k_means", use_container_width=True, on_click=toggle_other_buttons_5 , args=["kmeans"])
    dbscan = st.sidebar.button("DBSCAN",key="dbscan_", use_container_width=True, on_click=toggle_other_buttons_5 , args=["dbscan"])
    return_home = st.sidebar.button("Return Home",use_container_width=True)

    if return_home:
        st.session_state.page = "welcome"

    if kmeans or st.session_state["kmeans"]:
        st.title(f"Working On - K-Means")
        normalization = st.selectbox("Select the type of normalization", ["No normalization", "Min-Max normalization", "Normalizer"])
        normalization = 0 if normalization == "No normalization" else 1 if normalization == "Min-Max normalization" else 2
        dataset = st.session_state["dataset_0"] if normalization == 0 else st.session_state["dataset_1"] if normalization == 1 else st.session_state["dataset_2"]
        k = st.slider("Select the number of k", 2, 30, 1) # add input text to get the number of k to execute the knn algorithm
        dimention =st.selectbox("Select the dimention for visualisation :",["2D","3D"])
        if dimention:
            dimention = 2 if dimention == "2D" else 3
        launch_kmeans = st.button("Launch",use_container_width=True)
        
        if launch_kmeans:
       
            kmeans ,metrics, chart = run_kmeans(k , dataset , dimention)
            st.success("Metrics")
            st.table(metrics)
            st.pyplot(chart, use_container_width=True)

                
    elif dbscan or st.session_state["dbscan"]:
        st.title(f"Working On - DBSCAN")
        # choose type of normalization , 0 : no normalization , 1 : min-max normalization , 2 : use Normalizer
        normalization = st.selectbox("Select the type of normalization", ["No normalization", "Min-Max normalization", "Normalizer"])
        normalization = 0 if normalization == "No normalization" else 1 if normalization == "Min-Max normalization" else 2
        dataset = st.session_state["dataset_0"] if normalization == 0 else st.session_state["dataset_1"] if normalization == 1 else st.session_state["dataset_2"]
        eps = st.slider("Select the eps",0.1, 5.0, 0.1 )
        # minimum number of samples to split an internal node
        min_samples = st.slider("Select the minimum number of samples", 1, 30, 1)
        launch_dbscan = st.button("Launch",use_container_width=True)
        if launch_dbscan:
            # execute the Random Forest algorithm
            dbscan ,metrics,chart = run_dbscan(eps, min_samples , dataset)
            if dbscan == 0 and metrics==0 and chart==0:
                    st.error("The algorithm could not find any clusters")
            else:
                st.success("L'algorithme a trouvé des clusters , et voici les métriques d'evaluation :")
                st.table(metrics)
                st.pyplot(chart, use_container_width=True)