
import streamlit as st
import Dataset_1_logic as dc
import pandas as pd
import UI as ui
import Dataset_2_logic as d2
import Dataset_3_logic as d3


# ----------------------------------Dataset 1 -----------------------------------------------------------------------
def Normalisation(dataset ,methode):
    if methode == "min-max":
        dataset = dc.normalize_min_max_dataset(dataset)
    elif methode == "z-score":
        dataset = dc.normalize_z_score_dataset(dataset)
    return dataset
def dataset_manipulation_dataset1(): 

    dataset_choice = "Dataset 1"
    st.title(f"Process and visualize {dataset_choice}")

    dataset = ui.load_dataset(dataset_choice)
    if "visualisation" not in st.session_state:
        st.session_state["visualisation"] = 0
    if "description" not in st.session_state:
        st.session_state["description"] = 0
    if "process" not in st.session_state:
        st.session_state["process"] = 0
    
    # Add buttons to perform data manipulation in the sidebar
    visualisation = st.sidebar.button("Data visualisation", key="visualisation1", use_container_width=True,on_click=ui.toggle_other_buttons ,args=["visualisation"] )
    description = st.sidebar.button("Data description", key="description1", use_container_width=True ,on_click=ui.toggle_other_buttons ,args=["description"])
    process = st.sidebar.button("Process data", key="process1", use_container_width=True ,on_click=ui.toggle_other_buttons ,args=["process"])   

    if visualisation or st.session_state["visualisation"] == 1:
        st.subheader("Data visualisation")
        st.write(dataset)
        selected_task = st.selectbox("Select task", ["Plot Boxplot", "Plot Scatterplot", "Plot Histogram"])

        if selected_task == "Plot Boxplot" :
            ui.toggle_button("Plot Boxplot")
            column_for_boxplot = st.selectbox("Select Column for Boxplot", dataset.columns)
            plot=ui.plot_boxplot(dataset, column_for_boxplot)
            st.pyplot(plot)

        elif selected_task == "Plot Scatterplot" :
            ui.toggle_button("Plot Scatterplot")
            x_column = st.selectbox("Select X-axis Column", dataset.columns)
            y_column = st.selectbox("Select Y-axis Column", dataset.columns)
            plot=ui.plot_scatterplot(dataset, x_column, y_column)
            st.pyplot(plot)                 

        elif selected_task == "Plot Histogram" :    
            ui.toggle_button("Plot Histogram")
            column_for_histogram = st.selectbox("Select Column for Histogram", dataset.columns)
            plot=ui.plot_histogram(dataset, column_for_histogram)
            st.pyplot(plot)

    if description or st.session_state["description"]== 1  : 
        selected_description = st.selectbox("Data description", ["Data description", "Description table", "Description measures"])
        
        if selected_description == "Data description":
            ui.toggle_button("Data description")
            st.subheader("Data description")
            st.table(dc.Description_table(dataset))

        elif selected_description == "Description table":
                ui.toggle_button("Description table")
                st.subheader("Description table")
                st.write(dc.Description_measures(dataset))

        elif selected_description == "Description measures":
                ui.toggle_button("Description measures")
                st.subheader("Description measures")
                st.write(dc.Calculate_Measure(dataset))

    if process or  st.session_state["process"]== 1  :
        st.header("Process data :")
        st.subheader("Traitement des valeurs manquantes et aberrantes :")
        methode_R_VM=st.selectbox(" Méthode de remplacement des valeurs manquantes", ["" ,"mean", "median", "mode" ])
        if methode_R_VM== "mean" or methode_R_VM== "median" or methode_R_VM== "mode" : 
            dataset_fresh=dc.import_dataset(1)
            st.session_state['Dataset actualisé'] = dc.replace_missing_values(dataset ,methode_R_VM)
            st.write(dc.replace_missing_values(dataset_fresh ,methode_R_VM))

        methode_R_VA=st.selectbox(" Méthode de remplacement des valeurs aberrantes", ["" ,"remove", "mean", "median" ])
        if methode_R_VA== "remove" or methode_R_VA== "median" or methode_R_VA== "mean" : 

            st.write(dc.Replace_outliers( st.session_state['Dataset actualisé'], methode_R_VA))
            st.session_state['Dataset actualisé'] = dc.Replace_outliers(st.session_state['Dataset actualisé'], methode_R_VA)

        methode_red=st.selectbox("Reduction de la dimensionnalité", ["" ,"Verticale", "Horizontale" ])
        if methode_red== "Verticale"  :
            ui.toggle_button("Verticale")
            st.write(dc.Reduction_V(dataset))
            st.session_state['Dataset actualisé'] = dc.Reduction_V(dataset)
        
        elif methode_red== "Horizontale"  :
            ui.toggle_button("Horizontale")
            corr=st.slider("Coefficient de corrélation", 0.0, 1.0, 0.5)
            if "corr" not in st.session_state:
                st.session_state['corr'] = corr
            st.write(dc.Reduction_H(dataset ,corr))
            st.session_state['Dataset actualisé'] = dc.Reduction_H(dataset ,corr)


        st.subheader("Normalisation des données :")
        methode_N=st.selectbox(" Méthode de normalisation", ["" ,"min-max", "z-score" ])
        if methode_N== "min-max" or methode_N== "z-score" : 
            st.write(Normalisation(st.session_state['Dataset actualisé'], methode_N))
            st.session_state['Dataset actualisé'] = Normalisation(st.session_state['Dataset actualisé'], methode_N)

        # st.subheader("Discrétisation des données :")
        # methode_D=st.selectbox(" Méthode de discrétisation", ["" ,"Equal width", "Equal frequency" ])
        # if methode_D== "Equal width" or methode_D== "Equal frequency" : 
        #     st.write(dc.discretisation(st.session_state['Dataset actualisé'], methode_D))
        #     st.session_state['Dataset actualisé'] = dc.discretisation(st.session_state['Dataset actualisé'], methode_D)


    st.write("-----------------------------------------------------------")
    return_welcome1 = st.sidebar.button("Return home", use_container_width=True)
    if return_welcome1:
        st.session_state.page = "welcome"

    

    
# ---------------------------------------------------------------------------------------------------------
# ----------------------------------Dataset 2 ---------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
def dataset_manipulation_dataset2():
    dataset_choice = "Dataset 2"
    dataset = ui.load_dataset(dataset_choice)
    
    if "description_2" not in st.session_state:
        st.session_state["description_2"] = 0
    if "Visualisation_2" not in st.session_state:
        st.session_state["Visualisation_2"] = 0
    if "Pretraitement_2" not in st.session_state:
        st.session_state["Pretraitement_2"] = 0


    st.title(f"Process and visualize  {dataset_choice}")
    
    Description = st.sidebar.button("Data Description", key="description2", use_container_width=True,on_click=ui.toggle_other_buttons_2 ,args=["description_2"] )
    Pretraitement = st.sidebar.button("Data pretraitement", key="Pretraitement2", use_container_width=True ,on_click=ui.toggle_other_buttons_2 ,args=["Pretraitement_2"])
    Visualisation = st.sidebar.button("Data Visualisation", key="Visualisation2", use_container_width=True ,on_click=ui.toggle_other_buttons_2 ,args=["Visualisation_2"])   
 
    
    return_welcome3 =  st.sidebar.button("Return home" ,use_container_width=True)

    if return_welcome3:
        st.session_state.page = "welcome"
    reload =st.button("Reload the dataset", key="reload_dataset", use_container_width=True)

    if reload :
        if "reload_dataset" not in st.session_state:
            ui.toggle_button("reload_dataset")
            
        dataset = ui.load_dataset(dataset_choice)
        st.session_state['Dataset actualisé'] = dataset

    if Description or st.session_state["description_2"] == 1:
        st.subheader("Data description")
        st.write(dataset)
        st.write(d2.describe_data(dataset))

    elif Pretraitement or st.session_state["Pretraitement_2"] == 1  :
        
        st.session_state['Dataset actualisé'] = dataset

        st.subheader("Data Pretraitement")
        method_TVM= st.selectbox("Traitement des valeurs manquantes", ["" ,"mean", "median", "interpolate", "drop"])
        if method_TVM == "mean" or method_TVM == "median" or method_TVM == "interpolate" or method_TVM == "drop":
            if method_TVM not in st.session_state:
                ui.toggle_button(method_TVM)
              
            st.write(d2.Replace_missing_values( st.session_state['Dataset actualisé']) , method_TVM)
            st.session_state['Dataset actualisé'] = d2.Replace_missing_values(st.session_state['Dataset actualisé'] , method_TVM)
      
        method_TOL= st.selectbox("Traitement des outliers", ["" ,"drop", "median"])
        if method_TOL == "drop" or method_TOL == "median":
            if method_TOL not in st.session_state:
                ui.toggle_button(method_TOL)
                # ui.remove_button("Visualisation")
                # ui.remove_button("Description")
            st.write(d2.Replace_outliers( st.session_state['Dataset actualisé'], method_TOL))
            st.session_state['Dataset actualisé'] = d2.Replace_outliers(st.session_state['Dataset actualisé'], method_TOL)
  
    elif Visualisation or st.session_state["Visualisation_2"] == 1 :
        
        dataset = d2.process_dataset()
        st.session_state['Dataset actualisé'] = d2.process_dataset()
        st.subheader("Visualisation de boxplot") 
        column_for_boxplot = st.selectbox("Select Column for Boxplot",d2.Only_numeric_columns(dataset))
        if column_for_boxplot != "" :
            if "Select Column for Boxplot" not in st.session_state:
                st.session_state['Select Column for Boxplot'] = column_for_boxplot
            plot=d2.plot_boxplot(dataset, column_for_boxplot)
            st.pyplot(plot)
        
        st.subheader("Nombre total des cas confirmés et tests positifs par zones :")
        st.pyplot(d2.Total_confirm_cases_zone(dataset)) 
        
        st.subheader("Évolution des tests COVID-19 pour une zone :")
        zone = st.selectbox("Select zone",d2.Zones(dataset))
        st.pyplot(d2.Evolution_tests_zone(dataset,zone))

        st.subheader("Répartition des cas positifs de COVID-19 par zone et par année :")
        st.pyplot(d2.Repartition_cas_positifs_zone(dataset))

        st.subheader("Rapport entre la population et le nombre de tests effectués :")
        st.pyplot(d2.Rapport_population_tests(dataset))

        st.subheader("Les zones les plus touchées par COVID-19:")
        Number_zones = st.slider("Number of zones", 1, 10, 5)
        table_zone ,plt_zone = d2.Impacted_zones(dataset,Number_zones)
        display_opt=st.selectbox("select table or plot", ["table", "plot"])
        if display_opt == "table":
            st.table(table_zone)
        elif display_opt == "plot":
            st.pyplot(plt_zone)
      
        st.subheader("Nombre total des cas confirmés, tests effectués et tests positifs pour une periode :")
        period = st.selectbox("Select period", dataset["time_period"].unique())
        st.pyplot(d2.Total_confirm_cases(dataset,period))
        st.subheader("Generate ratio")
        # select many periods
        periods = st.multiselect("Select periods", dataset["time_period"].unique())
        generate = st.button("Generate" , use_container_width=True )
        if generate :
            st.pyplot(d2.Ratio(dataset,periods))

# ---------------------------------------------------------------------------------------------------------
# ----------------------------------Dataset 3 -------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

def dataset_manipulation_dataset3():
    dataset_choice = "Dataset 3"
    st.title(f"Process and visualize {dataset_choice}")
    dataset = ui.load_dataset("Dataset 3")

    if "Description_3" not in st.session_state:
        st.session_state["Description_3"] = 0
    if "Discretisation_3" not in st.session_state:
        st.session_state["Discretisation_3"] = 0
    if "Apriori_3" not in st.session_state:
        st.session_state["Apriori_3"] = 0
    if "Regles_associations_3" not in st.session_state:
        st.session_state["Regles_associations_3"] = 0
    if "Predictions_3" not in st.session_state:
        st.session_state["Predictions_3"] = 0

 
    Description = st.sidebar.button("Data description", key="Description3", use_container_width=True , on_click=ui.toggle_other_buttons_3 ,args=["Description_3"])
    Discretisation = st.sidebar.button("Data Discretisation", key="Discretisation", use_container_width=True , on_click=ui.toggle_other_buttons_3 ,args=["Discretisation_3"])
    Apriori = st.sidebar.button("Apriori", key="Apriori", use_container_width=True , on_click=ui.toggle_other_buttons_3 ,args=["Apriori_3"])
    Regles_associations = st.sidebar.button("Règles d'associations", key="Regles_associations", use_container_width=True , on_click=ui.toggle_other_buttons_3 ,args=["Regles_associations_3"])
    Predictions = st.sidebar.button("Predictions", key="Predictions", use_container_width=True , on_click=ui.toggle_other_buttons_3 ,args=["Predictions_3"])
    return_welcome3 =  st.sidebar.button("Return home" ,use_container_width=True )

    if return_welcome3:
        # Main button to return to the welcome page
        st.session_state.page = "welcome"

    if Description or st.session_state["Description_3"] == 1  :
       
        st.subheader("Dataset overwiew")
        st.table(dataset.head(10) )
        st.subheader("Data description  ")
        st.table(d3.describe_data(dataset))
    elif Discretisation  or st.session_state["Discretisation_3"] == 1 :
        st.subheader("Data Discretisation")
        Discretisation_method = st.selectbox("Select Discretisation method", ["","Equal width", "Equal frequency" ])
       
        if Discretisation_method :
            column_to_discretize = ""
            if column_to_discretize == "" :
                st.success("Select a column to discretize")
        column_to_discretize = st.selectbox("Select Column to discretize", ['','Temperature', 'Humidity', 'Rainfall'])


            
        if Discretisation_method == "Equal frequency" and column_to_discretize != "": 
                if "Equal frequency" not in st.session_state:
                    ui.toggle_button("Equal frequency")
                nb_classes = st.slider("Number of classes", 1, 10, 5)
                if "nb_classes" not in st.session_state:
                    st.session_state['nb_classes'] = nb_classes
                st.table(d3.Equal_frequency_discretize(dataset,column_to_discretize, nb_classes))
        elif Discretisation_method == "Equal width" and column_to_discretize != "":
                if "Equal width" not in st.session_state:
                    ui.toggle_button("Equal width")
                st.table(d3.Equal_width_discretize(dataset,column_to_discretize))
        elif column_to_discretize != "": 
                if "Equal width" not in st.session_state:
                    ui.toggle_button("Equal width")
                if "nb_classes" not in st.session_state:
                    st.session_state['nb_classes'] = nb_classes
                st.table(d3.Equal_width_discretize(dataset,column_to_discretize))

    elif Apriori or st.session_state["Apriori_3"] == 1 :
        dataset= d3.Apply_discritization(dataset)
        st.subheader("Apriori")
       
        min_support=st.slider("select min support", 5, 60, step=1)
        min_confiance=st.slider("select min confidence", 0.0, 1.0, 0.5)
        if min_support != "" and min_confiance != "":
            if "min_support" not in st.session_state:
                st.session_state['min_support'] = min_support
            if "min_confiance" not in st.session_state:
                st.session_state['min_confiance'] = min_confiance
            st.table(d3.Apriori(dataset,min_support,min_confiance))
    elif Regles_associations or st.session_state["Regles_associations_3"] == 1 :
        st.subheader("Règles d'associations")
        dataset= d3.Apply_discritization(dataset)

        min_support =st.slider("Select min support" ,5 ,70 ,step=1)
        if min_support != "" :
            if "min_support" not in st.session_state:
                st.session_state['min_support'] = min_support
        st.table(d3.association_rules(dataset ,min_support))
    
    elif Predictions or st.session_state["Predictions_3"] == 1 :
        st.subheader("Predictions")
        dataset= d3.Apply_discritization(dataset)
        dataset["Temperature"] = dataset["Temperature"].astype(float ,errors='ignore')
      
        Temperature = st.selectbox("Temperature" , [""] + list(dataset["Temperature"].unique()))
        Soil = st.selectbox("Soil" , [""] + list(dataset["Soil"].unique()))
        Crop = st.selectbox("Crop" , [""]+ list(dataset["Crop"].unique()))
        Fertilizer = st.selectbox("Fertilizer" , [""]+list(dataset["Fertilizer"].unique()))

        Predict = st.button("Predict" , use_container_width=True)
        if Predict:
            prediction_result =d3.Predict(dataset , Temperature,Soil ,Crop ,Fertilizer)
            Consequent = "Soil" if Soil == "" else "Crop" if Crop == "" else "Fertilizer" if Fertilizer == "" else ""

            if prediction_result == "No rules found":
                st.error(f"For the given inputs there is no  {Consequent} to predict")
            else:
                # select the empty value
                st.success(f"For the given inputs the predicted {Consequent} is : " + format(prediction_result[0]))
# ---------------------------------------------------------------------------------------------------------