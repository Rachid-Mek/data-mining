import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from dateutil.parser import parse


# ---------------------------------------------------------------------------------------------------------
def load_dataset():
    dataset = pd.read_csv('Dataset2.csv' , delimiter=',', quotechar='"')
    return dataset
# ---------------------------------------------------------------------------------------------------------
 
def handle_missing_values(dataset):
    numeric_columns = ['time_period', 'population', 'case count', 'test count', 'positive tests', 'case rate', 'test rate', 'positivity rate']
    dataset.dropna(subset=numeric_columns, inplace=True)
    dataset.reset_index(drop=True, inplace=True)  # Resetting the index and dropping the old index column

    return dataset
# ---------------------------------------------------------------------------------------------------------

def Replace_missing_values(dataset , methode="mean"): # Replace missing values

    numeric_columns = ['time_period', 'population', 'case count', 'test count', 'positive tests', 'case rate', 'test rate', 'positivity rate']
    if methode == "mean":
            dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mean())
    elif methode == "median":
            dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].median())
    elif methode == "interpolate":
            dataset[numeric_columns] = dataset[numeric_columns].interpolate(method='linear', limit_direction='forward')
    elif methode == "drop":
        dataset.dropna(subset=numeric_columns, inplace=True)
        dataset.reset_index(drop=True, inplace=True)  # Resetting the index and dropping the old index column
    return dataset

# ---------------------------------------------------------------------------------------------------------

def describe_data(dataset):
    return dataset.describe()
# ---------------------------------------------------------------------------------------------------------
# Rapport entre les années et les périodes
def Rap_years_periods(dataset):
    df_copy = dataset.copy(deep=True)
    df_copy['Start date'] = pd.to_datetime(df_copy['Start date'], format='%m/%d/%Y', errors='coerce')
    df_copy['Start date'] = df_copy['Start date'].combine_first(pd.to_datetime(df_copy['Start date'], format='%d-%b', errors='coerce'))
    plt.figure(figsize=(10, 6))
    plt.scatter(df_copy['time_period'], df_copy['Start date'].dt.year, color='green', marker='*')
    plt.title('Rapport entre les années et les périodes')
    plt.xlabel('Périodes')
    plt.ylabel('Années')
    return plt



# A function that returns the minimum and maximum periods for each year
def min_max_periods(dataset):
    df_copy = dataset.copy(deep=True)
    min_periods = df_copy.groupby(df_copy['Start date'].dt.year)['time_period'].min()
    max_periods = df_copy.groupby(df_copy['Start date'].dt.year)['time_period'].max()

    df_periods = pd.concat([min_periods, max_periods], axis=1)
    df_periods.columns = ['min_period', 'max_period']
    df_periods = df_periods.reset_index()
    df_periods['Start date'] = pd.to_datetime(df_periods['Start date'], format='%Y')
    return df_periods

# A function that handles the years
def handle_year(dataset):
    df_period = min_max_periods(dataset)
    for index, row in dataset.iterrows():
        if pd.notna(row['Start date']):
            start_date = pd.to_datetime(row['Start date'], errors='coerce')
            if pd.isna(start_date):  # Handle non-standard date formats
                start_date = parse(row['Start date'])
            if start_date.year > 2022:
                matching_period = df_period[
                    (df_period['min_period'] <= row['time_period']) & (df_period['max_period'] >= row['time_period'])]
                if not matching_period.empty:
                    new_year = matching_period.iloc[0]['Start date'].year
                    dataset.at[index, 'Start date'] = start_date.replace(year=new_year).strftime('%Y-%m-%d')
                    if pd.notna(row['end date']):
                        end_date = pd.to_datetime(row['end date'], errors='coerce')
                        if pd.isna(end_date):  # Handle non-standard date formats for 'end date'
                            end_date = parse(row['end date'])
                        dataset.at[index, 'end date'] = end_date.replace(year=new_year).strftime('%Y-%m-%d')

    return dataset	

def standardize_date(attribute): # Standardize the date
    try:
        # Try parsing as a standard date format
        parsed_date = pd.to_datetime(attribute, errors='raise') # errors='raise' to raise an exception if the date is not valid
        return parsed_date.strftime('%Y-%m-%d')# Format the date as YYYY-MM-DD
    except ValueError: # If the date is not a standard format
        try:
            # Try parsing using dateutil.parser
            parsed_date = parse(attribute)
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            return attribute
        
def standardize_dates(dataset,attribute): # Standardize the dates in the dataset
    dataset[attribute] = dataset[attribute].apply(standardize_date)
    return dataset

# ------------------------------------------------------------------------------------------------------------------

def median_outliers(column_name, values, min_threshold, max_threshold):
    """ Replaces outliers with the median of the column """
    numeric_columns = ['time_period', 'population', 'case count', 'test count', 'positive tests', 'case rate', 'test rate', 'positivity rate']
    if column_name in numeric_columns:
        med = values.mean()
        values = values.copy()  # Create a copy to avoid SettingWithCopyWarning
        values[values > max_threshold] = med
        values[values < min_threshold] = med

    return values
def Replace_outliers(dataset , methode="median"): # Replace outliers
    date_columns = ['Start date', 'end date']
    for column in dataset.columns:
        if column not in date_columns:
            # Convert the column to numeric if it's not already
            dataset[column] = pd.to_numeric(dataset[column], errors='coerce')

            Q1 = dataset[column].quantile(0.25)
            Q3 = dataset[column].quantile(0.75)
            IQR = Q3 - Q1
            min_threshold = Q1 - 1.5 * IQR
            max_threshold = Q3 + 1.5 * IQR
            if methode == "mean":
                dataset[column] = dataset[column].fillna(dataset[column].mean())
            elif methode == "median":
                dataset[column] = median_outliers(column, dataset[column], min_threshold, max_threshold)
            elif methode == "interpolate":
                dataset[column] = dataset[column].interpolate(method='linear', limit_direction='forward')
            elif methode == "drop":
                dataset.dropna(subset=[column], inplace=True)
                dataset.reset_index(drop=True, inplace=True)  # Resetting the index and dropping the old index column
    # Affect the date columns with the original values
    for column in date_columns:
        dataset[column] = dataset[column].astype(str)

    return dataset

def handle_year2(dataset, df_period):
    for index, row in dataset.iterrows():
        if pd.notna(row['Start date']):
            start_date = pd.to_datetime(row['Start date'], errors='coerce')
            if pd.isna(start_date):  # Handle non-standard date formats
                start_date = parse(row['Start date'])
            if start_date.year > 2022:
                matching_period = df_period[
                    (df_period['min_period'] <= row['time_period']) & (df_period['max_period'] >= row['time_period'])]
                if not matching_period.empty:
                    new_year = matching_period.iloc[0]['Start date'].year
                    dataset.at[index, 'Start date'] = start_date.replace(year=new_year).strftime('%Y-%m-%d')
                    if pd.notna(row['end date']):
                        end_date = pd.to_datetime(row['end date'], errors='coerce')
                        if pd.isna(end_date):  # Handle non-standard date formats for 'end date'
                            end_date = parse(row['end date'])
                        dataset.at[index, 'end date'] = end_date.replace(year=new_year).strftime('%Y-%m-%d')
    return dataset
# -----------------------------------------------------------------------------------------------------------------------------
def process_dataset():
    df = load_dataset() 
    df = handle_missing_values(df)
    df_copy= df.copy(deep=True)
    df_copy['Start date'] = pd.to_datetime(df_copy['Start date'], format='%m/%d/%Y', errors='coerce')
    df_copy['Start date'] = df_copy['Start date'].combine_first(pd.to_datetime(df_copy['Start date'], format='%d-%b', errors='coerce'))

    min_periods = df_copy.groupby(df_copy['Start date'].dt.year)['time_period'].min()
    max_periods = df_copy.groupby(df_copy['Start date'].dt.year)['time_period'].max()

    df_periods = pd.concat([min_periods, max_periods], axis=1)
    df_periods.columns = ['min_period', 'max_period']
    df_periods = df_periods.reset_index()
    df_periods['Start date'] = pd.to_datetime(df_periods['Start date'], format='%Y')
    df = handle_year2(df, df_periods)

    # Apply the standardize_date function to 'Start date' and 'end date'
    df['Start date'] = df['Start date'].apply(standardize_date)
    df['end date'] = df['end date'].apply(standardize_date)

    # Apply the handle_year function
    df_filtered = df.copy(deep=True)
    df_filtered = Replace_outliers(df_filtered)
 
    return df_filtered 
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ===================================== Visualisation =============================================================
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

def Only_numeric_columns(dataset): # Return only numeric columns
    numeric_columns = ['','time_period', 'population', 'case count', 'test count', 'positive tests', 'case rate', 'test rate', 'positivity rate']
    return numeric_columns
def plot_boxplot(dataset, column): # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=dataset[column])
    plt.title(f'Boxplot for {column}')
    return plt

def plot_all_boxplots(dataset): # Plot all boxplots
    numeric_col = ['time_period', 'population', 'case count', 'test count', 'positive tests', 'case rate', 'test rate', 'positivity rate']

    # Set up subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(numeric_col), figsize=(4 * len(numeric_col), 8))
    fig.subplots_adjust(wspace=0.5)  # Adjust horizontal space between subplots

    #giving general title 
    fig.suptitle('Boxplots before preprocessing', fontsize=16)

    # Plot vertical box plots for each numeric column
    for i, col in enumerate(numeric_col):
        sns.boxplot(y=dataset[col], ax=axes[i])
        axes[i].set_title(f'Boxplot for {col}')

    return plt


# --------------------------------------------------------------------------------------------------------- 
# Nombre total des cas confirmés et tests positifs par zones
def Total_confirm_cases_zone(dataset): # Plot total confirmed cases
    dataset.groupby('zcta').sum()[['case count', 'positive tests']].plot(kind='bar', figsize=(10, 6))
    plt.title('Nombre total des cas confirmés et tests positifs par zones')
    plt.ylabel('Nombre de cas')
    plt.xlabel('Zones')
    return plt
def Zones(dataset): 
    return dataset['zcta'].unique() 
# ---------------------------------------------------------------------------------------------------------
# Évolution des tests COVID-19 pour la zone choisie
def Evolution_tests_zone(dataset , zone=95127): # Plot evolution of tests

    # Filtrer le dataframe pour la zone choisie
    df_zone = dataset[dataset['zcta'] == zone]

    # Convertir les colonnes 'Start date' en datetime
    df_zone['Start date'] = pd.to_datetime(df_zone['Start date'])

    # Trier le dataframe par 'Start date'
    df_zone = df_zone.sort_values(by='Start date')

    # Créer un graphique linéaire pour l'évolution hebdomadaire
    plt.figure(figsize=(12, 6))
    plt.plot(df_zone['Start date'], df_zone['test count'], label='Tests')
    plt.plot(df_zone['Start date'], df_zone['positive tests'], label='Tests positifs') 
    plt.plot(df_zone['Start date'], df_zone['case count'], label='Nombre de cas') 
    plt.title(f'Évolution des tests COVID-19 pour la zone {zone}')
    plt.xlabel('Date')
    plt.ylabel('Nombre de tests') 
    plt.legend()
    return plt

# ---------------------------------------------------------------------------------------------------------

def Repartition_cas_positifs_zone(dataset): # Plot repartition of positive cases
    # print years in the dataset
    dataset['Start date'] = pd.to_datetime(dataset['Start date'], errors='coerce')
    dataset['year'] = dataset['Start date'].dt.year
    

    stacked_data = dataset.groupby(['zcta', 'year'])['positive tests'].sum().unstack()
    ax = stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6))

    plt.title('Répartition des cas positifs de COVID-19 par zone et par année')
    plt.ylabel('Nombre de cas positifs')
    plt.xlabel('Zones')

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x + width / 2, y + height / 2, int(height), horizontalalignment='center', verticalalignment='center')

    return plt
# ---------------------------------------------------------------------------------------------------------

def Rapport_population_tests(dataset): # Plot rapport population tests
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataset, x='population', y='test count', hue='time_period', palette='viridis', size='positive tests')

    # Ajouter une ligne de régression linéaire qui représente la relation entre la population et le nombre de tests effectués
    sns.regplot(data=dataset, x='population', y='test count', scatter=False, color='red') 
    plt.title('Rapport entre la population et le nombre de tests effectués')
    plt.xlabel('Population')
    plt.ylabel('Nombre de tests effectués')
    plt.legend(title='Période')
    return plt
# ---------------------------------------------------------------------------------------------------------
def Impacted_zones(dataset ,Number_of_zones=5): # Plot impacted zones
    zone_impact = dataset.groupby('zcta')['positive tests'].sum()
    zone_impact_sorted = zone_impact.sort_values(ascending=False)
    top_zones = zone_impact_sorted.head(Number_of_zones)


    zone_impact_sorted.head(Number_of_zones).plot(kind='bar', figsize=(10, 6)) # Plot the top impacted zones
    plt.title(f'Top {Number_of_zones} zones les plus touchées par COVID-19') # Add a title
    plt.ylabel('Nombre de cas positifs') # Add y-label
    plt.xlabel('Zones') # Add x-label
    return top_zones ,plt
# ---------------------------------------------------------------------------------------------------------
def Total_confirm_cases(dataset , selected_period=34): # Plot total confirmed cases
    # Filtrer les données pour la période sélectionnée
    period_data = dataset[dataset['time_period'] == selected_period]
    # Exclude datetime columns from sum operation
    numeric_columns = ['case count', 'test count', 'positive tests']
    grouped_data = period_data.groupby('zcta')[numeric_columns].sum()
    # Bar chart for case count, test count, positive tests by zone
    grouped_data.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Nombre total des cas confirmés, tests effectués et tests positifs pour la période {selected_period}')
    plt.ylabel('Nombre de cas')
    plt.xlabel('Zones')
    return plt
# ---------------------------------------------------------------------------------------------------------
def Ratio(dataset , periods=['34', '35', '36']): # Plot ratio) : 
    colors = ['blue', 'green', 'red', 'brown', 'black']
    # Choisir la période que vous souhaitez analyser
    selected_period =periods
    # Exclude datetime columns from sum operation
    numeric_columns = ['case count', 'test count', 'positive tests']

    # Create a figure with a 5x3 grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(20, 17))

    # Iterate over selected periods
    for i in range(0, len(selected_period)):
        period_data = dataset[dataset['time_period'] == int(selected_period[i])]
        grouped_data = period_data.groupby('zcta')[numeric_columns].sum()
        grouped_data['case count'].div(grouped_data['positive tests']).plot(kind='bar', color=colors[i], ax=axs[i, 0])
        axs[i, 0].set_title(f'case count/ positif test by zones pour la période {selected_period[i]}')
        axs[i, 0].set_ylabel('ratio case count/ positif test')
        axs[i, 0].set_xlabel('Zones')

        # grouping the data by zone and dividing the sum of test count by positive tests
        grouped_data['test count'].div(grouped_data['positive tests']).plot(kind='bar', color=colors[i], ax=axs[i, 1])
        axs[i, 1].set_title(f'test count/ positif test by zones pour la période {selected_period[i]}')
        axs[i, 1].set_ylabel('ratio test count/ positif test')
        axs[i, 1].set_xlabel('Zones')

        # grouping the data by zone and dividing the sum of case count by test count
        grouped_data['case count'].div(grouped_data['test count']).plot(kind='bar', color=colors[i], ax=axs[i, 2])
        axs[i, 2].set_title(f'case count/ test count by zones pour la période {selected_period[i]}')
        axs[i, 2].set_ylabel('ratio case count/ test count')
        axs[i, 2].set_xlabel('Zones')

    plt.tight_layout()
    return plt
# ---------------------------------------------------------------------------------------------------------
def  Ratio_filtered(df_filtered) : # 
    colors = ['brown', 'black']
    # Choisir la période que vous souhaitez analyser
    selected_period = ['37', '44']
    # Exclude datetime columns from sum operation
    numeric_columns = ['case count', 'test count', 'positive tests']

    # Create a figure with a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    # Iterate over selected periods
    for i in range(0, len(selected_period)):
        period_data = df_filtered[df_filtered['time_period'] == int(selected_period[i])]
        grouped_data = period_data.groupby('zcta')[numeric_columns].sum()
        grouped_data['case count'].div(grouped_data['positive tests']).plot(kind='bar', color=colors[i], ax=axs[i, 0])
        axs[i, 0].set_title(f'case count/ positif test by zones pour la période {selected_period[i]}')
        axs[i, 0].set_ylabel('ratio case count/ positif test')
        axs[i, 0].set_xlabel('Zones')

        # grouping the data by zone and dividing the sum of test count by positive tests
        grouped_data['test count'].div(grouped_data['positive tests']).plot(kind='bar', color=colors[i], ax=axs[i, 1])
        axs[i, 1].set_title(f'test count/ positif test by zones pour la période {selected_period[i]}')
        axs[i, 1].set_ylabel('ratio test count/ positif test')
        axs[i, 1].set_xlabel('Zones')

        # grouping the data by zone and dividing the sum of case count by test count
        grouped_data['case count'].div(grouped_data['test count']).plot(kind='bar', color=colors[i], ax=axs[i, 2])
        axs[i, 2].set_title(f'case count/ test count by zones pour la période {selected_period[i]}')
        axs[i, 2].set_ylabel('ratio case count/ test count')
        axs[i, 2].set_xlabel('Zones')

    plt.tight_layout()
    return plt
# ---------------------------------------------------------------------------------------------------------