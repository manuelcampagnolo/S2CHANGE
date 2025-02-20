# Load a layer to the project

def add_layer(fn, ln):
    """Inputs:
    fn = path to file
    ln = layername
    """
    # Add the joined layer to the project
    layer = QgsVectorLayer(fn, ln, 'ogr')
    my_project.addMapLayer(layer)


#function to create NVG table with dt_referenc and dt_plant in chronological order
def create_nvg_table(gdf, id_col, col1, col2, desc1, desc2):
    # Create an empty list to store the dictionaries
    result_rows = []

    # Iterate through each unique row in gdf_nvg
    for _, row_nvg in gdf.iterrows():
        result_row = {id_col: row_nvg[id_col]}
        dates_nvg = [row_nvg[col1], row_nvg[col2]]
        activities_nvg = [desc1 if pd.notna(row_nvg[col1]) else np.nan, desc2 if pd.notna(row_nvg[col2]) else np.nan]
        
        dates_nvg = pd.to_datetime(dates_nvg)
        sorted_dates, sorted_activities = zip(*sorted(zip(dates_nvg, activities_nvg)))
        
        for i, (date, activity) in enumerate(zip(sorted_dates, sorted_activities), start=1):
            result_row[f'date_{i}'] = date
            result_row[f'activity_{i}'] = activity
        
        # Append the dictionary to the list
        result_rows.append(result_row)

    # Create the DataFrame from the list of dictionaries
    nvg_df = pd.DataFrame(result_rows)

    # Convert date columns to datetime
    date_columns = [col for col in nvg_df.columns if col.startswith('date_')]
    for col in date_columns:
        nvg_df[col] = pd.to_datetime(nvg_df[col], errors='coerce')
        nvg_df[col] = nvg_df[col].dt.strftime('%d/%m/%Y')
    
    return nvg_df


# CREATE PIVOT TABLE to create pivot_table_exp and pivot_table_silv, sorted chronologically
def create_pivot_table(gdf, value1, value2, id_col):
    #gdf, value1, value2, index = gdf_exp, 'Data Real', 'Atividade', 'id_gleba'
    """
    Inputs:
    - gdf = dataframe to be converted into a pivot table
    - value1 = column to be converted
    - value2 = column to be converted
    - id_col = unique id column name (str)
    Output: pivot table with unique key 
    
    """
    pivot_table = pd.pivot_table(gdf, 
                                 values=[value1, value2], 
                                 index=id_col, 
                                 aggfunc={value1: lambda x: ', '.join(x.astype(str)),
                                          value2: lambda x: ', '.join(x.astype(str))})
    # Reset the index to make 'id_gleba' a column again
    pivot_table.reset_index(inplace=True)
    
    # Split concatenated values into multiple columns
    pivot_table[value1] = pivot_table[value1].apply(lambda x: x.split(','))
    pivot_table[value2] = pivot_table[value2].apply(lambda x: x.split(','))

    # Determine the maximum number of activities and dates
    max_activities = max(pivot_table[value2].apply(len))
    max_dates = max(pivot_table[value1].apply(len))

    # Create new column names
    activity_columns = [f'activity_{i}' for i in range(1, max_activities + 1)]
    date_columns = [f'date_{i}' for i in range(1, max_dates + 1)]

    # Expand the lists into separate columns
    pivot_table[activity_columns] = pd.DataFrame(pivot_table[value2].tolist(), index=pivot_table.index)
    pivot_table[date_columns] = pd.DataFrame(pivot_table[value1].tolist(), index=pivot_table.index)

    # Convert date columns to datetime
    #pivot_table[date_columns] = pivot_table[date_columns].apply(pd.to_datetime, dayfirst=True, errors='coerce')
    #pivot_table[date_columns] = pd.to_datetime(pivot_table[date_columns], format='%d/%m/%Y', errors='coerce')
    # for col in date_columns:
    #         pivot_table[col] = pd.to_datetime(pivot_table[col], format='%Y-%m-%d', errors='coerce')
    # Sort dates and activities in each row by chronological order
    sorted_columns = [id_col] + [col for pair in zip(date_columns, activity_columns) for col in pair]
    pivot_table_sorted = pivot_table[sorted_columns].copy()


    return pivot_table_sorted
    


# Function to merge the 3 dataframes - nvg, exp and silv
def merge_and_transform_dfs(df1, df2, df3, id_col, join_type):
    
    #df1, df2, df3, id_col, join_type = df_nvg, df_exp, df_silv, 'id_gleba', 'inner'
    """
    Inputs:
    - df1 = df_nvg
    - df2 = df_exp
    - df3 = df_silv
    - id_col = unique id column name (str)
    - join_type = (str) 'inner', 'left' or 'right'
    Output: result_df is a table with all three datasets
    """
    
    # Create an empty DataFrame to store the result
    result_df = pd.DataFrame()

    # Loop through unique id_gleba values in df_nvg
    for id_gleba in df1[id_col].unique():
        # Subset data for the current id_gleba from df_nvg
        subset1 = df1[df1[id_col] == id_gleba]

        # Perform the inner join with df_exp if id_gleba exists, else keep df_nvg_subset
        if id_gleba in df2[id_col].unique():
            subset2 = df2[df2[id_col] == id_gleba]
            merged_df = pd.merge(subset1, subset2, on=id_col, how=join_type)
        else:
            merged_df = subset1.copy()

        # Perform the inner join with df_silv if id_gleba exists, else keep merged_df
        if id_gleba in df3[id_col].unique():
            subset3 = df3[df3[id_col] == id_gleba]
            merged_df = pd.merge(merged_df, subset3, on=id_col, how=join_type)

        # Convert columns starting with "date" to datetime
        date_columns = [col for col in merged_df.columns if col.startswith('date')]
        merged_df[date_columns] = merged_df[date_columns].apply(pd.to_datetime, dayfirst=True)

        # Rename date columns to "data_{i}"
        for i, col in enumerate(date_columns, start=1):
            merged_df.rename(columns={col: f'data_{i}'}, inplace=True)

        # Rename activity columns to "actividade_{i}"
        activity_columns = [col for col in merged_df.columns if col.startswith('activity')]
        for i, col in enumerate(activity_columns, start=1):
            merged_df.rename(columns={col: f'actividade_{i}'}, inplace=True)

        # Append the result to the final DataFrame
        result_df = result_df._append(merged_df, ignore_index=True)
        
    return result_df


# function to crceate a list with all activities and a list with all dates
def process_dataframe(df, id_col, col_name1, col_name2):
    #df, id_col, col_name1, col_name2 = df_all, 'id_gleba', 'data', 'actividade'
    """
    Inputs:
    - df = dataframe to be processed
    - id_col = unique id column name (str)
    - col_name1 = column name first word (str) - in this case 'data'
    - col_name2 = column name first word (str) - in this case 'actividade'
    Outputs:
    """
    
    # Empty lists to store dates and activities
    all_datas = []
    all_atividades = []

    # Loop through unique id_gleba values
    for id_gleba in df[id_col].unique():
        subset_df = df[df[id_col] == id_gleba]

        # Extract non-NaN values from date and activity columns
        datas = subset_df.filter(like=col_name1).values.flatten()
        atividades = subset_df.filter(like=col_name2).values.flatten()

        # Remove NaN values
        datas = [str(date) for date in datas if not pd.isna(date)]
        atividades = [str(activity) for activity in atividades if not pd.isna(activity)]

        # Ensure datas and atividades have the same length
        min_length = min(len(datas), len(atividades))
        datas = datas[:min_length]
        atividades = atividades[:min_length]

        # Append values to list
        all_datas.append(datas)
        all_atividades.append(atividades)

    # Create a new DataFrame with id_gleba, datas, and atividades columns
    result_df_list = pd.DataFrame({
        id_col: df_all[id_col].unique(),
        'datas': all_datas,
        'atividades': all_atividades
    })

    return result_df_list


# Define a function to remove '[' or ']' from a cell
def remove_brackets(cell_value):
    if isinstance(cell_value, str): #check if cell is a string 
        return cell_value.replace('[', '').replace(']', '') # if its a string, replace brackets 
    return cell_value # if it is not a string leave it unchanged


def create_final_dataframe(df, id_col):
    
    """
    Inputs:
    - df = dataframe to be used to create the final df
    - id_col = unique id column name (str)
    Output:
    """
    
    # Split the "datas" column and create new date columns
    df[['data{}'.format(i) for i in range(1, df['datas'].apply(len).max() + 1)]] = df['datas'].apply(lambda x: pd.Series(str(x).split(',')))

    # Split the "atividades" column and create new atividade columns
    df[['atividade{}'.format(i) for i in range(1, df['atividades'].apply(len).max() + 1)]] = df['atividades'].apply(lambda x: pd.Series(str(x).split(',')))

    # Drop the original "datas" and "atividades" columns
    df.drop(['datas', 'atividades'], axis=1, inplace=True)

    # Get the column names for datas and atividades
    datas_columns = [col for col in df.columns if col.startswith('data')]
    atividades_columns = [col for col in df.columns if col.startswith('atividade')]

    # Sort the pairs of data and atividade columns by the dates in the data columns
    sorted_columns = sorted(zip(datas_columns, atividades_columns), key=lambda pair: int(pair[0].replace('data', '')))

    # Flatten the sorted pairs to create the reordered_columns list
    reordered_columns = [col for pair in sorted_columns for col in pair]

    # Reorder the DataFrame columns
    final_df = df[[id_col] + reordered_columns]
    
    # Delete the [ or ] from cells
    final_df = final_df.applymap(remove_brackets)

    return final_df

# # Function to sort columns of the final_df
def sort_df(df):
    # Remove extra spaces from date columns
    for col in df.columns:
        if col.startswith('data'):
            df[col] = df[col].str.strip()
            
    # Initialize an empty list to store sorted rows
    sorted_rows = []
    
    # Iterate over rows in the DataFrame
    for _, row in df.iterrows():
        # Initialize dictionary for the current row
        result_row = {'id_gleba': row['id_gleba']}
        
        # Extract date columns and activity columns
        dates_cols = [col for col in df.columns if col.startswith('data')]
        activities_cols = [col for col in df.columns if col.startswith('atividade')]
        
        # Convert date columns to datetime
        dates_df = pd.to_datetime(row[dates_cols], errors='coerce')
        
        # Sort dates and corresponding activities
        sorted_dates, sorted_activities = zip(*sorted(zip(dates_df, row[activities_cols]), key=lambda x: x[0] if not pd.isnull(x[0]) else pd.Timestamp('NAT')))
        
        # Update result_row with sorted dates and activities
        for i, (date, activity) in enumerate(zip(sorted_dates, sorted_activities), start=1):
            result_row[f'data{i}'] = date.strftime('%Y-%m-%d') if not pd.isnull(date) else ''
            result_row[f'atividade{i}'] = activity
        
        # Append the sorted row to the list
        sorted_rows.append(result_row)
        
    sorted_df = pd.DataFrame(sorted_rows)
    return sorted_df



def sort_cols(df, id_col, desc1, desc2):
    # Initialize an empty list to store sorted rows
    sorted_rows = []

    # Iterate over rows in the DataFrame
    for _, row in df.iterrows():
        # Initialize dictionary for the current row
        result_row = {id_col: row[id_col]}
        
        # Extract date columns and activity columns
        dates_cols = [col for col in df.columns if col.startswith('data')]
        activities_cols = [col for col in df.columns if col.startswith('atividade')]
        
        # Convert date columns to datetime
        dates_df = pd.to_datetime(row[dates_cols], dayfirst=True)
        
        # Sort dates and corresponding activities
        sorted_dates, sorted_activities = zip(*sorted(zip(dates_df, row[activities_cols])))
        
        # Update result_row with sorted dates and activities
        for i, (date, activity) in enumerate(zip(sorted_dates, sorted_activities), start=1):
            result_row[f'data{i}'] = date
            result_row[f'atividade{i}'] = activity
        
        # Append the sorted row to the list
        sorted_rows.append(result_row)
    
    # Return the list of sorted rows after iterating over all rows
    return sorted_rows


# Function to clean all the white spaces and special characters
def clean_atividade_columns(df):
    # Select columns that start with 'atividade'
    atividade_columns = [col for col in df.columns if col.startswith('atividade')]
    
    # Remove all white spaces and special characters from atividade columns
    for col in atividade_columns:
        df[col] = df[col].str.replace(r'\s+', '', regex=True)  # Remove white spaces
        df[col] = df[col].str.replace(r'[^a-zA-Z0-9]', '', regex=True)  # Remove special characters
    
    return df

##############
# extract a talhao from nvg shapefile
def extract_talhao_from_nvg(input_shp, id_gleba):
    # extract talhao from geopackage by id_gleba
    talhao = processing.run("native:extractbyexpression", 
     {'INPUT':input_shp,
     'EXPRESSION':'"id_gleba" = \'' + id_gleba + '\'',
     'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
     
    return talhao

#convert the talhao from multipart to singlepart
def multi_to_singlepart(talhao):
    # convert from multipart to singlepart
    talhao_singlepart = processing.run("native:multiparttosingleparts", 
    {'INPUT': talhao,
    'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    
    return talhao_singlepart



# add a primary key to the each subparcel of a talhao
def add_primary_key_talhao(input_shp):
    try:
        # Add a field for the primary key
        talhao_singlepart_pk = processing.run("native:addfieldtoattributestable", 
                                              {'INPUT': input_shp,
                                               'FIELD_NAME': 'id',
                                               'FIELD_TYPE': 2,
                                               'FIELD_LENGTH': 255,
                                               'FIELD_PRECISION': 0,
                                               'FIELD_ALIAS': '',
                                               'FIELD_COMMENT': '',
                                               'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

        # Create a dictionary to store the counts for each id_gleba
        id_gleba_counts = {}

        # Get the index of the newly added 'id' field
        idx = talhao_singlepart_pk.fields().indexFromName('id')

        # Update features with sequential IDs within each id_gleba
        with edit(talhao_singlepart_pk):
            for feature in talhao_singlepart_pk.getFeatures():
                id_gleba = feature['id_gleba']
                if id_gleba not in id_gleba_counts:
                    id_gleba_counts[id_gleba] = 1
                feature.setAttribute(idx, f"{id_gleba}_{id_gleba_counts[id_gleba]:02d}")
                id_gleba_counts[id_gleba] += 1
                talhao_singlepart_pk.updateFeature(feature)

        # Refresh the layer to reflect changes
        talhao_singlepart_pk.triggerRepaint()
        # select all features of layer to export
        ln_talhao_singlepart = 'nvg_singlepart_' + id_gleba + '.shp'
        # ln_nvg_singlepart = 'nvg_singlepart.shp'
        fn_talhao_singlepart = str(my_folder/output_folder/ln_talhao_singlepart)
        talhao_singlepart_pk.selectAll()
        # export layer
        talhao_singlepart_shp = processing.run("native:saveselectedfeatures", {'INPUT':talhao_singlepart_pk, 'OUTPUT':fn_talhao_singlepart})
        talhao_singlepart_pk.removeSelection()

        return talhao_singlepart_pk, talhao_singlepart_shp, fn_talhao_singlepart

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# extract a list of unique id_glebas from nvg shapefile
def extract_unique_id_gleba_from_nvg(input_shp, unique_value):
    # Extract unique id_gleba values from the geopackage
    unique_id_gleba = set()
    with fiona.open(input_shp, 'r') as src:
        for feature in src:
            unique_id_gleba.add(feature['properties'][unique_value])
    return list(unique_id_gleba)

def create_id_gleba_dates(fn_nvg, df_sorted):
    id_gleba_list = extract_unique_id_gleba_from_nvg(fn_nvg)
    id_gleba_dates = {}

    for id_gleba in id_gleba_list:
        date_pairs = filter_and_select_dates1(df_sorted, id_gleba)
        
        # Check if date_pairs is not empty
        if date_pairs:
            id_gleba_dates[id_gleba] = date_pairs
        else:
            pass #print(f"No 'CORTE' activity found for ID {id_gleba}")
    
    return id_gleba_dates



def filter_and_select_dates1(df, id_gleba):
    # Filter rows based on matching id_gleba
    matching_rows = df[df['id_gleba'] == id_gleba]

    # Check if matching_rows is empty
    if matching_rows.empty:
        print(f"No data found for ID {id_gleba}")
        return []  # Return an empty list if no data is found

    # Get date and activity columns
    date_columns = [col for col in matching_rows.columns if col.startswith('data')]
    activity_columns = [col for col in matching_rows.columns if col.startswith('atividade')]

    # Keep only non-null date values
    matching_rows = matching_rows.dropna(subset=date_columns + activity_columns, how='all')

    # Select columns where activity starts with 'CORTE' and their respective dates
    corte_activity_columns = [col for col in activity_columns if matching_rows[col].str.startswith('CORTE').any()]
    corte_date_columns = [col.replace('atividade', 'data') for col in corte_activity_columns]

    # Check if corte_date_columns is empty
    if not corte_date_columns:
        print(f"No 'CORTE' activity found for ID {id_gleba}")
        return []  # Return an empty list if no 'CORTE' activity is found

    # Select columns
    selected_columns = corte_activity_columns + corte_date_columns
    selected_data = matching_rows[selected_columns]

    # Filter out invalid dates (e.g., 'nan')
    selected_data = selected_data.replace('nan', np.nan)

    # Filter out rows with no valid date values
    selected_data = selected_data.dropna(subset=corte_date_columns, how='all')

    # Flatten the selected_data into a list of dates
    clear_cut_dates = sorted(selected_data[corte_date_columns].stack().dropna())

    if not clear_cut_dates:
        print(f"No valid dates found for 'CORTE' activity for ID {id_gleba}")
        return []  # Return an empty list if no valid dates are found

    # Convert dates from string to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in clear_cut_dates]

    # Initialize list to store pairs of dates with one-year span
    date_pairs = []

    # Initialize start date of the current year
    start_date_current_year = dates[0]

    # Iterate through the dates
    for i in range(len(dates)):
        # Find the end date that falls within one year from the start date of the current year
        if (dates[i] - start_date_current_year).days > (365*2):
            # Append the pair of start and end dates
            date_pairs.append((start_date_current_year.strftime('%Y-%m-%d'), dates[i - 1].strftime('%Y-%m-%d')))
            # Update start date of the current year
            start_date_current_year = dates[i]

    # Append the last pair of start and end dates
    date_pairs.append((start_date_current_year.strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')))

    return date_pairs, clear_cut_dates, corte_activity_columns, corte_date_columns



def find_date_pairs(df, id_gleba):
    # Filter rows based on matching id_gleba
    matching_rows = df[df['id_gleba'] == id_gleba]
    
    # Get date and activity columns
    date_columns = [col for col in matching_rows.columns if col.startswith('data')]
    activity_columns = [col for col in matching_rows.columns if col.startswith('atividade')]
  
    # Keep only non-null date values
    matching_rows = matching_rows.dropna(subset=date_columns + activity_columns, how='all')
    
    # Select columns where activity starts with 'CORTE' and their respective dates
    corte_activity_columns = [col for col in activity_columns if matching_rows[col].str.startswith('CORTE').any()]
    corte_date_columns = [col.replace('atividade', 'data') for col in corte_activity_columns]
    
    # Select columns
    selected_columns = corte_activity_columns + corte_date_columns
    selected_data = matching_rows[selected_columns]
    
    # Filter out invalid dates (e.g., 'nan')
    selected_data = selected_data.replace('nan', np.nan)
    
    # Filter out rows with no valid date values
    selected_data = selected_data.dropna(subset=corte_date_columns, how='all')
    
    # Flatten the selected_data into a list of dates
    dates = sorted(selected_data[corte_date_columns].stack().dropna())
    
    if not dates:
        return []
    
    # Convert dates from string to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    # Initialize list to store pairs of dates with one-year span
    date_pairs = []

    # Initialize the start date of the current pair
    start_date = dates[0]

    # Iterate through the dates starting from the second date
    for i in range(1, len(dates)):
        # Calculate the difference in days between consecutive dates
        time_diff_days = (dates[i] - dates[i - 1]).days

        # If the difference is greater than 730 days, it marks the end of the current pair and start of a new pair
        if time_diff_days > 730:
            # Append the pair of start and end dates
            date_pairs.append((start_date.strftime('%Y-%m-%d'), dates[i - 1].strftime('%Y-%m-%d')))
            # Update the start date for the new pair
            start_date = dates[i]

    # Handle cases where there's only one date or two identical dates
    if len(dates) == 1 or (len(dates) == 2 and dates[0] == dates[1]):
        date_pairs = [(dates[0].strftime('%Y-%m-%d'), dates[0].strftime('%Y-%m-%d'))]
    else:
        # Append the last pair of start and end dates if there are more than one unique dates
        if len(set(dates)) > 1:
            date_pairs.append((start_date.strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')))
    
    return date_pairs


def get_start_end_dates(modified_date_pairs):
    # Initialize lists to store start and end dates
    start_dates = []
    end_dates = []
    # Iterate over each date pair
    for start_date, end_date in modified_date_pairs:
        # Append start and end dates to respective lists
        start_dates.append(start_date)
        end_dates.append(end_date)
    
    # Return the lists of start and end dates
    return start_dates, end_dates





# # Function to add or subtract months from a date to pairs of dates

def add_subtract_days(date_str, days):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    new_date = date + timedelta(days=days)
    return new_date.strftime('%Y-%m-%d')


def dates_with_two_months_diff(date_pairs):
    # List to store modified date pairs
    modified_date_pairs = []
    new_start_dates = []
    new_end_dates = []
    
    # # Print original date pairs
    # print("Original Date Pairs:", date_pairs)
    
    # Loop through the date pairs, modify the dates and append to modified_date_pairs
    for start_date, end_date in date_pairs:
        new_start_date = add_subtract_days(start_date, -60)
        new_end_date = add_subtract_days(end_date,60)
        # Print modified dates
        #print("Original Start Date:", start_date)
        #print("Original End Date:", end_date)
        #print("Modified Start Date:", new_start_date)
        #print("Modified End Date:", new_end_date)
        # Append modified dates to lists
        modified_date_pairs.append((str(new_start_date), str(new_end_date)))
        new_start_dates.append(str(new_start_date))
        new_end_dates.append(str(new_end_date))
    
    # # Print modified date pairs
    # print("Modified Date Pairs for id_gleba:", id_gleba)
    # print(modified_date_pairs)
    
    return new_start_dates, new_end_dates, modified_date_pairs


def start_and_end_dates_two_months (first_start_date, first_end_date):
    # Convert start_date and end_date strings to datetime objects
    first_start_date = datetime.strptime(first_start_date, '%Y-%m-%d')  # Adjust format if needed
    first_end_date = datetime.strptime(first_end_date, '%Y-%m-%d')  # Adjust format if needed

    # Subtract 2 months from start_date
    start_date_minus_2_months = first_start_date - relativedelta(months=2)

    # Add 2 months to end_date
    end_date_plus_2_months = first_end_date + relativedelta(months=18)

    # Convert datetime objects to desired format if needed
    start_date = start_date_minus_2_months.strftime('%Y-%m-%d')
    end_date = end_date_plus_2_months.strftime('%Y-%m-%d')
    
    return start_date, end_date

def join_pivot_tables(fn_folder):
    all_files = os.listdir(fn_folder)
    csv_files = [f for f in all_files if f.endswith('_count.csv')]
    df_list = []
    ## to delete files, if needed
    # files_to_delete = glob.glob(os.path.join(fn_folder, '*count.csv'))
    # # Iterate over the list of files and delete them
    # for file_path in files_to_delete:
    #     try:
    #         os.remove(file_path)
    #         print(f"Deleted: {file_path}")
    #     except Exception as e:
    #         print(f"Error deleting {file_path}: {e}")

    # create new files with columns with count number of all S2 images, empty and non empty NDVI values
    for csv_file in csv_files:
        # Construct the full file path
        file_path = os.path.join(fn_folder, csv_file)
        df = pd.read_csv(file_path)

        date_columns = [col for col in df.columns if col.startswith('date_20')]
        
        # count the number of empty and non-empty cells for each row
        df['nr_empty_cells'] = df[date_columns].isna().sum(axis=1)
        df['nr_non_empty_cells'] = df[date_columns].notna().sum(axis=1)
        df['nr_s2_dates'] = df['nr_empty_cells'] + df['nr_non_empty_cells']
        
        new_file_name = os.path.splitext(csv_file)[0] + '_count.csv'
        new_file_path = os.path.join(fn_folder, new_file_name)
        # save
        df.to_csv(new_file_path, index=False)


    for csv in csv_files:
        file_path = os.path.join(fn_folder, csv)
        df = pd.read_csv(file_path)
        df_list.append(df[['id','id_gleba','biggest_drop_NDVI','date_of_biggest_drop', 'estimated_date','nr_empty_cells', 'nr_non_empty_cells','nr_s2_dates']])

    all_pivot_tables = pd.concat(df_list, ignore_index=True)
    date_columns = [col for col in all_pivot_tables.columns if col.startswith('date_20')]
    all_pivot_tables = all_pivot_tables.drop(columns=date_columns)
    #save
    ln_all_pivot_tables = 'all_pivot_tables.csv'
    fn_all_pivot_tables = str(my_folder / output_folder / ln_all_pivot_tables)
    all_pivot_tables.to_csv(fn_all_pivot_tables)

    df1 = pd.read_csv(fn_all_pivot_tables)
    df2 = pd.read_csv(fn_csv_dates)

    #join attributes (dates, areas and nr of clear cuts) to the joined pivot tables from 
    #result = join_attribute_to_layer(fn_all_pivot_tables, 'id_gleba', fn_csv_dates, 'id_gleba', ['area_ha','start_date','end_date','date_difference_days','nr_clear_cuts'])
    result = pd.merge(df1, df2, on='id_gleba', how='inner')

    # save layer
    ln_nvg_pt = 'all_nvg_pivot_tables.csv'
    fn_nvg_pt = str(my_folder / output_folder / ln_nvg_pt)
    result.to_csv(fn_nvg_pt)
    
    return fn_nvg_pt

def mask_s2_clouds(image):
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start", "system:time_end", "system:id", "system:version", "system:asset_size", "system:footprint", "system:index"])

  
  
  
# Add NDVI band
def addNDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Calculate median NDVI for each image and each polygon
def calculateMedianNDVI(image):
    medianNDVI = image.select('NDVI').reduceRegions(
        collection=nvg,
        reducer=ee.Reducer.median().unweighted(),
        scale=10
    )
    # return medianNDVI
    return medianNDVI.map(lambda feature: ee.Feature(feature).set('date', image.date().format('YYYY-MM-dd')))




def ndvi_median_gee(start_date, end_date, nvg, cloud_percentage):
    # Filter by Geo and Growing days
    S2_SR = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filterBounds(nvg) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_percentage)

    # Apply across the whole collection 
    S2_NDVI = S2_SR.map(addNDVI)
    medianNDVI = S2_NDVI.map(calculateMedianNDVI).flatten()
    
    return medianNDVI

def ndvi_median_gee_masks2clouds(start_date, end_date, nvg, cloud_percentage):
    # Filter by Geo and Growing days
    S2_SR = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filterBounds(nvg) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_percentage)\
            .map(mask_s2_clouds)

    # Apply across the whole collection 
    S2_NDVI = S2_SR.map(addNDVI)
    medianNDVI = S2_NDVI.map(calculateMedianNDVI).flatten()
    
    return medianNDVI

def map_features(feature):
    properties = feature.toDictionary(['id', 'date', 'id_gleba', 'median'])
    return ee.Feature(feature.geometry(), properties)


def convert_to_pivot_table(df_median_ndvi):
    # Convert the date column to string format with 'yyyymmdd' format
    df_median_ndvi['date'] = pd.to_datetime(df_median_ndvi['date']).dt.strftime('%Y%m%d')
    # Pivot the DataFrame
    pivot_table = df_median_ndvi.pivot_table(index=['id', 'id_gleba'], columns='date', values='median', aggfunc='first')
    # Reset index to make 'id' a column again
    pivot_table.reset_index(inplace=True)
    
    return pivot_table




def extract_clear_cut_dates_and_find_closest_date(df_sorted, id_gleba, pivot_table):
    # Filter rows based on matching id_gleba
    matching_rows = df_sorted[df_sorted['id_gleba'] == id_gleba]
    # Get date and activity columns
    date_columns = [col for col in matching_rows.columns if col.startswith('data')]
    activity_columns = [col for col in matching_rows.columns if col.startswith('atividade')]
    # Keep only non-null date values
    matching_rows = matching_rows.dropna(subset=date_columns + activity_columns, how='all')
    # Select columns where activity starts with 'CORTE' and their respective dates
    corte_activity_columns = [col for col in activity_columns if matching_rows[col].str.startswith('CORTE').any()]
    corte_date_columns = [col.replace('atividade', 'data') for col in corte_activity_columns]
    # Select columns
    selected_columns = corte_activity_columns + corte_date_columns
    selected_data = matching_rows[selected_columns]
    # Filter out invalid dates (e.g., 'nan')
    selected_data = selected_data.replace('nan', np.nan)
    # Filter out rows with no valid date values
    selected_data = selected_data.dropna(subset=corte_date_columns, how='all')
    # Flatten the selected_data into a list of dates
    clear_cut_dates = sorted(selected_data[corte_date_columns].stack().dropna())
    # Convert dates from string to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in clear_cut_dates]

    # Find closest date
    def find_closest_date(row):
        id_gleba = row['id_gleba']
        date_of_biggest_drop = row['date_of_biggest_drop']
        closest_date = None
        min_time_diff = float('inf')
        
        if isinstance(date_of_biggest_drop, str):
            date_of_biggest_drop = datetime.strptime(date_of_biggest_drop, '%Y-%m-%d')

        # Iterate through corte_date_columns to find the closest date before 'date_of_biggest_drop'
        for col in corte_date_columns:
            for _, sorted_row in df_sorted[df_sorted['id_gleba'] == id_gleba].iterrows():
                if pd.notna(sorted_row[col]):  # Skip NaN values
                    sorted_date = datetime.strptime(str(sorted_row[col]), '%Y-%m-%d')
                    if sorted_date < date_of_biggest_drop:  # Consider only dates before 'date_of_biggest_drop'
                        time_diff = abs((date_of_biggest_drop - sorted_date).total_seconds())
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            #closest_date = sorted_row[col]
                            closest_date = sorted_date.strftime('%Y-%m-%d')  

        return closest_date

    # Apply the function to create the 'estimated_date' column in df_pivot_table
    pivot_table['estimated_date'] = pivot_table.apply(find_closest_date, axis=1)

    return corte_activity_columns, corte_date_columns, clear_cut_dates, pivot_table

def calculate_biggest_ndvi_drop_and_estimated_date(df_sorted, id_gleba, pivot_table):
    #BIGGEST DROP AND DATE
    # Filter columns that start with '20'
    ndvi_columns = [col for col in pivot_table.columns if col.startswith('20')]
    # Calculate the differences between consecutive columns
    ndvi_diff = pivot_table[ndvi_columns].diff(axis=1)
    # Find the biggest drop per row
    biggest_drop_per_row = ndvi_diff.min(axis=1)
    # Add a column named 'biggest_drop_NDVI' to the DataFrame
    pivot_table['biggest_drop_NDVI'] = biggest_drop_per_row
    # Find the column name where the biggest drop occurred per row
    date_of_biggest_drop = ndvi_diff.idxmin(axis=1)
    # Add a column named 'date_of_biggest_drop' to the DataFrame
    pivot_table['date_of_biggest_drop'] = date_of_biggest_drop
    # Convert 'date_of_biggest_drop' column to string format
    pivot_table['date_of_biggest_drop'] = pivot_table['date_of_biggest_drop'].astype(str)
    # Convert 'date_of_biggest_drop' column to datetime format
    pivot_table['date_of_biggest_drop'] = pd.to_datetime(pivot_table['date_of_biggest_drop'], format='%Y%m%d', errors='coerce')
    
    # Call the combined function to extract clear cut dates and find closest date
    corte_activity_columns, corte_date_columns, clear_cut_dates, pivot_table = extract_clear_cut_dates_and_find_closest_date(df_sorted, id_gleba, pivot_table)
    
    # Rename columns
    pivot_table.rename(columns={col: 'date_' + col for col in pivot_table.columns if col.startswith('20')}, inplace=True)
    
    return pivot_table


# def calculate_biggest_ndvi_drop_and_estimated_date(df_sorted, id_gleba, pivot_table, date_pairs):
#     # Utility function to check if a date is within any of the date ranges in date_pairs
#     def is_within_date_pairs(date_str, date_pairs):
#         # Convert the date string from 'YYYYMMDD' to datetime object
#         date = datetime.strptime(date_str, '%Y%m%d')
#         for start_date, end_date in date_pairs:
#             if datetime.strptime(start_date, '%Y-%m-%d') <= date <= datetime.strptime(end_date, '%Y-%m-%d'):
#                 return True
#         return False

#     # Filter columns that start with '20' and are within the date_pairs range
#     ndvi_columns = [col for col in pivot_table.columns if col.startswith('20')]
#     ndvi_columns = [col for col in ndvi_columns if is_within_date_pairs(col, date_pairs)]

#     if not ndvi_columns:
#         raise ValueError("No NDVI columns found within the specified date ranges.")

#     # Calculate the differences between consecutive columns
#     ndvi_diff = pivot_table[ndvi_columns].diff(axis=1)
#     # Find the biggest drop per row
#     biggest_drop_per_row = ndvi_diff.min(axis=1)
#     # Add a column named 'biggest_drop_NDVI' to the DataFrame
#     pivot_table['biggest_drop_NDVI'] = biggest_drop_per_row
#     # Find the column name where the biggest drop occurred per row
#     date_of_biggest_drop = ndvi_diff.idxmin(axis=1)
#     # Add a column named 'date_of_biggest_drop' to the DataFrame
#     pivot_table['date_of_biggest_drop'] = date_of_biggest_drop
#     # Convert 'date_of_biggest_drop' column to string format
#     pivot_table['date_of_biggest_drop'] = pivot_table['date_of_biggest_drop'].astype(str)
#     # Convert 'date_of_biggest_drop' column to datetime format
#     pivot_table['date_of_biggest_drop'] = pd.to_datetime(pivot_table['date_of_biggest_drop'], format='%Y%m%d', errors='coerce')

#     # Call the combined function to extract clear cut dates and find closest date
#     corte_activity_columns, corte_date_columns, clear_cut_dates, pivot_table = extract_clear_cut_dates_and_find_closest_date(df_sorted, id_gleba, pivot_table)
    
#     # Rename columns
#     pivot_table.rename(columns={col: 'date_' + col for col in pivot_table.columns if col.startswith('20')}, inplace=True)
    
#     return pivot_table





def join_attribute_to_layer(input1, field1, input2, field2, attribute_to_copy):
    result = processing.run("native:joinattributestable", {
        'INPUT': input1, 
        'FIELD': field1,
        'INPUT_2': input2,
        'FIELD_2': field2,
        'FIELDS_TO_COPY': attribute_to_copy,
        'METHOD': 1,
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    return result



def create_expanded_df(df1, df2, col1, col2, col_to_join, join_type):
    # Merge the two dataframes
    expanded_df = pd.merge(df1, df2[[col1, col2]], on=col_to_join, how=join_type)
    
    # Get the index of 'col_to_join' column
    col_to_join_index = expanded_df.columns.get_loc(col_to_join)
    
    # Insert the 'col2' column next to 'col_to_join' column
    expanded_df.insert(col_to_join_index + 1, col2, expanded_df.pop(col2))
    
    # Iterate over the columns containing 'data' in their names
    for col in expanded_df.columns:
        if 'data' in col:
            # Extracting the index of the data column
            index = int(col.replace('data', ''))
            
            # Creating a new column with name 'data_estimada{i}'
            new_col_name = f'data_estimada{index}'
            
            # Inserting the new empty column next to the corresponding 'data' column
            expanded_df.insert(expanded_df.columns.get_loc(col) + 1, new_col_name, None)
    return expanded_df






# Function to convert shapefile to Earth Engine geometry
def get_ee_geometry_from_shapefile(shapefile_path):
    try:
        ee_geometry = geemap.shp_to_ee(shapefile_path, encoding='latin-1')
        return ee_geometry.geometry()
    except Exception as e:
        print(f"Error converting shapefile to EE geometry: {e}")
        raise



# Function to apply the cloud mask
def apply_cloud_mask(img):
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))

# Function to apply cloud mask and compute median NDVI
def ndvi_cloud_score(start_date, end_date, geometry):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date)
    
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date)
    
    QA_BAND = 'cs_cdf'
    CLEAR_THRESHOLD = 0.60

    def apply_cloud_mask(img):
        return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))
    
    composite = s2.map(lambda img: img.addBands(
                        csPlus.filterBounds(img.geometry()) \
                              .filterDate(img.date(), img.date()) \
                              .first().select([QA_BAND]))) \
                  .map(apply_cloud_mask)
        
    # Apply across the whole collection 
    S2_NDVI = composite.map(addNDVI)
    medianNDVI = S2_NDVI.map(calculateMedianNDVI).flatten()
    
    return medianNDVI


# count non-null values in activity columns that start with 'CORTE'
def count_corte_activities(row):
    activity_columns = [col for col in row.index if col.startswith('atividade')]
    corte_activities = [col for col in activity_columns if pd.notnull(row[col]) and isinstance(row[col], str) and row[col].startswith('CORTE')]
    return len(corte_activities)


# Function to update the 'first_estimated_date' if the conditions are met
def update_first_estimated_date(row):
    if pd.notna(row['date_of_biggest_drop']) and row['first_start_date'] == row['first_end_date'] and pd.isna(row['first_estimated_date']):
        row['first_estimated_date'] = row['first_start_date']
    return row

# Create a function to classify mean_month into groups
def classify_month(month):
    if month in group_1:
        return 'group_1'
    elif month in group_2:
        return 'group_2'
    elif month in group_3:
        return 'group_3'
    else:
        return 'other'



def extract_by_location(input1, input2):
    result = processing.run("native:extractbylocation", 
     {'INPUT':input1,
     'PREDICATE':[6],
     'INTERSECT':input2,
     'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    return result


def extract_by_location_permanent(input1, input2, fn_to_save):
    result = processing.run("native:extractbylocation", 
    {'INPUT':input1,
    'PREDICATE':[0,6],
    'INTERSECT':input2,
    'OUTPUT':fn_to_save})
    
    return result


# Function to convert 'ddmmyyyy' to datetime
def convert_to_date(date_str):
    return datetime.strptime(date_str, '%d%m%Y')

# Function to filter dates within the interval
def filter_dates(dates, start_date, end_date):
    dates = [convert_to_date(date) for date in dates]
    filtered_dates = [date for date in dates if start_date <= date <= end_date]
    return filtered_dates[0] if filtered_dates else None




def find_closest_date_before(drop_date, date_list):
    """
    Find the closest date in date_list that is before drop_date.
    If no such date exists, return None.
    """
    try:
        drop_date = datetime.strptime(drop_date, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None
    
    past_dates = [date for date in date_list if date < drop_date]
    if past_dates:
        return max(past_dates).strftime('%Y-%m-%d')
    else:
        return None


# def extract_id_glebas(file_name):
#     # Extract base IDs from file_name and add '_EG'
#     base_ids = re.findall(r'\d{5}-T\d{3}', file_name)
#     id_glebas = [f"{base_id}_EG" for base_id in base_ids]
#     return id_glebas

def extract_id_glebas(filename):
    # This is a placeholder function; replace with actual implementation
    base_name = Path(filename).stem  # Use Path to get the stem (filename without extension)
    id_glebas = base_name.split('_')
    return [id_gleba + '_EG' for id_gleba in id_glebas]
    
    
def join_field_by_location (input1, input2, list_fields, output):
    att_by_loc =  processing.run("native:joinattributesbylocation", 
    {'INPUT':input1,
    'PREDICATE':[0,5],
    'JOIN': input2,
    'JOIN_FIELDS':list_fields,
    'METHOD':0,
    'DISCARD_NONMATCHING':False,
    'PREFIX':'',
    'OUTPUT': output})
    return att_by_loc

# Function to add a new row to the DataFrame
def add_row_to_df(df, id_gleba, status, limitations):
    new_row = pd.DataFrame({'id_gleba': [id_gleba], 'status': [status], 'limitations': [limitations]})
    df = pd.concat([df, new_row], ignore_index=True)
    return df


# # Function to convert ddmmyyyy to datetime object
# def parse_ddmmyyyy(date_str):
#     return datetime.strptime(date_str, '%d%m%Y')

def parse_ddmmyyyy(date_string):
    try:
        # Clean the input
        date_string = date_string.replace(',', '').strip()
        # Attempt to parse the date
        return datetime.strptime(date_string, "%d%m%Y")
    except ValueError as e:
        print(f"Error parsing date: {date_string} - {e}")
        return None  # Or handle as appropriate

# Function to filter dates between start_date and end_date
def filter_dates(tbreak_ddm_list, start_date, end_date):
    if not tbreak_ddm_list:  # If the list is empty
        return None
    
    # Convert the dates in the list to datetime format
    tbreak_dates = [parse_ddmmyyyy(date) for date in tbreak_ddm_list]
    
    # Filter dates within the interval
    valid_dates = [date for date in tbreak_dates if start_date <= date <= end_date]
    
    # If no valid dates, return None
    if not valid_dates:
        return None
    
    # Return the earliest valid date
    return min(valid_dates)



# Function to convert milliseconds to a date string
def ms_to_date_str(ms):
    return datetime.utcfromtimestamp(ms / 1000).strftime('%Y%m%d')

def rename_tiff_s2_images_to_dates(folder_path):
    # Get a list of all .tif files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            # Extract the milliseconds part from the filename
            # Assuming it's always in the format 'S2SR_image_<ms>_tile_....tif'
            try:
                ms_str = filename.split('_')[2]  # Get the milliseconds part
                milliseconds = int(ms_str)  # Convert to integer

                # Convert the milliseconds to a date string
                new_date_str = ms_to_date_str(milliseconds)

                # Create the new filename using the date string
                new_filename = f"S2SR_image_{new_date_str}_tile_{filename.split('_')[-1]}"

                # Construct full paths for the old and new file
                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")


def create_drop_date(df):
    
    # Ensure the tBreak_ddm column contains lists of strings
    df['tBreak_ddm'] = df['tBreak_ddm'].apply(eval)

    # Convert start_date and end_date to datetime
    df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d')

    # Apply the function to create the 'drop_date' column
    df['drop_date'] = df.apply(
        lambda row: filter_dates(row['tBreak_ddm'], row['start_date'], row['end_date']),
        axis=1
    )

    # Convert datetime fields back to string because esri shapefile formats do not support datetime fields
    df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
    df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d')
    df['drop_date'] = df['drop_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)

    # Convert tBreak_ddm lists to strings
    df['tBreak_ddm'] = df['tBreak_ddm'].apply(lambda x: ','.join(x))

    return df


# Function to wrap values in tBreak_ddm appropriately
def format_tBreak_ddm(x):
    if pd.notna(x):  # If the value is not null
        if isinstance(x, str) and ',' in x:  # Check if there are multiple values
            # Split by commas, strip whitespace, and join with commas in the required format
            values = [f"'{val.strip()}'" for val in x.split(',')]
            return f"[{','.join(values)}]"  # Join values as a list
        else:
            return f"[\'{x}\']"  # Single value wrapped in quotes
    else:
        return "[]"  # If NaN





def set_layer_legend(layer, fieldname, rampname):
    # set labels according to estimated clear-cut dates 
    layer = iface.activeLayer()
    #field_name = 'estimated_'
    field_name = fieldname
    field_index = layer.fields().indexFromName(field_name)
    unique_values = sorted(layer.uniqueValues(field_index))

    category_list = []
    for value in unique_values:
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        category = QgsRendererCategory(value, symbol, str(value))
        category_list.append(category)

    # color ramp
    ramp_name = rampname
    default_style = QgsStyle().defaultStyle()
    color_ramp = default_style.colorRamp(ramp_name)
    renderer = QgsCategorizedSymbolRenderer(field_name, category_list)
    renderer.updateColorRamp(color_ramp)
    layer.setRenderer(renderer)
    layer.triggerRepaint()
    
    return layer


def convert_drop_date_to_datetime(layer, drop_date_field, new_field_name):
    """
    Converts the specified drop_date field in the given layer to datetime format.
    
    Parameters:
    layer (QgsVectorLayer): The layer containing the drop_date field.
    drop_date_field (str): The name of the field to convert.
    new_field_name (str): The name of the new field to store datetime values.
    """
    # Check if the new field already exists
    if new_field_name not in layer.fields().names():
        # Start editing the layer
        layer.startEditing()
        
        # Add a new field for the datetime values
        layer.addAttribute(QgsField(new_field_name, QVariant.DateTime))
        
        # Loop through the features and convert the drop_date values
        for feature in layer.getFeatures():
            drop_date_value = feature[drop_date_field]  # Get the drop_date value
            if drop_date_value:  # Check if it's not null
                # Convert the drop_date string to QDateTime
                datetime_value = QDateTime.fromString(drop_date_value, 'yyyy-MM-dd')  # Adjust format as needed
                
                # Update the new field with the converted value
                feature[new_field_name] = datetime_value
                layer.updateFeature(feature)  # Update the feature with new value

        # Commit changes to the layer
        layer.commitChanges()


def convert_dates_to_yyyymmdd(result):
    """
    Extracts start and end dates from the result layer and converts them to YYYYMMDD format.
    
    Args:
        result: The QgsVectorLayer from which to extract dates.
    
    Returns:
        tuple: A tuple containing the formatted start and end dates in YYYYMMDD format.
    """
    # Initialize variables to store the dates
    start_date_yyyymmdd = None  # Initialize in case of None value
    end_date_yyyymmdd = None     # Initialize in case of None value
    
    # Extract start and end date from the features
    for feature in result.getFeatures():
        start_date_value = feature['start_date']  # Extract the 'start_date'
        end_date_value = feature['end_date']      # Extract the 'end_date'
        break  # Only need to get these values from one feature since they are the same for all rows

    # Convert to YYYYMMDD format
    if start_date_value:  # Check if start_date_value is not None
        start_date_dt = datetime.strptime(start_date_value, '%Y-%m-%d')  # Adjust format as needed
        start_date_yyyymmdd = start_date_dt.strftime('%Y%m%d')

    if end_date_value:  # Check if end_date_value is not None
        end_date_dt = datetime.strptime(end_date_value, '%Y-%m-%d')  # Adjust format as needed
        end_date_yyyymmdd = end_date_dt.strftime('%Y%m%d')
        
    return start_date_yyyymmdd, end_date_yyyymmdd  # Return the formatted dates


def find_first_last_s2_images(s2_folder: str, start_date_str: str, end_date_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the first and last Sentinel-2 images within a specified date range.

    Args:
        s2_folder (str): Path to the folder containing Sentinel-2 images.
        start_date_str (str): Start date in 'YYYYMMDD' format.
        end_date_str (str): End date in 'YYYYMMDD' format.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the first and last image dates in 'YYYYMMDD' format,
        or None if no images are found.
    """
    # Convert start and end dates from string to datetime
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    # List to hold valid image dates
    valid_images = []

    # Iterate through files in the S2 folder
    for filename in os.listdir(s2_folder):
        if filename.startswith('S2SR_image_') and filename.endswith('.tif'):
            # Extract date from the filename
            date_str = filename.split('_')[2]  # This gets the YYYYMMDD part
            image_date = datetime.strptime(date_str, '%Y%m%d')
            
            # Check if the image date is within the specified range
            if start_date <= image_date <= end_date:
                valid_images.append(image_date)

    # Determine the first and last image dates
    if valid_images:
        first_image_date = min(valid_images)
        last_image_date = max(valid_images)
        return first_image_date.strftime('%Y%m%d'), last_image_date.strftime('%Y%m%d')

    return None, None  # Return None if no valid images found


def select_and_extract_by_expression(input, expression):
    selection = processing.run("native:extractbyexpression", 
     {'INPUT':input,
     'EXPRESSION': expression,
     'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    return selection

def buffer(input, distance):
    buffer = processing.run("native:buffer", 
     {'INPUT':input,
     'DISTANCE':500,
     'SEGMENTS':5,
     'END_CAP_STYLE':0,
     'JOIN_STYLE':0,
     'MITER_LIMIT':2,
     'DISSOLVE':False,
     'SEPARATE_DISJOINT':False,
     'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    return buffer


def clip_raster_by_mask_layer(input, mask, output_filepath):
    clip = processing.run("gdal:cliprasterbymasklayer", 
     {'INPUT': input_image_path,
     'MASK': buffer,  
     'SOURCE_CRS': None,
     'TARGET_CRS': None,
     'TARGET_EXTENT': None,
     'NODATA': None,
     'ALPHA_BAND': False,
     'CROP_TO_CUTLINE': True,
     'KEEP_RESOLUTION': False,
     'SET_RESOLUTION': False,
     'X_RESOLUTION': None,
     'Y_RESOLUTION': None,
     'MULTITHREADING': False,
     'OPTIONS': '',
     'DATA_TYPE': 0,
     'EXTRA': '',
     'OUTPUT': output_filepath 
    })['OUTPUT']
    
    return clip





def create_df_ccd_per_tile(my_folder, ccd_folder, tilename, fields_to_join):
    """
    Process a tile shapefile and optionally apply conditional formatting for specific tiles.
    
    Parameters:
    - my_folder: Path to the base folder.
    - ccd_folder: Subdirectory for CCD-related files.
    - tilename: Name of the tile file (e.g., 'tile29TNG.shp').
    
    Returns:
    - Path to the final processed shapefile.
    """
    # Define file paths
    fn_tile = str(my_folder / ccd_folder / 'tiles' / tilename)
    ln_output = tilename.replace('.shp', '_with_id_gleba_and_dates.shp')
    fn_output = str(my_folder / ccd_folder / 'tiles' / ln_output)
    
    # Read tile shapefile
    df_tile = gpd.read_file(fn_tile)
    
    # List of tiles requiring special handling for tBreak and tBreak_ddm
    tiles_needing_special_processing = ['tile29SNB.shp']  # Add exact filenames as needed
    
    # Conditional processing for specific tiles
    if tilename in tiles_needing_special_processing:
        # Replace NULL (NaN) values with '[]' and wrap all values in square brackets for tBreak
        df_tile['tBreak'] = df_tile['tBreak'].apply(lambda x: f'[{x}]' if pd.notna(x) else '[]')
        # Apply the custom format function to tBreak_ddm
        df_tile['tBreak_ddm'] = df_tile['tBreak_ddm'].apply(format_tBreak_ddm)
        # Ensure the columns are of string type
        df_tile['tBreak'] = df_tile['tBreak'].astype(str)
        df_tile['tBreak_ddm'] = df_tile['tBreak_ddm'].astype(str)
        # Save the modified GeoDataFrame back to a shapefile
        df_tile.to_file(fn_tile)
    
    # Join attributes by location
    fn_to_join = str(my_folder / ccd_folder / 'nvg_2018_ccd.gpkg|layername=nvg_2018_ccd')
    list_fields_to_join = fields_to_join
    
    # Assuming `join_field_by_location` is a defined function
    join_field_by_location(fn_tile, fn_to_join, list_fields_to_join, fn_output)
    
    # Read the resulting tile with joined fields
    df_tile = gpd.read_file(fn_output)
    
    # Process the GeoDataFrame (e.g., create drop_date column)
    # Assuming `create_drop_date` is a defined function
    df_tile = create_drop_date(df_tile)
    
    # Save the final GeoDataFrame to a shapefile
    output_shapefile = str(my_folder / ccd_folder / 'tiles' / f'df_ccd_{tilename}')
    df_tile.to_file(output_shapefile)
    
    return output_shapefile, fn_output, df_tile




def unique_id_glebas_per_tile(df_tile, output_file_name):
    """
    Create a list of unique id_glebas within a tile.
    This can work as a checklist for the visual analysis.
    
    Parameters:
    - df_tile: DataFrame containing the data for the tile.
    - output_file_name: Name of the output file (e.g., 'tile29TNG.csv').
    
    Returns:
    - Path to the final processed CSV file.
    """
    # Extract unique id_glebas
    unique_id_glebas = df_tile['id_gleba'].unique()
    
    # Create a DataFrame with the unique ids
    unique_id_glebas_df = pd.DataFrame(unique_id_glebas, columns=['id_gleba'])
    
    # Define the output file path
    unique_id_glebas_csv = str(my_folder / ccd_folder / 'tiles' / output_file_name)
    
    # Save to CSV
    unique_id_glebas_df.to_csv(unique_id_glebas_csv, index=False)
    
    return unique_id_glebas_csv



def merge_visual_analysis_single_shp(my_folder, ccd_folder, output_folder, fn_filename_output_va):
    """
    Process CSV files and shapefiles to generate a merged GeoDataFrame.
    
    Parameters:
        my_folder (str or Path): Base directory containing the input folders.
        ccd_folder (str or Path): Subdirectory containing the 'tiles' folder.
        output_folder (str or Path): Directory to search for the shapefiles.
    
    Returns:
        str: Path to the saved merged shapefile.
    """
    # Define the folder containing the files
    tile_folder = Path(my_folder) / ccd_folder / 'tiles'
    
    # Find all CSV files that start with "unique"
    unique_files = tile_folder.glob('unique*.csv')
    
    # Load and combine the files
    all_glebas = []
    for file in unique_files:
        df = pd.read_csv(file)
        all_glebas.append(df)
    
    # Combine all DataFrames and drop duplicates
    combined_glebas = pd.concat(all_glebas).drop_duplicates()
    
    # Convert the 'id_gleba' column to a list
    combined_glebas_list = combined_glebas['id_gleba'].tolist()
    
    # Stats on visual analysis for the specified tile
    nvg_single_2018 = Path(my_folder) / output_folder
    geodf_list = []
    
    # Load the corresponding shapefiles for id_glebas
    for id_gleba in combined_glebas_list:
        shapefile_path = nvg_single_2018 / f'nvg_singlepart_{id_gleba}.shp'
        
        # Check if the shapefile exists
        if shapefile_path.exists():
            # Read the shapefile and append it to the list
            geodf = gpd.read_file(shapefile_path)
            geodf_list.append(geodf)
        else:
            print(f"Shapefile for id_gleba {id_gleba} does not exist.")
    
    # Merge all GeoDataFrames
    merged_gdf = gpd.GeoDataFrame(pd.concat(geodf_list, ignore_index=True))
    
    # Save the merged GeoDataFrame to the output file
    merged_gdf.to_file(fn_filename_output_va)
    
    return fn_filename_output_va



def merge_ccd_shapefiles(tile_folder, columns_to_keep, fn_output_merged_ccd):
    """
    Merge all shapefiles starting with "df_ccd_" in a specified folder into a single GeoDataFrame.
    
    Parameters:
        tile_folder (str or Path): Directory containing the shapefiles.
        output_filename (str): Name of the output merged shapefile. Default is "merged_ccd_visual_analysis.shp".
    
    Returns:
        str: Path to the saved merged shapefile.
    """
    # Find all shapefiles that start with "df_ccd_"
    tile_folder = Path(tile_folder)
    ccd_files = tile_folder.glob('df_ccd_*.shp')
    
    # Load and combine all GeoDataFrames
    gdfs_ccd = []
    for file in ccd_files:
        gdf = gpd.read_file(file)
        gdfs_ccd.append(gdf)
    
    # Concatenate all GeoDataFrames into one
    merged_gdf_ccd = gpd.GeoDataFrame(pd.concat(gdfs_ccd, ignore_index=True))
    
    # Select the columns to keep
    columns_to_keep = columns_to_keep
    merged_gdf_ccd = merged_gdf_ccd[columns_to_keep]
    
    # Save the merged GeoDataFrame to a new shapefile
    merged_gdf_ccd.to_file(fn_output_merged_ccd)
    
    return fn_output_merged_ccd

def create_time_interval_columns(gdf_filepath, output_filepath):
    """
    Process CCD data to create 'data0' and 'data1' columns based on specified conditions.
    
    Parameters:
        gdf_filepath (str or Path): File path to the input shapefile.
        output_filepath (str or Path): File path for saving the processed shapefile.
    
    Returns:
        str: Path to the saved shapefile.
    """
    # Load the GeoDataFrame
    gdf = gpd.read_file(gdf_filepath)
    
    # Convert 'drop_date' to datetime
    gdf['drop_date'] = pd.to_datetime(gdf['drop_date'], format='%Y-%m-%d', errors='coerce')
    
    # Create and populate the 'data0' column
    gdf['data0'] = None
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD1'] == '1000-01-01'), 'data0'] = gdf['start_date']
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD1'] != '1000-01-01') & (gdf['ECCD1'].notnull()), 'data0'] = gdf['ECCD1']
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD1'].isnull()) & (gdf['NC'].isnull()), 'data0'] = (gdf['drop_date'] - pd.Timedelta(days=5)).dt.strftime('%Y-%m-%d')
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD1'].isnull()) & (gdf['NC'].notnull()), 'data0'] = None
    gdf.loc[gdf['drop_date'].isnull() & (gdf['ECCD1'].isnull()) & (gdf['NC'].isnull()), 'data0'] = None
    gdf.loc[gdf['drop_date'].isnull() & (gdf['ECCD1'] != '1000-01-01') & (gdf['ECCD1'].notnull()), 'data0'] = gdf['ECCD1']
    gdf.loc[gdf['drop_date'].isnull() & (gdf['ECCD1'] == '1000-01-01'), 'data0'] = gdf['start_date']
    
    # Create and populate the 'data1' column
    gdf['data1'] = None
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD2'] == '1000-01-01'), 'data1'] = gdf['end_date']
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD2'] != '1000-01-01') & (gdf['ECCD2'].notnull()), 'data1'] = gdf['ECCD2']
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD2'].isnull()) & (gdf['NC'].isnull()), 'data1'] = (gdf['drop_date'] + pd.Timedelta(days=5)).dt.strftime('%Y-%m-%d')
    gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD2'].isnull()) & (gdf['NC'].notnull()), 'data1'] = None
    gdf.loc[gdf['drop_date'].isnull() & (gdf['ECCD2'].isnull()) & (gdf['NC'].isnull()), 'data1'] = None
    gdf.loc[gdf['drop_date'].isnull() & (gdf['ECCD2'] != '1000-01-01') & (gdf['ECCD2'].notnull()), 'data1'] = gdf['ECCD2']
    gdf.loc[gdf['drop_date'].isnull() & (gdf['ECCD2'] == '1000-01-01'), 'data1'] = gdf['end_date']
    
    # Format 'drop_date' as string
    gdf['drop_date'] = gdf['drop_date'].dt.strftime('%Y-%m-%d')
    
    # Save the processed GeoDataFrame
    gdf.to_file(output_filepath)
    
    return str(output_filepath)


def calculate_ccd_correct_subparcels(gdf):
    ### calculate the total of CCD-correct sub-parcels
    ## CCD-correct
    total_subparcels = gdf['id'].nunique()
    total_parcels = gdf['id_gleba'].nunique()
    ccd_correct_1 = gdf.loc[gdf['drop_date'].notnull() & (gdf['ECCD1'].isnull()) & (gdf['ECCD2'].isnull()) & (gdf['NC'].isnull())] 
    ccd_correct_2 = gdf.loc[gdf['drop_date'].isnull() & (gdf['NC'] == 1)]
    
    # nr of sub-parcels CCD correct
    ccd_correct_subparcels1 = ccd_correct_1['id'].nunique()
    ccd_correct_subparcels2 = ccd_correct_2['id'].nunique()
    total_ccd_correct_sp = ccd_correct_subparcels1 + ccd_correct_subparcels2
    percentage_ccd_correct = (total_ccd_correct_sp / total_subparcels) * 100
    
    #nr of sub-parcels corrected in the visual analysis with new time interval
    ccd_eccd1 = gdf.loc[gdf['ECCD1'].notnull()]
    ccd_eccd1_subparcels = ccd_eccd1['id'].nunique()
    percentage_ccd_eccd = (ccd_eccd1_subparcels / total_subparcels) * 100
    
    # Return the results as a tuple
    return (ccd_correct_subparcels1, 
            ccd_correct_subparcels2, 
            total_ccd_correct_sp, 
            percentage_ccd_correct, 
            ccd_eccd1_subparcels, 
            percentage_ccd_eccd)


def calculate_totals(gdf):
    """
    Calculate the total number of pixels, parcels, and sub-parcels in the GeoDataFrame.
    
    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing the data.
    
    Returns:
    dict: A dictionary with total counts of pixels, parcels, and sub-parcels.
    """
    total_pixels = len(gdf)
    total_parcels = gdf['id_gleba'].nunique()
    total_subparcels = gdf['id'].nunique()
    
    return total_pixels, total_parcels, total_subparcels

def analyze_null_pixels(gdf):
    """
    Analyze null pixels, including total nulls, isolated nulls, and other statistics.
    
    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing the data.
    
    Returns:
    tuple: A tuple containing counts, percentages, and subsets related to null pixels.
    """
    # Subset for total nulls
    subset_total_nulls = gdf[gdf['drop_date'].isnull()]
    
    # Isolated nulls
    subset_isolated_nulls = subset_total_nulls[
        (subset_total_nulls['data0'].isnull() & 
         subset_total_nulls['data1'].isnull() & 
         subset_total_nulls['drop_date'].isnull() &
         subset_total_nulls['ECCD1'].isnull() & 
         subset_total_nulls['ECCD2'].isnull() & 
         subset_total_nulls['NC'].isnull())
    ]
    
    # Non-isolated nulls
    subset_nulls_not_iso = subset_total_nulls[~subset_total_nulls.index.isin(subset_isolated_nulls.index)]
    
    # Counts
    iso_null_count = len(subset_isolated_nulls)
    total_nulls_count = len(subset_total_nulls)
    non_iso_null_count = total_nulls_count - iso_null_count
    
    # Percentages
    total_pixels = len(gdf)
    percentage_nulls_total = (total_nulls_count / total_pixels) * 100
    percentage_iso_nulls_total = (iso_null_count / total_pixels) * 100
    percentage_iso_within_null = (iso_null_count / total_nulls_count) * 100
    
    # Return the results
    return (
        total_nulls_count, 
        iso_null_count, 
        non_iso_null_count, 
        percentage_nulls_total, 
        percentage_iso_nulls_total, 
        percentage_iso_within_null, 
        subset_total_nulls, 
        subset_isolated_nulls, 
        subset_nulls_not_iso
    )


def calculate_date_difference(gdf, col1, col2, output_file, new_col_name='date_diff'):
    """
    Processes a GeoDataFrame by calculating the difference in days between two date columns
    and saves the resulting GeoDataFrame to a file.

    Parameters:
        file_path (str): Path to the input GeoDataFrame file.
        col1 (str): Name of the first date column.
        col2 (str): Name of the second date column.
        output_file (str): Path to save the processed GeoDataFrame.
        new_col_name (str): Name of the new column to store the date difference. Default is 'date_diff'.
    
    Returns:
        GeoDataFrame: The processed GeoDataFrame.
    """
    
    # Create a subset where both columns are not null
    subset_gdf = gdf[gdf[col1].notnull() & gdf[col2].notnull()]
    print(f"Number of valid rows: {len(subset_gdf)}")
    
    # Convert columns to datetime format
    subset_gdf[col1] = pd.to_datetime(subset_gdf[col1], format='%Y-%m-%d', errors='coerce')
    subset_gdf[col2] = pd.to_datetime(subset_gdf[col2], format='%Y-%m-%d', errors='coerce')
    
    # Calculate the difference in days
    subset_gdf[new_col_name] = (subset_gdf[col2] - subset_gdf[col1]).dt.days
    
    # Convert columns back to string format
    subset_gdf[col1] = subset_gdf[col1].dt.strftime('%Y-%m-%d')
    subset_gdf[col2] = subset_gdf[col2].dt.strftime('%Y-%m-%d')
    
    # Save the processed GeoDataFrame to a file
    subset_gdf.to_file(output_file)
    
    return subset_gdf


def calculate_date_diff_stats(gdf, column):
    total_pixels_hist = len(gdf['date_diff'])
    print(f"Total Pixels (Histogram): {total_pixels_hist}")
    
    # difference = 10 days
    diff_equals_10 = len(gdf[gdf['date_diff'] == 10])
    perc_equals_10 = (diff_equals_10 / total_pixels_hist) * 100
    print(f"Percentage of differences = 10 days: {perc_equals_10:.2f}%")
    
    # difference < 10 days
    diff_lessthan_10 = len(gdf[gdf['date_diff'] < 10])
    perc_lessthan_10 = (diff_lessthan_10 / total_pixels_hist) * 100
    print(f"Percentage of differences < 10 days: {perc_lessthan_10:.2f}%")
    
    # difference > 10 and <= 30 days
    diff_between_10_and_30 = len(gdf[(gdf['date_diff'] > 10) & (gdf['date_diff'] <= 30)])
    perc_between_10_and_30 = (diff_between_10_and_30 / total_pixels_hist) * 100
    print(f"Percentage of differences > 10 and <= 30 days: {perc_between_10_and_30:.2f}%")
    
    # difference > 30 and <= 60 days
    diff_between_30_and_60 = len(gdf[(gdf['date_diff'] > 30) & (gdf['date_diff'] <= 60)])
    perc_between_30_and_60 = (diff_between_30_and_60 / total_pixels_hist) * 100
    print(f"Percentage of differences > 30 and <= 60 days: {perc_between_30_and_60:.2f}%")
    
    # difference > 60 days
    diff_more_than_60 = len(gdf[gdf['date_diff'] > 60])
    perc_morethan_60 = (diff_more_than_60 / total_pixels_hist) * 100
    print(f"Percentage of differences > 60 days: {perc_morethan_60:.2f}%")
    
    # total percentage
    total_percentage_diff = perc_equals_10 + perc_lessthan_10 + perc_between_10_and_30 + perc_between_30_and_60 + perc_morethan_60
    print(f"Total Percentage Difference: {total_percentage_diff:.2f}%")
    
    return total_percentage_diff, perc_equals_10, perc_lessthan_10, perc_between_10_and_30, perc_between_30_and_60, perc_morethan_60


def analyze_nc_values(subset_total_nulls, subset_nulls_not_iso):
    """
    Analyze NC values in null pixels, including counts for specific NC ranges.
    
    Parameters:
    subset_total_nulls (GeoDataFrame): GeoDataFrame of all null pixels.
    subset_nulls_not_iso (GeoDataFrame): GeoDataFrame of non-isolated null pixels.
    
    Returns:
    dict: A dictionary containing statistics about NC values.
    """
    # NC = 1
    null_nc1 = subset_total_nulls.loc[subset_total_nulls['NC'] == 1]
    subparcels_null_nc1 = null_nc1.groupby('id').nunique()
    
    # NC between 0.2 and <1
    nc_between_02_and_1 = subset_nulls_not_iso.loc[
        (subset_nulls_not_iso['NC'] >= 0.2) & (subset_nulls_not_iso['NC'] < 1)
    ]
    
    # NC is NULL
    nc_is_null = subset_nulls_not_iso.loc[subset_nulls_not_iso['NC'].isnull()]
    
    # Return the results
    return {
        "nc1_count": len(null_nc1),
        "subparcels_null_nc1_count": len(subparcels_null_nc1),
        "nc_between_02_and_1_count": len(nc_between_02_and_1),
        "nc_is_null_count": len(nc_is_null)
    }





































