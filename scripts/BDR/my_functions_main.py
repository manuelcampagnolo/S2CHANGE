

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
        merged_df[date_columns] = merged_df[date_columns].apply(pd.to_datetime)

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
def extract_unique_id_gleba_from_nvg(input_shp):
    # Extract unique id_gleba values from the geopackage
    unique_id_gleba = set()
    with fiona.open(input_shp, 'r') as src:
        for feature in src:
            unique_id_gleba.add(feature['properties']['id_gleba'])
    return list(unique_id_gleba)


####### for id_gleba - no conditions just the first and last date
def filter_and_select_dates(df, id_gleba):
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

    # Select first and last date
    first_start_date = selected_data[corte_date_columns].min(axis=1)
    first_end_date = selected_data[corte_date_columns].max(axis=1)
    
    return first_start_date, first_end_date


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
    dates = sorted(selected_data[corte_date_columns].stack().dropna())

    if not dates:
        print(f"No valid dates found for 'CORTE' activity for ID {id_gleba}")
        return []  # Return an empty list if no valid dates are found

    # Convert dates from string to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

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

    return date_pairs


# CORRECT FUNCTION!!! for extract date pairs and identifying 2 years interval

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




# Function to add or subtract months from a date to pairs of dates
def add_subtract_months(date_str, months):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year + (date.month + months - 1) // 12
    month = (date.month + months - 1) % 12 + 1
    day = min(date.day, (date.replace(year=year, month=month, day=1) + timedelta(days=-1)).day)
    return datetime(year, month, day).strftime('%Y-%m-%d')


def dates_with_two_months_diff(date_pairs):
    # List to store modified date pairs
    modified_date_pairs = []
    new_start_dates = []
    new_end_dates = []
    # Loop through the date pairs, modify the dates and append to modified_date_pairs
    for start_date, end_date in date_pairs:
        new_start_date = add_subtract_months(start_date, -2)
        new_end_date = add_subtract_months(end_date, 2)
        # Convert modified dates back to strings
        modified_date_pairs.append((str(new_start_date), str(new_end_date)))
        new_start_dates.append(str(new_start_date))
        new_end_dates.append(str(new_end_date))
    return new_start_dates, new_end_dates, modified_date_pairs




def start_and_end_dates_two_months (first_start_date, first_end_date):
    # Convert start_date and end_date strings to datetime objects
    first_start_date = datetime.strptime(first_start_date, '%Y-%m-%d')  # Adjust format if needed
    first_end_date = datetime.strptime(first_end_date, '%Y-%m-%d')  # Adjust format if needed

    # Subtract 2 months from start_date
    start_date_minus_2_months = first_start_date - relativedelta(months=2)

    # Add 2 months to end_date
    end_date_plus_2_months = first_end_date + relativedelta(months=2)

    # Convert datetime objects to desired format if needed
    start_date = start_date_minus_2_months.strftime('%Y-%m-%d')
    end_date = end_date_plus_2_months.strftime('%Y-%m-%d')
    
    return start_date, end_date


# def start_and_end_dates_two_months(first_start_date, first_end_date):
#     if not isinstance(first_start_date, str) or not isinstance(first_end_date, str):
#         raise ValueError("Input dates must be strings")

#     try:
#         # Convert start_date and end_date strings to datetime objects
#         first_start_date = datetime.strptime(first_start_date, '%Y-%m-%d')
#         first_end_date = datetime.strptime(first_end_date, '%Y-%m-%d')

#         # Subtract 2 months from start_date
#         start_date_minus_2_months = first_start_date - relativedelta(months=2)

#         # Add 2 months to end_date
#         end_date_plus_2_months = first_end_date + relativedelta(months=2)

#         # Convert datetime objects back to strings in the desired format
#         start_date = start_date_minus_2_months.strftime('%Y-%m-%d')
#         end_date = end_date_plus_2_months.strftime('%Y-%m-%d')

#         return start_date, end_date
#     except ValueError as e:
#         print(f"Error processing dates: {e}")
#         return None, None



# Define a function to mask clouds using the Sentinel-2 QA band.
def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image \
        .updateMask(mask) \
        .divide(10000) \
        .copyProperties(image, ["system:index"])



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
    return medianNDVI.map(lambda feature: ee.Feature(feature).set('date', image.date().format('YYYY-MM-dd')))




def ndvi_median_gee(start_date, end_date, nvg, cloud_percentage):
    # Filter by Geo and Growing days
    S2_SR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filterBounds(nvg) \
            .map(lambda image: image.clip(nvg)) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_percentage) 

    # Apply across the whole collection 
    S2_NDVI = S2_SR.map(addNDVI)
    medianNDVI = S2_NDVI.map(calculateMedianNDVI).flatten()
    
    return medianNDVI






def ndvi_mediana_from_gee(start_date, end_date, modified_date_pairs, nvg, cloud_percentage, id_gleba):
    csv_paths = []  # List to store CSV paths
    
    for i, (start_date, end_date) in enumerate(modified_date_pairs):
        # Filter by Geo and Growing days
        S2_SR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(start_date, end_date) \
                .filterBounds(nvg) \
                .map(lambda image: image.clip(nvg)) \
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_percentage) 
        
        # Apply across the whole collection 
        S2_NDVI = S2_SR.map(addNDVI)
        medianNDVI = S2_NDVI.map(calculateMedianNDVI).flatten()
        
        # Convert 'medianNDVI' to a FeatureCollection if needed
        # medianNDVI_fc = medianNDVI.map(lambda image: ee.Feature(None, image))
        
        output_dir = str(my_folder / ndvi_folder)
        
        # Export the result as a CSV file using geemap
        geemap.ee_to_csv(
            ee_object=medianNDVI,
            filename=os.path.join(output_dir, f'Median_NDVI_{id_gleba}_{i}.csv')
        )
        
        # get csv file
        ln_median_ndvi = f'Median_NDVI_{id_gleba}_{i}.csv'
        csv_path = str(my_folder / ndvi_folder / ln_median_ndvi)
        csv_paths.append(csv_path)
        
    return csv_paths

#s2cloudless

def add_cloud_bands(image):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(image.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return image.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(image):
    # Identify water pixels from the SCL band.
    not_water = image.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = image.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (image.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': image.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(image):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(image)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': image.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(image):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = image.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return image.select('B.*').updateMask(not_cld_shdw)



def get_s2_sr_cld_col(nvg, start_date, end_date):
    # Filter by Geo and Growing days
    S2_SR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filterBounds(nvg) \
            .map(lambda image: image.clip(nvg))
 
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(nvg)
        .filterDate(start_date, end_date))
    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': S2_SR,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def convert_to_pivot_table(df_median_ndvi):
    # Convert the date column to string format with 'yyyymmdd' format
    df_median_ndvi['date'] = pd.to_datetime(df_median_ndvi['date']).dt.strftime('%Y%m%d')
    # Pivot the DataFrame
    pivot_table = df_median_ndvi.pivot_table(index=['id', 'id_gleba'], columns='date', values='median', aggfunc='first')
    # Reset index to make 'id' a column again
    pivot_table.reset_index(inplace=True)
    
    return pivot_table


def find_closest_date(row):
    id_gleba = row['id_gleba']
    date_of_biggest_drop = row['date_of_biggest_drop']
    closest_date = None
    min_time_diff = float('inf')
    
    # Iterate through df_sorted to find the closest date before 'date_of_biggest_drop'
    for col in df_sorted.columns:
        if col.startswith('data'):
            for _, sorted_row in df_sorted[df_sorted['id_gleba'] == id_gleba].iterrows():
                if pd.notna(sorted_row[col]):  # Skip NaN values
                    sorted_date = datetime.strptime(str(sorted_row[col]), '%Y-%m-%d')
                    if sorted_date < date_of_biggest_drop:  # Consider only dates before 'date_of_biggest_drop'
                        time_diff = abs((date_of_biggest_drop - sorted_date).total_seconds())
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_date = sorted_row[col]
    
    return closest_date

def calculate_biggest_ndvi_drop_and_estimated_date(pivot_table):
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
    # Apply the function to create the 'estimated_date' column in df_pivot_table
    pivot_table['estimated_date'] = pivot_table.apply(find_closest_date, axis=1)
    #rename
    pivot_table.rename(columns={col: 'date_' + col for col in pivot_table.columns if col.startswith('20')}, inplace=True)
    pivot_table_with_estimated_date = pivot_table
    
    return pivot_table_with_estimated_date




def join_attribute_to_layer(input1, field1, input2, field2, attribute_to_copy):
    result = processing.run("native:joinattributestable", {
        'INPUT': input1, 
        'FIELD': field1,
        'INPUT_2': input2,
        'FIELD_2': field2,
        'FIELDS_TO_COPY': [attribute_to_copy],
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














