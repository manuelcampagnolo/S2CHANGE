
def create_scatterplot(data, x_col, y_col, color_col, 
                       x_label='X-Axis', y_label='Y-Axis', title='Scatterplot', 
                       x_lim=(0, 20), y_lim=(0, 60), colorbar_label='Color Scale', 
                       figsize=(10, 6), cmap='hsv', alpha=0.7, grid=True):
    """
    Create a scatterplot with customizable options.

    Parameters:
    - data (DataFrame): The dataset containing the columns for plotting.
    - x_col (str): Column name for the x-axis values.
    - y_col (str): Column name for the y-axis values.
    - color_col (str): Column name for the color scale.
    - x_label (str): Label for the x-axis. Default is 'X-Axis'.
    - y_label (str): Label for the y-axis. Default is 'Y-Axis'.
    - title (str): Title of the scatterplot. Default is 'Scatterplot'.
    - x_lim (tuple): Limits for the x-axis as (min, max). Default is (0, 20).
    - y_lim (tuple): Limits for the y-axis as (min, max). Default is (0, 60).
    - colorbar_label (str): Label for the colorbar. Default is 'Color Scale'.
    - figsize (tuple): Figure size as (width, height). Default is (10, 6).
    - cmap (str): Colormap for the scatterplot. Default is 'hsv'.
    - alpha (float): Transparency level of the points. Default is 0.7.
    - grid (bool): Whether to show grid lines. Default is True.

    Returns:
    - None: Displays the scatterplot.
    """
    # Create the plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(data[x_col], data[y_col], c=data[color_col], cmap=cmap, alpha=alpha)
    
    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Set axis limits
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    
    # Add a colorbar
    plt.colorbar(scatter, label=colorbar_label)
    
    # Display grid if specified
    if grid:
        plt.grid(True)
    
    # Show the plot
    plt.show()


def add_area_field_to_layer(input_shapefile, output_shapefile, field_name='area_ha_subtalhao', 
                            field_type=1, field_length=5, field_precision=2):
    """
    Adds an area field to a shapefile and saves the updated shapefile.

    Parameters:
    - input_shapefile (str): Path to the input shapefile.
    - output_shapefile (str): Path to save the updated shapefile.
    - field_name (str): Name of the field to store area values. Default is 'area_ha_subtalhao'.
    - field_type (int): Field type (e.g., 1 for double). Default is 1.
    - field_length (int): Maximum length of the field. Default is 5.
    - field_precision (int): Number of decimal places. Default is 2.

    Returns:
    - str: Path to the saved shapefile.
    """
    # Load the input shapefile as a vector layer
    layer = QgsVectorLayer(input_shapefile, 'Input Layer', 'ogr')
    if not layer.isValid():
        raise ValueError(f"Failed to load the layer from {input_shapefile}")
    
    # Add the new field using QGIS processing
    area_singlepart = processing.run(
        "native:addfieldtoattributestable",
        {
            'INPUT': layer,
            'FIELD_NAME': field_name,
            'FIELD_TYPE': field_type,
            'FIELD_LENGTH': field_length,
            'FIELD_PRECISION': field_precision,
            'FIELD_ALIAS': '',
            'FIELD_COMMENT': '',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
    )
    
    output_layer = area_singlepart['OUTPUT']
    
    # Ensure the output layer is added to the QGIS project
    if output_layer not in QgsProject.instance().mapLayers().values():
        QgsProject.instance().addMapLayer(output_layer)
    
    # Start editing to calculate area
    if output_layer.isEditable() or output_layer.startEditing():
        for feature in output_layer.getFeatures():
            geom = feature.geometry()
            # Calculate area in hectares (1 hectare = 10,000 m²)
            area = geom.area() / 10000  
            feature[field_name] = area
            output_layer.updateFeature(feature)
        
        # Commit the changes
        output_layer.commitChanges()
    
    # Save the updated layer to a shapefile
    save_result = QgsVectorFileWriter.writeAsVectorFormat(
        output_layer, output_shapefile, "UTF-8", output_layer.crs(), "ESRI Shapefile"
    )
    
    if save_result[0] != QgsVectorFileWriter.NoError:
        raise ValueError(f"Error saving the file: {save_result[1]}")
    
    return output_shapefile



def create_histogram(data, column, bin_edges, title, x_label, y_label, log_scale=False, color='green', figsize=(10, 6)):
    """
    Creates and displays a histogram for a specified column in the dataset.

    Parameters:
        data (GeoDataFrame or DataFrame): The dataset containing the data to plot.
        column (str): The column to create the histogram for.
        bin_edges (list): List of bin edges for the histogram.
        title (str): Title of the histogram.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis. Default is False.
        color (str): Color of the histogram bars. Default is 'green'.
        figsize (tuple): Size of the figure. Default is (10, 6).
    
    Returns:
        None
    """
    # Plot the histogram
    plt.figure(figsize=figsize)
    counts, bins, patches = plt.hist(data[column], bins=bin_edges, edgecolor='black', color=color)
    
    # Add frequency labels on top of each bar
    for count, bin_edge in zip(counts, bins[:-1]):
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom')
    
    # Set titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    
    # Apply logarithmic scale if specified
    if log_scale:
        plt.yscale('log')
    
    # Display the plot
    plt.grid(False)
    plt.show()









































