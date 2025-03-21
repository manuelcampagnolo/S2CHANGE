# Run this file with the command:
# python parquet_compare.py <path_to_sklearn_parquet> <path_to_lasso_parquet> --output-dir <output_directory_path (default ./analysis_results)>
"""
Compare two parquet files from PyCCD to analyze differences between sklearn and Cython implementations.
This script identifies differences in the structure and content of the files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime

def load_parquet_files(sklearn_file, cython_file):
    """Load the two parquet files and return as dataframes"""
    print(f"Loading sklearn results from: {sklearn_file}")
    df_sklearn = pd.read_parquet(sklearn_file)
    print(f"Found {len(df_sklearn)} records in sklearn results")
    
    print(f"Loading Cython results from: {cython_file}")
    df_cython = pd.read_parquet(cython_file)
    print(f"Found {len(df_cython)} records in Cython results")
    
    return df_sklearn, df_cython

def identify_common_keys(df_sklearn, df_cython):
    """Identify keys that exist in both dataframes for comparison"""
    sklearn_keys = set(df_sklearn.columns)
    cython_keys = set(df_cython.columns)
    
    common_keys = sklearn_keys.intersection(cython_keys)
    sklearn_only = sklearn_keys - cython_keys
    cython_only = cython_keys - sklearn_keys
    
    print(f"\nFound {len(common_keys)} common columns")
    if sklearn_only:
        print(f"Columns only in sklearn: {sklearn_only}")
    if cython_only:
        print(f"Columns only in Cython: {cython_only}")
        
    return list(common_keys)

def analyze_column_statistics(df1, df2, common_keys, output_dir, name1="sklearn", name2="cython"):
    """
    Analyze statistical differences between columns in the two dataframes
    """
    print(f"\n--- Statistical Comparison of {name1} vs {name2} ---")
    
    # Create a DataFrame to store comparison results
    comparison_results = []
    
    for col in common_keys:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df1[col]) or not pd.api.types.is_numeric_dtype(df2[col]):
            continue
            
        # Calculate basic statistics for both dataframes
        stats1 = df1[col].describe()
        stats2 = df2[col].describe()
        
        # Calculate percent difference for key metrics
        percent_diff = {}
        for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            if stat in stats1 and stat in stats2 and stats2[stat] != 0:
                diff = ((stats1[stat] - stats2[stat]) / stats2[stat]) * 100
                percent_diff[f"{stat}_pct_diff"] = diff
            else:
                percent_diff[f"{stat}_pct_diff"] = np.nan
                
        # Combine results
        result = {
            'column': col,
            f'{name1}_count': stats1['count'],
            f'{name2}_count': stats2['count'],
            f'{name1}_mean': stats1['mean'],
            f'{name2}_mean': stats2['mean'],
            f'{name1}_std': stats1['std'],
            f'{name2}_std': stats2['std'],
            f'{name1}_min': stats1['min'],
            f'{name2}_min': stats2['min'],
            f'{name1}_max': stats1['max'],
            f'{name2}_max': stats2['max'],
            'mean_pct_diff': percent_diff['mean_pct_diff'],
            'std_pct_diff': percent_diff['std_pct_diff']
        }
        
        comparison_results.append(result)
    
    # Convert to DataFrame
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save to CSV
        output_path = Path(output_dir) / f'statistical_comparison_{name1}_vs_{name2}.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"Statistical comparison saved to: {output_path}")
        
        # Print summary of differences
        print("\nSummary of key differences (columns with >5% difference in mean):")
        significant_diffs = comparison_df[abs(comparison_df['mean_pct_diff']) > 5].sort_values(by='mean_pct_diff', ascending=False)
        
        if len(significant_diffs) > 0:
            for _, row in significant_diffs.iterrows():
                print(f"Column {row['column']}: {name1} mean = {row[f'{name1}_mean']:.4f}, {name2} mean = {row[f'{name2}_mean']:.4f} (diff: {row['mean_pct_diff']:.2f}%)")
        else:
            print("No columns with significant mean differences found.")
        
        return comparison_df
    else:
        print("No numerical columns found for comparison.")
        return pd.DataFrame()

def create_value_distributions(df_sklearn, df_cython, common_keys, output_dir):
    """Create histograms comparing value distributions between datasets"""
    # Select numerical columns
    numerical_cols = [col for col in common_keys 
                      if pd.api.types.is_numeric_dtype(df_sklearn[col]) and 
                         pd.api.types.is_numeric_dtype(df_cython[col])]
    
    # Limit to a reasonable number of interesting columns
    if len(numerical_cols) > 6:
        # Try to find interesting columns first (look for model parameters, quality metrics, etc.)
        interesting_patterns = ['rmse', 'coef', 'intercept', 'residual', 'score', 'quality', 'change']
        interesting_cols = [col for col in numerical_cols 
                          if any(pattern in col.lower() for pattern in interesting_patterns)]
        
        if len(interesting_cols) > 0:
            numerical_cols = interesting_cols[:6]
        else:
            numerical_cols = numerical_cols[:6]
    
    if not numerical_cols:
        print("No numerical columns found for distribution visualization.")
        return
    
    # Set up the plots
    n_cols = min(2, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        # Get the axis
        ax = axes[i]
        
        # Create labels for legend
        sklearn_label = f"sklearn (n={len(df_sklearn)})"
        cython_label = f"cython (n={len(df_cython)})"
        
        # Plot histograms with KDE
        try:
            sns.histplot(df_sklearn[col].dropna(), kde=True, alpha=0.4, 
                        color='blue', label=sklearn_label, ax=ax)
            
            sns.histplot(df_cython[col].dropna(), kde=True, alpha=0.4, 
                        color='red', label=cython_label, ax=ax)
            
            ax.set_title(f'Distribution of {col}')
            ax.legend()
            
            # Set reasonable axis limits to handle outliers
            combined_data = pd.concat([df_sklearn[col].dropna(), df_cython[col].dropna()])
            if not combined_data.empty:
                q1, q3 = np.percentile(combined_data, [5, 95])
                iqr = q3 - q1
                ax.set_xlim([q1 - 1.5*iqr, q3 + 1.5*iqr])
        except Exception as e:
            print(f"Error plotting column {col}: {str(e)}")
            ax.text(0.5, 0.5, f"Error plotting {col}", ha='center', va='center')
            ax.set_visible(False)
    
    # Hide unused subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'value_distributions.png'
    plt.savefig(output_path)
    plt.close()
    print(f"\nValue distributions visualization saved to: {output_path}")

def analyze_sample_records(df_sklearn, df_cython, common_keys, output_dir):
    """Extract and analyze sample records from both datasets"""
    # Take a small sample from each dataset
    sample_size = min(5, min(len(df_sklearn), len(df_cython)))
    
    if sample_size == 0:
        print("One or both datasets are empty. Cannot analyze sample records.")
        return
    
    sklearn_sample = df_sklearn.sample(sample_size, random_state=42) if len(df_sklearn) >= sample_size else df_sklearn
    cython_sample = df_cython.sample(sample_size, random_state=42) if len(df_cython) >= sample_size else df_cython
    
    # Save samples to CSV
    sklearn_path = Path(output_dir) / 'sklearn_sample_records.csv'
    cython_path = Path(output_dir) / 'cython_sample_records.csv'
    
    sklearn_sample.to_csv(sklearn_path, index=False)
    cython_sample.to_csv(cython_path, index=False)
    
    print(f"\nSaved {sample_size} sample records from sklearn to: {sklearn_path}")
    print(f"Saved {sample_size} sample records from cython to: {cython_path}")
    
    # Look for nested content like JSON
    json_columns = []
    for col in common_keys:
        # Check if column might contain JSON or complex objects
        if df_sklearn[col].dtype == 'object':
            try:
                # Try to parse as JSON if it's a string
                sample_val = df_sklearn[col].iloc[0]
                if isinstance(sample_val, str) and (sample_val.startswith('{') or sample_val.startswith('[')):
                    json.loads(sample_val)
                    json_columns.append(col)
            except:
                pass
    
    if json_columns:
        print("\nDetected potential JSON columns: ", json_columns)
        print("Consider manually extracting and comparing these complex structures.")

def analyze_model_structure(df_sklearn, df_cython, output_dir):
    """Analyze model structure to identify possible algorithmic differences"""
    print("\n--- Model Structure Analysis ---")
    
    # Extract information about model structure if available
    structure_indicators = {
        'sklearn_records': len(df_sklearn),
        'cython_records': len(df_cython),
        'record_difference': len(df_sklearn) - len(df_cython),
        'percent_difference': ((len(df_sklearn) - len(df_cython)) / len(df_sklearn) * 100) if len(df_sklearn) > 0 else 0
    }
    
    # Look for specific indicators in column names
    model_indicators = ['model', 'coefficient', 'coef', 'intercept', 'rmse', 'alpha']
    
    # Check both dataframes for these indicators
    found_columns = {}
    for indicator in model_indicators:
        sklearn_cols = [col for col in df_sklearn.columns if indicator in col.lower()]
        cython_cols = [col for col in df_cython.columns if indicator in col.lower()]
        
        if sklearn_cols or cython_cols:
            found_columns[indicator] = {
                'sklearn': sklearn_cols,
                'cython': cython_cols
            }
    
    # Write analysis to file
    with open(Path(output_dir) / 'model_structure_analysis.txt', 'w') as f:
        f.write("Model Structure Analysis\n")
        f.write("=====================\n\n")
        f.write(f"sklearn records: {structure_indicators['sklearn_records']}\n")
        f.write(f"cython records: {structure_indicators['cython_records']}\n")
        f.write(f"Difference: {structure_indicators['record_difference']} records\n")
        f.write(f"Percent difference: {structure_indicators['percent_difference']:.2f}%\n\n")
        
        f.write("Model-related columns:\n")
        for indicator, cols in found_columns.items():
            f.write(f"\n{indicator.upper()} related columns:\n")
            f.write(f"  sklearn: {cols['sklearn']}\n")
            f.write(f"  cython: {cols['cython']}\n")
            
    print(f"Model structure analysis saved to: {Path(output_dir) / 'model_structure_analysis.txt'}")
    
    return structure_indicators

def compare_model_parameters(df_sklearn, df_cython, common_keys, output_dir):
    """Compare model parameters between sklearn and cython implementations"""
    # Look for coefficient and intercept columns
    coef_cols = [col for col in common_keys if 'coef' in col.lower()]
    intercept_cols = [col for col in common_keys if 'intercept' in col.lower()]
    
    if not (coef_cols or intercept_cols):
        print("\nNo coefficient or intercept columns found for comparison.")
        return
    
    print("\n--- Model Parameter Comparison ---")
    
    # Analyze coefficients
    if coef_cols:
        print(f"Comparing coefficient columns: {coef_cols}")
        for col in coef_cols:
            if pd.api.types.is_numeric_dtype(df_sklearn[col]) and pd.api.types.is_numeric_dtype(df_cython[col]):
                # Calculate basic statistics
                sklearn_mean = df_sklearn[col].mean()
                cython_mean = df_cython[col].mean()
                
                if cython_mean != 0:
                    diff_pct = ((sklearn_mean - cython_mean) / cython_mean) * 100
                    print(f"{col}: sklearn mean = {sklearn_mean:.4f}, cython mean = {cython_mean:.4f}, diff = {diff_pct:.2f}%")
    
    # Analyze intercepts
    if intercept_cols:
        print(f"\nComparing intercept columns: {intercept_cols}")
        for col in intercept_cols:
            if pd.api.types.is_numeric_dtype(df_sklearn[col]) and pd.api.types.is_numeric_dtype(df_cython[col]):
                # Calculate basic statistics
                sklearn_mean = df_sklearn[col].mean()
                cython_mean = df_cython[col].mean()
                
                if cython_mean != 0:
                    diff_pct = ((sklearn_mean - cython_mean) / cython_mean) * 100
                    print(f"{col}: sklearn mean = {sklearn_mean:.4f}, cython mean = {cython_mean:.4f}, diff = {diff_pct:.2f}%")
    
    # Create correlation plots for model parameters
    param_cols = coef_cols + intercept_cols
    if len(param_cols) > 0:
        try:
            # Create a wide-format dataset for correlation analysis
            param_data = pd.DataFrame()
            for col in param_cols:
                if pd.api.types.is_numeric_dtype(df_sklearn[col]) and pd.api.types.is_numeric_dtype(df_cython[col]):
                    param_data[f'{col}_sklearn'] = df_sklearn[col]
                    param_data[f'{col}_cython'] = df_cython[col]
            
            if param_data.shape[1] > 0:
                # Calculate correlation matrix
                corr = param_data.corr()
                
                # Plot correlation matrix
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Correlation Matrix of Model Parameters')
                plt.tight_layout()
                
                # Save figure
                output_path = Path(output_dir) / 'parameter_correlations.png'
                plt.savefig(output_path)
                plt.close()
                print(f"Parameter correlation matrix saved to: {output_path}")
        except Exception as e:
            print(f"Error creating correlation plot: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Compare sklearn and Cython PyCCD results')
    parser.add_argument('sklearn_file', type=str, help='Path to sklearn results parquet file')
    parser.add_argument('cython_file', type=str, help='Path to Cython results parquet file')
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_sklearn, df_cython = load_parquet_files(args.sklearn_file, args.cython_file)
    
    # Identify common columns
    common_keys = identify_common_keys(df_sklearn, df_cython)
    
    # Analyze statistical differences
    stats_comparison = analyze_column_statistics(df_sklearn, df_cython, common_keys, output_dir)
    
    # Create value distribution plots
    create_value_distributions(df_sklearn, df_cython, common_keys, output_dir)
    
    # Analyze sample records
    analyze_sample_records(df_sklearn, df_cython, common_keys, output_dir)
    
    # Analyze model structure
    structure_info = analyze_model_structure(df_sklearn, df_cython, output_dir)
    
    # Compare model parameters
    compare_model_parameters(df_sklearn, df_cython, common_keys, output_dir)
    
    # Write summary report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(f"PyCCD Implementation Comparison - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"sklearn file: {args.sklearn_file}\n")
        f.write(f"Cython file: {args.cython_file}\n\n")
        f.write(f"Total sklearn records: {len(df_sklearn)}\n")
        f.write(f"Total Cython records: {len(df_cython)}\n")
        f.write(f"Record difference: {len(df_sklearn) - len(df_cython)} ({((len(df_sklearn) - len(df_cython)) / len(df_sklearn) * 100):.2f}%)\n\n")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()