import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from itertools import combinations

# List of subjects 
subjects = ['Subj1', 'Subj2', 'Subj3','Subj4', 'Subj4', 'Subj5','Subj6']  # Add all your subjects here

for subject in subjects:
    print(f"Processing {subject}...")

    # Paths to the CSV file and output files for the current subject
    base_path = f'/Users/aleksandraadler/Desktop/empat/eye_tracker/et_data/{subject}/averages'
    file_path = os.path.join(base_path, f'grand_averages_{subject}.csv')
    output_file_path = os.path.join(base_path, f'long_format_{subject}.csv')
    ttest_results_file_path = os.path.join(base_path, f'ttest_results_{subject}.txt')
    significant_times_csv_path = os.path.join(base_path, f'significant_times_{subject}.csv')
    plot_file_path = os.path.join(base_path, f'pupil_size_plot_{subject}.png')

    # Load the CSV file into a DataFrame with semicolon delimiter
    df = pd.read_csv(file_path, delimiter=';')

    # Replace empty strings with 0
    df.replace("", 0, inplace=True)

    # Convert the DataFrame from wide format to long format
    df_long = pd.melt(df, id_vars=[df.columns[0]], var_name='condition', value_name='pupil_value')

    # Rename the time column for consistency
    df_long.rename(columns={df.columns[0]: 'time_bin'}, inplace=True)
    df_long[['task', 'emotion']] = df_long['condition'].str.split('+', expand=True)

    # Convert time bins to milliseconds
    df_long['time_bin'] = df_long['time_bin'] * 20

    # Remove rows where pupil_value is smaller than -3 or larger than 3
    df_long = df_long[(df_long['pupil_value'] > -3) & (df_long['pupil_value'] < 3)]

    # Save the long format DataFrame to a new CSV file
    df_long.to_csv(output_file_path, index=False)
    print(f"Long format DataFrame saved to {output_file_path}")

    # Perform pairwise t-tests for all emotions between 750 and 2500 ms
    emotions = df_long['emotion'].unique()
    significant_times = []

    # Get all unique pairs of emotions
    emotion_pairs = list(combinations(emotions, 2))

    # Calculate the number of comparisons for Bonferroni correction
    time_bins_to_test = df_long[(df_long['time_bin'] >= 750) & (df_long['time_bin'] <= 2500)]['time_bin'].unique()
    num_comparisons = len(emotion_pairs) * len(time_bins_to_test)
    alpha = 0.05
    adjusted_alpha = alpha / num_comparisons

    # Perform t-tests for each pair
    ttest_results = []
    for emotion1, emotion2 in emotion_pairs:
        # Get the data for the two emotions
        data_emotion1 = df_long[(df_long['emotion'] == emotion1) & (df_long['time_bin'] >= 750) & (df_long['time_bin'] <= 2500)]
        data_emotion2 = df_long[(df_long['emotion'] == emotion2) & (df_long['time_bin'] >= 750) & (df_long['time_bin'] <= 2500)]
        
        # Perform t-tests at each time point
        for time_bin in time_bins_to_test:
            values1 = data_emotion1[data_emotion1['time_bin'] == time_bin]['pupil_value']
            values2 = data_emotion2[data_emotion2['time_bin'] == time_bin]['pupil_value']
            t_stat, p_val = ttest_ind(values1, values2, nan_policy='omit')
            if p_val < adjusted_alpha:
                ttest_results.append((time_bin, emotion1, emotion2, t_stat, p_val))
                significant_times.append((time_bin, emotion1, emotion2, p_val))

    # Convert significant times to a DataFrame
    significant_times_df = pd.DataFrame(significant_times, columns=['time_bin', 'emotion1', 'emotion2', 'p_value'])

    # Save significant t-test results to a text file
    ttest_results_df = pd.DataFrame(ttest_results, columns=['time_bin', 'emotion1', 'emotion2', 't_stat', 'p_value'])
    ttest_results_df.to_csv(ttest_results_file_path, index=False, sep='\t')
    print(f"T-test results saved to {ttest_results_file_path}")

    # Save significant times DataFrame to a CSV file
    significant_times_df.to_csv(significant_times_csv_path, index=False)
    print(f"Significant times saved to {significant_times_csv_path}")

    # Plotting pupil size change by condition
    plt.figure(figsize=(12, 6))

    # Group the data by the 'condition' column
    grouped = df_long.groupby('condition')

    # Define a color map for the conditions
    colors = plt.cm.tab10.colors

    # Plot each group with a different color
    for i, (condition, group) in enumerate(grouped):
        plt.plot(group['time_bin'], group['pupil_value'], color=colors[i % len(colors)], alpha=0.5, label=f'Condition {condition}')

    # Add labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Pupil Size')
    plt.title('Pupil Size change')

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

    # Limit the x-axis
    plt.xlim(-200, 3000)

    # Set x-axis ticks and labels to reflect every 500 ms
    plt.xticks(ticks=range(0, 3001, 500), labels=range(0, 3001, 500))

    # Add shaded area between 0 and 200 ms and 750 and 2500 ms
    plt.axvspan(0, 200, color='gray', alpha=0.3)
    plt.axvspan(750, 2500, color='red', alpha=0.2)

    # Highlight significant time points with vertical lines
    for time, emotion1, emotion2, p_val in significant_times:
        plt.axvline(x=time, color='red', linestyle='--', alpha=0.5)

    # Save the plot to a file
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")

    # Show the plot
    plt.show()
