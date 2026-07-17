import os
import csv

# Specify the directory path here
directory_path = '/home/user/Downloads/v1.0-trainval01/sweeps/LIDAR_TOP'  # Change this to your directory

# Get list of all files in the directory
filenames = [f for f in os.listdir(directory_path)]
print(filenames)
# Write filenames to CSV
csv_filename = 'file_list_sweeps.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(['filename'])  # Header
    for name in filenames:
        writer.writerow([name])

print(f"CSV file '{csv_filename}' created with {len(filenames)} filenames.")
