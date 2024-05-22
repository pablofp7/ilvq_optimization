import os
import re

def insert_pd_set_option(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    inside_read_dataset = False
    pd_option_inserted = False

    for line in lines:
        updated_lines.append(line)
        if re.match(r'\s*def\s+read_dataset\s*\(\s*\)\s*:', line) and not pd_option_inserted:
            inside_read_dataset = True
        if inside_read_dataset and not pd_option_inserted:
            # Insert pd.set_option after the function definition
            updated_lines.append("    pd.set_option('future.no_silent_downcasting', True)\n")
            pd_option_inserted = True
            inside_read_dataset = False

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

def update_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                insert_pd_set_option(file_path)

# Set the directory you want to update
directory_to_update = '.'
update_directory(directory_to_update)
