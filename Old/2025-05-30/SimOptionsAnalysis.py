
import os
import SimOptions

LOADING_FOLDER = r"D:\NewGraphCompetition Outputs\Outputs"

all_simulation_options = []
for folder1 in os.listdir(LOADING_FOLDER):
    try:
        folder_path1 = os.path.join(LOADING_FOLDER, folder1)
        print(folder_path1)
        for folder in os.listdir(folder_path1):
            folder_path = os.path.join(folder_path1, folder)

            sample_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            for file in sample_files:
                simulation_options = SimOptions.NewSimOptions()
                simulation_options.load(file, folder_path)
                all_simulation_options.append(simulation_options)
    except:
        pass

result_dir = {}
for cur_simulation_option in all_simulation_options:

    for cur_option in cur_simulation_option.all_options.values():
        cur_option_name = cur_option.name
        cur_option_set = cur_option.option
        if cur_option_name not in list(result_dir.keys()):
            result_dir[cur_option_name] = {}
        if cur_option_set not in list(result_dir[cur_option_name].keys()):
            result_dir[cur_option_name][cur_option_set] = 0
        result_dir[cur_option_name][cur_option_set] += 1

print(result_dir)


# Function to calculate percentage and sort dynamically without hardcoding 42
def calculate_percentage_and_sort(input_dict):
    percentage_dict = {}
    max_keys_dict = {}  # To store the key of the largest percentage from each sub-dictionary

    # Iterate over the main dictionary
    for key, sub_dict in input_dict.items():
        # Calculate the total sum of values in the sub-dictionary
        total = sum(sub_dict.values())
        # Find the key with the maximum value in the sub-dictionary
        max_key = max(sub_dict, key=sub_dict.get)
        max_value = sub_dict[max_key]
        # Calculate percentage for the maximum value
        max_percentage = (max_value / total) * 100 if total != 0 else 0
        # Store the percentage and the key with the maximum value
        percentage_dict[key] = max_percentage
        max_keys_dict[key] = max_key  # Store the key with the max value

    # Sort the dictionary by percentage in descending order
    sorted_percentage_dict = {k: v for k, v in sorted(percentage_dict.items(), key=lambda item: item[1], reverse=True)}

    return sorted_percentage_dict, max_keys_dict


# Call the function and store the result
sorted_result, max_keys = calculate_percentage_and_sort(result_dir)

# Print the sorted result along with the key of the largest percentage
for key, percentage in sorted_result.items():
    print(f"{key}: {percentage:.2f}%, Largest percentage key: {max_keys[key]}")