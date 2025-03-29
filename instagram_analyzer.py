import json
import os
import webbrowser
import argparse

# Define the paths to the JSON files relative to the script location
followers_file = os.path.join('connections', 'followers_and_following', 'followers_1.json')
following_file = os.path.join('connections', 'followers_and_following', 'following.json')
output_html_file = 'unfollowers.html' # Name of the output HTML file

def extract_usernames(filepath, data_key):
    """
    Extracts usernames from the Instagram JSON data structure.
    Handles potential file not found, JSON decoding errors, and unexpected data structures.

    Args:
        filepath (str): The path to the JSON file.
        data_key (str): The main key containing the list of relationships
                        (e.g., 'relationships_followers' or 'relationships_following').

    Returns:
        set: A set of usernames, or None if a critical error occurs.
    """
    usernames = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if the main data key exists and is a list
        if data_key in data and isinstance(data[data_key], list):
             relationship_list = data[data_key]
        # Sometimes the top level is the list itself (older formats?)
        elif isinstance(data, list):
             relationship_list = data
        else:
             print(f"Warning: Could not find expected data structure under key '{data_key}' or as a top-level list in {filepath}.")
             return set() # Return empty set if structure is unexpected

        for item in relationship_list:
            if 'string_list_data' in item and isinstance(item['string_list_data'], list):
                for entry in item['string_list_data']:
                    if 'value' in entry:
                        usernames.add(entry['value'])
                    else:
                         print(f"Warning: Found entry without 'value' key in {filepath}: {entry}")
            else:
                print(f"Warning: Found item without 'string_list_data' list in {filepath}: {item}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None # Indicate critical error
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None # Indicate critical error
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")
        return None # Indicate critical error

    return usernames

def generate_html_output(usernames_list, filename, title):
    """
    Generates an HTML file listing usernames with links to their profiles
    and attempts to open it in the default browser.

    Args:
        usernames_list (list): A list of usernames to include in the report.
        filename (str): The path to the output HTML file.
        title (str): The title to display in the HTML header.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang=\"en\">\n")
            f.write("<head>\n")
            f.write("    <meta charset=\"UTF-8\">\n")
            f.write("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
            f.write(f"    <title>{title}</title>\n")
            # Basic styling
            f.write("    <style>\n")
            f.write("        body { font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 600px; margin: auto; background-color: #f4f4f4; color: #333; }\n")
            f.write("        h1 { color: #d62976; text-align: center; }\n")
            f.write("        ul { list-style: none; padding: 0; }\n")
            f.write("        li { background-color: #fff; margin-bottom: 10px; padding: 10px 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n")
            f.write("        a { text-decoration: none; color: #405de6; font-weight: bold; }\n")
            f.write("        a:hover { text-decoration: underline; }\n")
            f.write("        p { text-align: center; margin-top: 20px; }\n")
            f.write("    </style>\n")
            f.write("</head>\n")
            f.write("<body>\n")

            f.write(f"    <h1>{title} ({len(usernames_list)})</h1>\n")

            if usernames_list:
                f.write("    <ul>\n")
                for username in usernames_list:
                    profile_url = f"https://www.instagram.com/{username}/"
                    f.write(f'        <li><a href="{profile_url}" target="_blank" rel="noopener noreferrer">{username}</a></li>\n')
                f.write("    </ul>\n")
            else:
                f.write("    <p>No accounts found for this report.</p>\n") # More generic message

            f.write("</body>\n")
            f.write("</html>\n")
        print(f"\nSuccessfully generated HTML file: {filename}")

        # Open the file in the browser
        try:
            file_path = os.path.abspath(filename)
            webbrowser.open(f'file://{file_path}')
            print(f"Attempting to open {file_path} in your default browser...")
        except Exception as e:
             print(f"Could not automatically open the file in browser: {e}")

    except IOError:
        print(f"\nError: Could not write to file {filename}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while generating HTML: {e}")


# --- Specific Analysis Functions ---

def analyze_unfollowers(followers_filepath, following_filepath, output_filename):
    """
    Analyzes follower/following lists to find users who don't follow back
    and generates an HTML report.

    Args:
        followers_filepath (str): Path to the followers JSON file.
        following_filepath (str): Path to the following JSON file.
        output_filename (str): Path for the output HTML report.
    """
    print("--- Starting Unfollower Analysis ---")

    print(f"Loading followers from: {followers_filepath}")
    # Adjust the key based on your specific file structure if needed
    followers = extract_usernames(followers_filepath, 'relationships_followers')
    if followers is None: # Check for critical error from extract_usernames
         print("Failed to load followers data. Aborting unfollower analysis.")
         return
    print(f"Found {len(followers)} followers.")

    print(f"\nLoading following from: {following_filepath}")
    # Adjust the key based on your specific file structure if needed
    following = extract_usernames(following_filepath, 'relationships_following')
    if following is None: # Check for critical error from extract_usernames
         print("Failed to load following data. Aborting unfollower analysis.")
         return
    print(f"Found {len(following)} accounts you follow.")

    # Calculate who doesn't follow back
    not_following_back = following - followers
    sorted_list = sorted(list(not_following_back))

    print(f"\nFound {len(sorted_list)} accounts that you follow but don't follow you back.")

    # Generate the HTML output file
    report_title = "Accounts You Follow Who Don't Follow Back"
    generate_html_output(sorted_list, output_filename, report_title)

    print("--- Unfollower Analysis Complete ---")


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Instagram data export.")
    parser.add_argument(
        "task",
        choices=['unfollowers'], # Add more choices here as we add functions
        help="The analysis task to perform (e.g., 'unfollowers')"
    )
    # Add other arguments here if needed in the future (e.g., --output-file)

    args = parser.parse_args()

    # Define file paths (relative to the script location)
    # These might need adjustment or become arguments themselves depending on the task
    base_data_path = os.path.join('connections', 'followers_and_following')
    followers_file = os.path.join(base_data_path, 'followers_1.json')
    following_file = os.path.join(base_data_path, 'following.json')
    unfollowers_output_file = 'unfollowers.html'

    # --- Execute requested task ---
    if args.task == 'unfollowers':
        analyze_unfollowers(followers_file, following_file, unfollowers_output_file)
    # --- Add elif blocks for future tasks ---
    # elif args.task == 'content_performance':
    #     content_posts_file = os.path.join('content', 'posts_1.json') # Example path
    #     analyze_content_performance(content_posts_file, 'content_report.html')
    else:
        # This part should ideally not be reachable if choices are defined correctly,
        # but it's good practice for robustness.
        print(f"Error: Unknown task '{args.task}'")
        parser.print_help()

    print("\nAnalysis script finished.") # Changed final message slightly 