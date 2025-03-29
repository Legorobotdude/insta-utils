import json
import os
import webbrowser
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

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

def analyze_post_metadata(posts_filepath, output_dir):
    """
    Analyzes metadata from the posts JSON file (posts_1.json).
    Calculates posting frequency, media type distribution, caption stats,
    and generates plots.

    Args:
        posts_filepath (str): Path to the posts JSON file (e.g., content/posts_1.json).
        output_dir (str): Directory to save generated plots.
    """
    print("--- Starting Post Metadata Analysis ---")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(posts_filepath, 'r', encoding='utf-8') as f:
            posts_data = json.load(f)
        print(f"Loaded data for {len(posts_data)} posts from {posts_filepath}")
    except FileNotFoundError:
        print(f"Error: Posts file not found at {posts_filepath}")
        print("Please ensure you have the correct path to your 'posts_1.json' file.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {posts_filepath}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading {posts_filepath}: {e}")
        return

    if not isinstance(posts_data, list) or not posts_data:
        print("Error: Posts data is not a valid list or is empty.")
        return

    # --- Data Extraction (Adapt based on actual file structure) ---
    extracted_data = []
    for post in posts_data:
        try:
            # Timestamp (adjust key if needed)
            timestamp = None
            if 'media' in post and isinstance(post['media'], list) and post['media']:
                 if 'creation_timestamp' in post['media'][0]:
                      timestamp = post['media'][0]['creation_timestamp']
            elif 'taken_at_timestamp' in post:
                 timestamp = post['taken_at_timestamp']
            
            if timestamp is None:
                print(f"Warning: Could not find timestamp for post: {post.get('uri', '[URI not found]')}")
                continue # Skip posts without timestamp
            
            post_time = datetime.fromtimestamp(timestamp)

            # Caption (adjust key if needed)
            caption = ""
            if 'media' in post and isinstance(post['media'], list) and post['media']:
                caption = post['media'][0].get('title', "")
            elif 'caption' in post:
                caption = post.get('caption', "")

            caption_length = len(caption)

            # Hashtags
            hashtags = re.findall(r"#(\w+)", caption)
            hashtag_count = len(hashtags)

            # Media Type (adjust logic based on actual keys)
            media_type = "Unknown"
            if 'media' in post and isinstance(post['media'], list) and post['media']:
                 # Check for carousels first
                 if 'node' in post and 'edge_sidecar_to_children' in post['node'] and 'edges' in post['node']['edge_sidecar_to_children']:
                      if len(post['node']['edge_sidecar_to_children']['edges']) > 1:
                           media_type = "Carousel"
                 # Check for video flag
                 elif 'node' in post and post['node'].get('is_video', False):
                       media_type = "Video"
                 # Infer from URI if possible
                 elif 'uri' in post['media'][0]:
                      uri = post['media'][0]['uri'].lower()
                      if uri.endswith( ('.mp4', '.mov') ): # Add other video extensions if needed
                          media_type = "Video"
                      elif uri.endswith( ('.jpg', '.jpeg', '.png', '.webp') ): # Add other image extensions if needed
                          media_type = "Image"

            extracted_data.append({
                'timestamp': post_time,
                'caption_length': caption_length,
                'hashtag_count': hashtag_count,
                'media_type': media_type,
                # Add like/comment counts here later if available
            })

        except Exception as e:
            print(f"Warning: Error processing a post, skipping. Error: {e}. Post data: {post}")
            continue

    if not extracted_data:
        print("Error: No valid post data could be extracted.")
        return

    # --- Convert to Pandas DataFrame ---
    df = pd.DataFrame(extracted_data)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    print("\n--- Summary Statistics ---")
    print(f"Total posts analyzed: {len(df)}")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print("\nCaption Length Stats:")
    print(df['caption_length'].describe())
    print("\nHashtag Count Stats:")
    print(df['hashtag_count'].describe())
    print("\nMedia Type Distribution:")
    print(df['media_type'].value_counts(normalize=True).map("{:.1%}".format))

    # --- Plotting --- 
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # 1. Posting Frequency Over Time (e.g., Monthly)
    try:
        posts_per_month = df.resample('ME').size()
        if not posts_per_month.empty:
            ax = posts_per_month.plot(kind='line', marker='o')
            ax.set_title('Posts Per Month Over Time')
            ax.set_ylabel('Number of Posts')
            ax.set_xlabel('Month')
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, 'posts_per_month.png')
            plt.savefig(plot_filename)
            print(f"\nSaved plot: {plot_filename}")
            plt.close()
        else:
            print("\nSkipping 'Posts Per Month' plot: No data after resampling.")
    except Exception as e:
        print(f"\nError generating 'Posts Per Month' plot: {e}")

    # 2. Media Type Distribution
    try:
        if not df['media_type'].value_counts().empty:
            plt.figure(figsize=(8, 8))
            media_counts = df['media_type'].value_counts()
            plt.pie(media_counts, labels=media_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Distribution of Media Types')
            plt.ylabel('') # Hide the default ylabel
            plot_filename = os.path.join(output_dir, 'media_type_distribution.png')
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plt.close()
        else:
             print("\nSkipping 'Media Type Distribution' plot: No media type data.")
    except Exception as e:
        print(f"\nError generating 'Media Type Distribution' plot: {e}")

    # 3. Caption Length Distribution
    try:
        if not df['caption_length'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['caption_length'], bins=30, kde=True)
            plt.title('Distribution of Caption Lengths')
            plt.xlabel('Caption Length (characters)')
            plt.ylabel('Number of Posts')
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, 'caption_length_distribution.png')
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plt.close()
        else:
             print("\nSkipping 'Caption Length Distribution' plot: No caption length data.")
    except Exception as e:
        print(f"\nError generating 'Caption Length Distribution' plot: {e}")

    # 4. Hashtag Count Distribution
    try:
        if not df['hashtag_count'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[df['hashtag_count'] > 0]['hashtag_count'], bins=20, kde=False) # Focus on posts with hashtags
            plt.title('Distribution of Hashtag Counts (Posts with >0 Hashtags)')
            plt.xlabel('Number of Hashtags')
            plt.ylabel('Number of Posts')
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, 'hashtag_count_distribution.png')
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plt.close()
        else:
             print("\nSkipping 'Hashtag Count Distribution' plot: No hashtag count data.")
    except Exception as e:
        print(f"\nError generating 'Hashtag Count Distribution' plot: {e}")


    print("--- Post Metadata Analysis Complete ---")

def track_unfollowers(current_followers_filepath, history_filepath="follower_history.json"):
    """
    Compares the current follower list against a previously saved list
    to identify users who have unfollowed. Updates the history file.

    Args:
        current_followers_filepath (str): Path to the latest followers JSON file.
        history_filepath (str): Path to the JSON file storing follower history.
    """
    print("--- Starting Follower Change Tracking ---")

    # 1. Load Current Followers
    print(f"Loading current followers from: {current_followers_filepath}")
    # Assuming the key is 'relationships_followers', adjust if needed
    current_followers_set = extract_usernames(current_followers_filepath, 'relationships_followers')

    if current_followers_set is None:
        print("Error: Could not load current followers. Aborting tracking.")
        return

    print(f"Found {len(current_followers_set)} current followers.")

    # 2. Load History
    previous_followers_set = set()
    last_checked_date = "Never"
    first_run = True

    try:
        if os.path.exists(history_filepath):
            with open(history_filepath, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            if isinstance(history_data, dict) and 'followers' in history_data and 'last_checked_date' in history_data:
                previous_followers_set = set(history_data['followers'])
                last_checked_date = history_data['last_checked_date']
                first_run = False
                print(f"Loaded previous follower list ({len(previous_followers_set)} followers) from {last_checked_date}.")
            else:
                print(f"Warning: History file '{history_filepath}' has incorrect format. Treating as first run.")
        else:
            print(f"History file '{history_filepath}' not found. Treating as first run.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from history file '{history_filepath}'. Treating as first run.")
    except Exception as e:
        print(f"An unexpected error occurred loading history file '{history_filepath}': {e}. Treating as first run.")

    # 3. Compare and Report
    if not first_run:
        unfollowed_set = previous_followers_set - current_followers_set
        new_followers_set = current_followers_set - previous_followers_set

        if unfollowed_set:
            print(f"\nUnfollowed since {last_checked_date}:")
            for user in sorted(list(unfollowed_set)):
                print(f"- {user}")
        else:
            print(f"\nNo users unfollowed since {last_checked_date}.")

        if new_followers_set:
             print(f"\nNew followers since {last_checked_date}:")
             for user in sorted(list(new_followers_set)):
                  print(f"+ {user}")
        else:
             print(f"\nNo new followers since {last_checked_date}.")

    else:
        print("\nThis is the first run, creating baseline follower history.")

    # 4. Update History File (only if current followers were loaded successfully)
    try:
        today_date = datetime.now().strftime("%Y-%m-%d")
        history_to_save = {
            "last_checked_date": today_date,
            "followers": sorted(list(current_followers_set)) # Save sorted list
        }
        with open(history_filepath, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4)
        print(f"\nUpdated follower history file: {history_filepath}")
    except IOError:
        print(f"\nError: Could not write to history file {history_filepath}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while saving history: {e}")

    print("--- Follower Change Tracking Complete ---")


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Instagram data export.")
    parser.add_argument(
        "task",
        choices=['unfollowers', 'post_metadata', 'track_unfollowers'],
        help="The analysis task to perform (e.g., 'unfollowers', 'post_metadata', 'track_unfollowers')"
    )
    parser.add_argument(
        '--output-dir',
        default='plots',
        help='Directory to save generated plots (for post_metadata task, default: plots)'
    )
    parser.add_argument(
        '--history-file',
        default='follower_history.json',
        help='Path to the follower history file (for track_unfollowers task, default: follower_history.json)'
    )

    args = parser.parse_args()

    # --- Execute requested task ---
    if args.task == 'unfollowers':
        # File paths specific to unfollowers task
        base_data_path = os.path.join('connections', 'followers_and_following')
        followers_file = os.path.join(base_data_path, 'followers_1.json')
        following_file = os.path.join(base_data_path, 'following.json')
        unfollowers_output_file = 'unfollowers.html'
        analyze_unfollowers(followers_file, following_file, unfollowers_output_file)

    elif args.task == 'post_metadata':
        # File path specific to post metadata task (assuming standard location)
        posts_file = os.path.join('content', 'posts_1.json') 
        analyze_post_metadata(posts_file, args.output_dir)

    elif args.task == 'track_unfollowers':
        # File path specific to follower tracking task
        followers_file = os.path.join('connections', 'followers_and_following', 'followers_1.json')
        track_unfollowers(followers_file, args.history_file)

    else:
        # Should not be reachable with choices defined
        print(f"Error: Unknown task '{args.task}'")
        parser.print_help()

    print("\nAnalysis script finished.") 