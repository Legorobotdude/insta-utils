import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import csv # Added for manual parsing
import io # Added for manual parsing
import re # Ensure re is imported
import webbrowser # Added for HTML report generation
import json # Added
import urllib.request # Added
import urllib.error # Added

# --- Config Loading ---
def load_config(config_path):
    """Loads the JSON configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if "llm_settings" not in config or "user_context" not in config:
            print(f"Error: Config file '{config_path}' missing 'llm_settings' or 'user_context' section.")
            return None
        print(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'. Please create it or check the path.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from config file '{config_path}'. Check format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading config file '{config_path}': {e}")
        return None

# --- Data Loading Functions ---

def load_timeseries_csv(filepath, value_column_name):
    """Loads time-series data (Date, Primary) from common dashboard CSVs."""
    try:
        df = pd.read_csv(filepath, sep=',', skiprows=2, header=0, encoding='utf-16')
        df.rename(columns={'Primary': value_column_name}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date', value_column_name], inplace=True)
        df.set_index('Date', inplace=True)
        df[value_column_name] = pd.to_numeric(df[value_column_name], errors='coerce')
        df.dropna(subset=[value_column_name], inplace=True) # Drop if value conversion failed
        print(f"Successfully loaded {value_column_name} data: {len(df)} days.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading time-series data from {filepath}: {e}")
        return None

def load_audience_data(filepath):
    """Loads and parses the multi-section Audience CSV file."""
    data_frames = {}
    try:
        # Read all lines first to find section boundaries
        with open(filepath, 'r', encoding='utf-16') as f:
            lines = f.readlines()

        # Find line numbers for headers (0-indexed)
        # Remove quotes AND the initial sep= line if present
        header_lines = {line.strip().replace('"', ''): i 
                        for i, line in enumerate(lines) 
                        if line.strip().startswith('"') or line.strip().startswith('sep=')}
        
        # --- 1. Age & Gender --- 
        age_gender_header = "Age & gender"
        cities_header = "Top cities"
        if age_gender_header in header_lines:
            start_row = header_lines[age_gender_header] + 1 # Skip the header itself
            end_row_marker = len(lines)
            if cities_header in header_lines:
                 end_row_marker = header_lines[cities_header]
            
            # Calculate number of rows for age/gender data
            # We need the line index of the header row + 1 to start reading the data table header
            age_data_start_line = header_lines[age_gender_header] + 1 
            num_rows = end_row_marker - age_data_start_line - 1 # Subtract header row and potential blank lines
            
            if num_rows > 0:
                # Use skiprows to get to the actual data table header
                df_age_gender = pd.read_csv(filepath, sep=',', encoding='utf-16',
                                            skiprows=age_data_start_line, 
                                            nrows=num_rows,
                                            index_col=0)
                df_age_gender = df_age_gender.apply(pd.to_numeric, errors='coerce') 
                df_age_gender.index.name = 'Age Range'
                data_frames['age_gender'] = df_age_gender
                print("Successfully loaded Age & Gender data.")
            else:
                 print("Warning: Could not determine valid rows for Age & Gender data.")
        else:
            print("Warning: 'Age & gender' section header not found.")

        # --- 2. Top Cities --- 
        countries_header = "Top countries"
        if cities_header in header_lines:
            city_name_line_index = header_lines[cities_header] + 1
            city_value_line_index = city_name_line_index + 1
            end_row_marker = len(lines)
            if countries_header in header_lines:
                 end_row_marker = header_lines[countries_header]

            if city_value_line_index < end_row_marker:
                # Manual parsing using csv module to handle quotes correctly
                city_names_line = lines[city_name_line_index]
                city_values_line = lines[city_value_line_index]
                
                # Use io.StringIO to treat the string line as a file for csv.reader
                city_names_reader = csv.reader(io.StringIO(city_names_line))
                city_names = next(city_names_reader)
                
                city_values_reader = csv.reader(io.StringIO(city_values_line))
                city_values_str = next(city_values_reader)
                city_values = [pd.to_numeric(v, errors='coerce') for v in city_values_str]

                if len(city_names) == len(city_values):
                    df_cities = pd.DataFrame({'City': city_names, 'Percentage': city_values})
                    df_cities.dropna(inplace=True)
                    df_cities.sort_values(by='Percentage', ascending=False, inplace=True)
                    data_frames['top_cities'] = df_cities
                    print("Successfully loaded Top Cities data.")
                else:
                     print(f"Warning: Mismatch city names ({len(city_names)}) and values ({len(city_values)}).")
            else:
                print("Warning: Could not determine valid rows for Top Cities data.")
        else:
            print("Warning: 'Top cities' section header not found.")

        # --- 3. Top Countries --- 
        if countries_header in header_lines:
            country_name_line_index = header_lines[countries_header] + 1
            country_value_line_index = country_name_line_index + 1
            end_row_marker = len(lines) # Assume ends at file end

            if country_value_line_index < end_row_marker:
                # Manual parsing
                country_names_line = lines[country_name_line_index]
                country_values_line = lines[country_value_line_index]
                
                country_names_reader = csv.reader(io.StringIO(country_names_line))
                country_names = next(country_names_reader)
                
                country_values_reader = csv.reader(io.StringIO(country_values_line))
                country_values_str = next(country_values_reader)
                country_values = [pd.to_numeric(v, errors='coerce') for v in country_values_str]

                if len(country_names) == len(country_values):
                    df_countries = pd.DataFrame({'Country': country_names, 'Percentage': country_values})
                    df_countries.dropna(inplace=True)
                    df_countries.sort_values(by='Percentage', ascending=False, inplace=True)
                    data_frames['top_countries'] = df_countries
                    print("Successfully loaded Top Countries data.")
                else:
                     print(f"Warning: Mismatch country names ({len(country_names)}) and values ({len(country_values)}).")
            else:
                print("Warning: Could not determine valid rows for Top Countries data.")
        else:
            print("Warning: 'Top countries' section header not found.")

        return data_frames

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading or parsing audience data from {filepath}: {e}")
        # Raise e # Optional: re-raise for debugging
        return None

def load_post_performance_data(filepath):
    """Loads detailed post performance data from the CSV file."""
    try:
        # Revert to default encoding (likely UTF-8)
        df = pd.read_csv(filepath, sep=',', header=0) # Removed encoding parameter
        
        # Rename columns for easier access (adjust if names differ slightly)
        # Let's be explicit with renaming to handle variations
        rename_map = {
            'Publish time': 'PublishTime',
            'Post type': 'PostType',
            # Add other metrics if needed
        }
        # Only rename columns that exist in the DataFrame
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Convert PublishTime to datetime
        if 'PublishTime' in df.columns:
             df['PublishTime'] = pd.to_datetime(df['PublishTime'], errors='coerce')
        else:
             print("Warning: 'Publish time' column not found.")
             # Consider exiting or handling if this is critical

        # Convert metric columns to numeric
        metric_cols = ['Views', 'Reach', 'Likes', 'Shares', 'Comments', 'Saves']
        for col in metric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                print(f"Warning: Metric column '{col}' not found.")
        
        print(f"Successfully loaded post performance data: {len(df)} posts.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading post performance data from {filepath}: {e}")
        return None

# TODO: Add function to load other CSVs as needed

# --- Analysis & Plotting Functions ---

def plot_timeseries_trend(df, value_column_name, title, ylabel, output_filepath):
    """Plots a generic time-series trend over time and saves it."""
    if df is None or df.empty:
        print(f"Cannot plot {title}: DataFrame is empty or None.")
        return
    if value_column_name not in df.columns:
        print(f"Cannot plot {title}: Column '{value_column_name}' not found in DataFrame.")
        return
    try:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(14, 7))
        
        ax = sns.lineplot(data=df, x=df.index, y=value_column_name, marker='o')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_filepath)
        print(f"Saved plot to: {output_filepath}")
        plt.close()
    except Exception as e:
        print(f"Error plotting {title}: {e}")

def plot_age_gender(df, output_filepath):
    """Plots the age and gender distribution."""
    if df is None or df.empty:
        print("Cannot plot age/gender: DataFrame is empty or None.")
        return
    try:
        sns.set_theme(style="whitegrid")
        # Ensure data is numeric, summing rows to handle potential NaNs if needed
        df_plot = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        ax = df_plot.plot(kind='bar', stacked=True, figsize=(10, 7))
        
        plt.title('Audience Distribution by Age and Gender (%)')
        plt.xlabel('Age Range')
        plt.ylabel('Percentage of Audience')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Gender')
        plt.tight_layout()
        
        plt.savefig(output_filepath)
        print(f"Saved age/gender plot to: {output_filepath}")
        plt.close()
    except Exception as e:
        print(f"Error plotting age/gender distribution: {e}")

def plot_top_locations(df, location_type, output_filepath):
    """Plots top cities or countries as a horizontal bar chart."""
    if df is None or df.empty:
        print(f"Cannot plot top {location_type}: DataFrame is empty or None.")
        return
    try:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 8))
        
        # Ensure data is numeric
        df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')
        df_plot = df.dropna(subset=['Percentage']).head(15) # Plot top 15
        
        ax = sns.barplot(data=df_plot, y=location_type, x='Percentage', palette="viridis", hue=location_type, legend=False) # Using hue avoids UserWarning 
        
        plt.title(f'Top {location_type} by Audience Percentage')
        plt.xlabel('Percentage of Audience (%)')
        plt.ylabel(location_type)
        plt.tight_layout()
        
        plt.savefig(output_filepath)
        print(f"Saved top {location_type} plot to: {output_filepath}")
        plt.close()
    except Exception as e:
        print(f"Error plotting top {location_type}: {e}")

def plot_post_engagement_distribution(df, output_filepath):
    """Plots the engagement rate distribution by post type."""
    if df is None or df.empty or 'EngagementRate' not in df.columns or 'PostType' not in df.columns:
        print("Cannot plot engagement distribution: DataFrame missing required columns or empty.")
        return
    try:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x='PostType', y='EngagementRate', palette='Set2')
        plt.title('Engagement Rate Distribution by Post Type')
        plt.xlabel('Post Type')
        plt.ylabel('Engagement Rate (%)')
        plt.tight_layout()
        plt.savefig(output_filepath)
        print(f"Saved plot: {output_filepath}")
        plt.close()
    except Exception as e:
        print(f"\nError plotting engagement rate by type: {e}")

# TODO: Add function for other analyses

# --- Summarization Functions --- 

def summarize_timeseries(df, metric_name):
    """Calculates basic summary stats for a time-series DataFrame."""
    if df is None or df.empty:
        return f"No data available for {metric_name}."
    
    summary = []
    total_metric = df[metric_name].sum()
    avg_daily_metric = df[metric_name].mean()
    peak_day = df[metric_name].idxmax()
    peak_value = df[metric_name].max()
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    num_days = len(df)

    summary.append(f"- {metric_name} Summary ({start_date} to {end_date}, {num_days} days):")
    summary.append(f"  - Total: {total_metric:,.0f}")
    summary.append(f"  - Average Daily: {avg_daily_metric:,.2f}")
    summary.append(f"  - Peak Day: {peak_day.strftime('%Y-%m-%d')} ({peak_value:,.0f})")
    
    # Simple trend indicator (comparing first/last half averages - rudimentary)
    try:
        mid_point = df.index[len(df) // 2]
        first_half_avg = df.loc[df.index < mid_point, metric_name].mean()
        second_half_avg = df.loc[df.index >= mid_point, metric_name].mean()
        if second_half_avg > first_half_avg * 1.1: # 10% increase
            summary.append(f"  - Trend: Generally Increasing (Avg {first_half_avg:.2f} -> {second_half_avg:.2f})")
        elif first_half_avg > second_half_avg * 1.1: # 10% decrease
            summary.append(f"  - Trend: Generally Decreasing (Avg {first_half_avg:.2f} -> {second_half_avg:.2f})")
        else:
            summary.append(f"  - Trend: Relatively Stable (Avg {first_half_avg:.2f} -> {second_half_avg:.2f})")
    except Exception:
        summary.append("  - Trend: Could not be calculated.") # Handle cases with few data points
        
    return "\n".join(summary)

def summarize_audience(audience_data):
    """Extracts key highlights from the parsed audience data dictionary."""
    if not audience_data:
        return "No audience data available."
    
    summary = ["- Audience Summary:"]
    
    # Age & Gender
    if 'age_gender' in audience_data and not audience_data['age_gender'].empty:
        df_ag = audience_data['age_gender'].copy()
        df_ag['Total'] = df_ag.sum(axis=1)
        top_age_group = df_ag['Total'].idxmax()
        top_gender_in_group = df_ag.loc[top_age_group, ['Men', 'Women']].idxmax()
        summary.append(f"  - Top Age Group: {top_age_group}")
        summary.append(f"  - Dominant Gender in Top Group: {top_gender_in_group}")
        # Calculate overall gender split
        overall_men = df_ag['Men'].sum()
        overall_women = df_ag['Women'].sum()
        total_audience = overall_men + overall_women
        if total_audience > 0:
             summary.append(f"  - Overall Gender Split: Men {overall_men/total_audience:.1%}, Women {overall_women/total_audience:.1%}")
    else:
        summary.append("  - Age/Gender data not available.")
        
    # Top Countries
    if 'top_countries' in audience_data and not audience_data['top_countries'].empty:
        top_3_countries = audience_data['top_countries']['Country'].head(3).tolist()
        summary.append(f"  - Top 3 Countries: {', '.join(top_3_countries)}")
    else:
        summary.append("  - Top Countries data not available.")
        
    # Top Cities
    if 'top_cities' in audience_data and not audience_data['top_cities'].empty:
        top_3_cities = audience_data['top_cities']['City'].head(3).tolist()
        summary.append(f"  - Top 3 Cities: {', '.join(top_3_cities)}")
    else:
        summary.append("  - Top Cities data not available.")
        
    return "\n".join(summary)

# --- LLM Interaction ---
def call_llm_api(api_url, messages, model=None, api_key=None, temperature=0.7):
    """Calls a generic OpenAI-compatible Chat Completions API using urllib."""
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    data = {
        "messages": messages,
        "temperature": temperature,
    }
    if model: # Only include model if specified (some servers use default)
        data["model"] = model
        
    # Convert data payload to JSON bytes
    json_data = json.dumps(data).encode('utf-8')
    
    # Create the request object
    request = urllib.request.Request(api_url, data=json_data, headers=headers, method='POST')
    
    print(f"Sending request to {api_url}...")
    try:
        with urllib.request.urlopen(request, timeout=180) as response: # Added timeout (3 mins)
            response_body = response.read().decode('utf-8')
            status_code = response.getcode()
            print(f"API Response Status Code: {status_code}")
            
            if 200 <= status_code < 300:
                response_json = json.loads(response_body)
                # Extract content from the first choice's message
                if 'choices' in response_json and response_json['choices']:
                    first_choice = response_json['choices'][0]
                    if 'message' in first_choice and 'content' in first_choice['message']:
                        return first_choice['message']['content'].strip()
                    else:
                        print("Error: API response missing 'message' or 'content' in first choice.")
                        print(f"Response JSON: {response_json}") # Log for debugging
                        return None
                else:
                    print("Error: API response missing 'choices' array or it's empty.")
                    print(f"Response JSON: {response_json}") # Log for debugging
                    return None
            else:
                 print(f"Error: API returned non-success status {status_code}")
                 print(f"Response body: {response_body}") # Log error response body
                 return None

    except urllib.error.URLError as e:
        print(f"Error calling LLM API (URLError): {e}")
        if hasattr(e, 'reason'): print(f"Reason: {e.reason}")
        if hasattr(e, 'code'): print(f"HTTP Code: {e.code}")
        if hasattr(e, 'read'): # Try reading error body if available
             try: print(f"Error Body: {e.read().decode()}")
             except: pass
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding API JSON response: {e}")
        print(f"Raw response: {response_body}") # Log raw response on JSON error
        return None
    except Exception as e:
        print(f"An unexpected error occurred during LLM API call: {e}")
        return None

def perform_llm_analysis(config, data_dir, output_dir, post_file_arg):
    """Loads data, summarizes, uses config context, calls LLM, prints response."""
    if not config:
        print("Error: LLM analysis requires a valid configuration.")
        return
        
    print("--- Performing LLM Analysis --- ")
    
    # --- Load and Summarize Data --- 
    print("Loading and summarizing data...")
    reach_df = load_timeseries_csv(os.path.join(data_dir, 'Reach.csv'), 'Reach')
    follows_df = load_timeseries_csv(os.path.join(data_dir, 'Follows.csv'), 'Follows')
    visits_df = load_timeseries_csv(os.path.join(data_dir, 'Visits.csv'), 'Visits')
    views_df = load_timeseries_csv(os.path.join(data_dir, 'Views.csv'), 'Views')
    audience_data = load_audience_data(os.path.join(data_dir, 'Audience.csv'))
    
    reach_summary = summarize_timeseries(reach_df, 'Reach')
    follows_summary = summarize_timeseries(follows_df, 'Follows')
    visits_summary = summarize_timeseries(visits_df, 'Visits')
    views_summary = summarize_timeseries(views_df, 'Views')
    audience_summary = summarize_audience(audience_data)

    # Load and analyze post performance
    post_summary_text = "Post performance data could not be loaded or analyzed."
    post_perf_file_name = post_file_arg 
    if not post_perf_file_name:
         try:
            for f in os.listdir(data_dir):
                if f.endswith('.csv') and re.match(r'^[A-Za-z]{3}-\d{2}-\d{4}_', f):
                    post_perf_file_name = f
                    break
         except Exception:
             pass
    if post_perf_file_name:
        post_perf_file_path = os.path.join(data_dir, post_perf_file_name)
        posts_df_raw = load_post_performance_data(post_perf_file_path)
        if posts_df_raw is not None:
            _, summary = analyze_and_summarize_post_performance(posts_df_raw)
            if summary: post_summary_text = summary
    
    # --- Get Context from Config --- 
    user_context = config.get('user_context', {}) 
    page_niche = user_context.get('page_niche', '<<Niche not specified in config.json>>')
    page_goals = user_context.get('page_goals', '<<Goals not specified in config.json>>')
    target_audience_desc = user_context.get('target_audience_desc', '<<Target audience not specified in config.json>>')
    typical_content = user_context.get('typical_content', '<<Typical content not specified>>')
    recent_strategy = user_context.get('recent_strategy', '')
    
    print("\n--- Using Context from Config --- ")
    print(f"- Niche: {page_niche}")
    print(f"- Goals: {page_goals}")
    print(f"- Target Audience: {target_audience_desc}")
    print(f"- Typical Content: {typical_content}")
    print(f"- Recent Strategy: {recent_strategy if recent_strategy else 'None specified'}")

    # --- Format Prompt Messages --- 
    system_message = """
    You are an expert Instagram growth strategist. Analyze the provided data summary and user context for an Instagram page. 
    Provide specific, actionable insights and recommendations based *only* on the information given. 
    Focus on interpreting trends, audience alignment, and post performance to suggest tailored content ideas and strategy adjustments. 
    Avoid generic advice. Justify your suggestions.
    Structure your response clearly, perhaps with sections for Observations, Recommendations, and Opportunities.
    """ 

    user_message_content = f"""
**Context:**
- Page Niche: {page_niche}
- Primary Goals: {page_goals}
- Target Audience Description: {target_audience_desc}
- Typical Content: {typical_content}
- Recent Strategy/Changes: {recent_strategy if recent_strategy else 'None specified'}

**Data Summary:**
{reach_summary}
{follows_summary}
{visits_summary}
{views_summary}
{audience_summary}
{post_summary_text}
"""

    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message_content.strip()}
    ]

    # --- Call LLM API --- 
    print("\nCalling LLM API...")
    llm_settings = config.get('llm_settings', {})
    api_response_content = call_llm_api(
        api_url=llm_settings.get('api_url', 'http://localhost:1234/v1/chat/completions'), 
        messages=messages, 
        model=llm_settings.get('model'), # Can be None
        api_key=llm_settings.get('api_key') # Can be None
    )

    # --- Print Response --- 
    print("\n" + "="*40)
    print("--- LLM Analysis Result --- ")
    print("="*40)
    if api_response_content:
        print(api_response_content)
    else:
        print("Failed to get a response from the LLM API.")
    print("-"*40)

# --- Post Performance Analysis ---

def analyze_and_summarize_post_performance(df):
    """Analyzes the detailed post performance DataFrame and generates a text summary."""
    if df is None or df.empty:
        print("Cannot analyze post performance: DataFrame is empty or None.")
        return df, None # Return original df and None summary

    print("\n--- Analyzing Post Performance --- (Based on provided file)")
    analysis_summary = []
    df_analyzed = df.copy() # Work on a copy

    # Ensure necessary columns exist
    required_cols = ['Description', 'PostType', 'Reach', 'Likes', 'Comments', 'Saves']
    if not all(col in df_analyzed.columns for col in required_cols):
        print(f"Error: Missing one or more required columns for analysis: {required_cols}")
        return df, None # Return original df and None summary

    # 1. Calculate Engagement Rate (ER = (Likes+Comments+Saves) / Reach * 100)
    df_analyzed['EngagementRate'] = 0.0
    df_analyzed.loc[df_analyzed['Reach'] > 0, 'EngagementRate'] = \
        (df_analyzed['Likes'] + df_analyzed['Comments'] + df_analyzed['Saves'] ) / df_analyzed['Reach'] * 100

    # 2. Calculate Caption Length and Hashtag Count
    df_analyzed['CaptionLength'] = df_analyzed['Description'].fillna('').apply(len)
    df_analyzed['HashtagCount'] = df_analyzed['Description'].fillna('').apply(lambda x: len(re.findall(r"#(\w+)", x)))

    # --- Summaries & Insights --- 
    analysis_summary.append("- Overall Performance Summary:")
    analysis_summary.append(f"  - Total Posts Analyzed: {len(df_analyzed)}")
    analysis_summary.append(f"  - Average Reach: {df_analyzed['Reach'].mean():.2f}")
    analysis_summary.append(f"  - Average Likes: {df_analyzed['Likes'].mean():.2f}")
    analysis_summary.append(f"  - Average Comments: {df_analyzed['Comments'].mean():.2f}")
    analysis_summary.append(f"  - Average Saves: {df_analyzed['Saves'].mean():.2f}")
    analysis_summary.append(f"  - Average Engagement Rate: {df_analyzed['EngagementRate'].mean():.2f}%")

    # 3. Top Performing Posts
    top_n = 5 
    top_by_reach = df_analyzed.nlargest(top_n, 'Reach')
    top_by_likes = df_analyzed.nlargest(top_n, 'Likes')
    top_by_engagement_rate = df_analyzed.nlargest(top_n, 'EngagementRate')

    analysis_summary.append(f"\n- Top {top_n} Posts by Reach:")
    for idx, row in top_by_reach.iterrows():
        pub_time = row['PublishTime'].strftime('%m-%d %H:%M') if pd.notna(row['PublishTime']) else 'N/A'
        analysis_summary.append(f"  - {pub_time} ({row['PostType']}): {row['Reach']} reach")
        # Optionally add Permalink: row['Permalink']

    analysis_summary.append(f"\n- Top {top_n} Posts by Engagement Rate:")
    for idx, row in top_by_engagement_rate.iterrows():
        pub_time = row['PublishTime'].strftime('%m-%d %H:%M') if pd.notna(row['PublishTime']) else 'N/A'
        analysis_summary.append(f"  - {pub_time} ({row['PostType']}): {row['EngagementRate']:.2f}% ER")

    # 4. Performance by Post Type
    analysis_summary.append("\n- Average Performance by Post Type:")
    avg_by_type = df_analyzed.groupby('PostType')[[ 'Reach', 'Likes', 'Comments', 'Saves', 'EngagementRate']].mean().round(2)
    analysis_summary.append(avg_by_type.to_string()) # Convert DataFrame to string for printing

    # Print the summary string
    full_summary_text = "\n".join(analysis_summary)
    print(full_summary_text)
    
    return df_analyzed, full_summary_text # Return ANALYZED df and summary text

# --- HTML Report Generation ---
def generate_html_report(plot_files, output_html_path):
    """Generates an HTML report embedding the specified plot images."""
    print(f"\n--- Generating HTML Report: {output_html_path} ---")
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang=\"en\">\n")
            f.write("<head>\n")
            f.write("    <meta charset=\"UTF-8\">\n")
            f.write("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
            f.write("    <title>Instagram Dashboard Analysis Report</title>\n")
            f.write("    <style>\n")
            f.write("        body { font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: auto; background-color: #f4f4f4; color: #333; }\n")
            f.write("        h1 { color: #d62976; text-align: center; margin-bottom: 30px; }\n")
            f.write("        .plot-container { background-color: #fff; margin-bottom: 25px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }\n")
            f.write("        .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }\n")
            f.write("        h2 { color: #405de6; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px; }\n")
            f.write("        p.error { color: #c00; font-style: italic; }\n")
            f.write("    </style>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            f.write("    <h1>Instagram Dashboard Analysis Report</h1>\n")

            for title, plot_path in plot_files.items():
                f.write(f'    <div class="plot-container">\n')
                f.write(f'        <h2>{title}</h2>\n')
                if plot_path and os.path.exists(plot_path):
                    # Use relative path for embedding in HTML
                    relative_plot_path = os.path.relpath(plot_path, start=os.path.dirname(output_html_path))
                    f.write(f'        <img src="{relative_plot_path}" alt="{title} Plot">\n')
                else:
                    f.write(f'        <p class="error">Plot not generated or file not found: {plot_path}</p>\n')
                f.write("    </div>\n")

            f.write("</body>\n")
            f.write("</html>\n")
        print(f"Successfully generated HTML report: {output_html_path}")
        
        # Attempt to open the report
        try:
            webbrowser.open(f'file://{os.path.abspath(output_html_path)}')
        except Exception as e:
            print(f"Could not automatically open report in browser: {e}")
            
    except IOError:
        print(f"\nError: Could not write HTML report to {output_html_path}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while generating HTML report: {e}")

def generate_full_report(data_dir, output_dir, post_file_arg):
    """Loads all data, generates all plots, and creates an HTML report."""
    print("--- Generating Full Analysis Report --- ")
    plot_files = {} # Dictionary to store plot titles and paths

    # --- Load and Plot Time Series Data ---
    timeseries_files = {
        'Reach': 'Reach.csv',
        'Follows': 'Follows.csv',
        'Visits': 'Visits.csv',
        'Views': 'Views.csv'
        # Add Interactions, Link Clicks etc. here later
    }
    for metric, filename in timeseries_files.items():
        filepath = os.path.join(data_dir, filename)
        data_df = load_timeseries_csv(filepath, metric)
        if data_df is not None:
            plot_filepath = os.path.join(output_dir, f'{metric.lower()}_trend.png')
            plot_title = f'Instagram {metric} Over Time'
            ylabel = f'Daily {metric}'
            plot_timeseries_trend(data_df, metric, plot_title, ylabel, plot_filepath)
            plot_files[plot_title] = plot_filepath
        else:
            plot_files[f'Instagram {metric} Over Time'] = None # Mark as failed

    # --- Load and Plot Audience Data ---
    audience_file = os.path.join(data_dir, 'Audience.csv')
    audience_data = load_audience_data(audience_file)
    if audience_data:
        if 'age_gender' in audience_data:
            plot_filepath = os.path.join(output_dir, 'audience_age_gender.png')
            plot_title = 'Audience Distribution by Age and Gender (%)'
            plot_age_gender(audience_data['age_gender'], plot_filepath)
            plot_files[plot_title] = plot_filepath
        else: plot_files['Audience Distribution by Age and Gender (%)'] = None
        
        if 'top_cities' in audience_data:
            plot_filepath = os.path.join(output_dir, 'audience_top_cities.png')
            plot_title = 'Top Cities by Audience Percentage'
            plot_top_locations(audience_data['top_cities'], 'City', plot_filepath)
            plot_files[plot_title] = plot_filepath
        else: plot_files['Top Cities by Audience Percentage'] = None

        if 'top_countries' in audience_data:
            plot_filepath = os.path.join(output_dir, 'audience_top_countries.png')
            plot_title = 'Top Countries by Audience Percentage'
            plot_top_locations(audience_data['top_countries'], 'Country', plot_filepath)
            plot_files[plot_title] = plot_filepath
        else: plot_files['Top Countries by Audience Percentage'] = None
    else:
        print("Skipping audience plots due to loading error.")
        plot_files['Audience Distribution by Age and Gender (%)'] = None
        plot_files['Top Cities by Audience Percentage'] = None
        plot_files['Top Countries by Audience Percentage'] = None

    # --- Load, Analyze, and Plot Post Performance Data ---
    post_perf_file_name = post_file_arg # Use the value passed from args
    if not post_perf_file_name:
         # Try to find automatically ONLY if not provided
         try:
            for f in os.listdir(data_dir):
                if f.endswith('.csv') and re.match(r'^[A-Za-z]{3}-\d{2}-\d{4}_', f):
                    post_perf_file_name = f
                    break
         except Exception: pass # Ignore errors finding it here

    if post_perf_file_name:
        post_perf_file_path = os.path.join(data_dir, post_perf_file_name)
        posts_df_raw = load_post_performance_data(post_perf_file_path)
        if posts_df_raw is not None:
            posts_df_analyzed, _ = analyze_and_summarize_post_performance(posts_df_raw)
            if posts_df_analyzed is not None:
                 plot_filepath = os.path.join(output_dir, 'engagement_rate_by_type.png')
                 plot_title = 'Engagement Rate Distribution by Post Type'
                 plot_post_engagement_distribution(posts_df_analyzed, plot_filepath)
                 plot_files[plot_title] = plot_filepath
            else: plot_files['Engagement Rate Distribution by Post Type'] = None
        else: plot_files['Engagement Rate Distribution by Post Type'] = None
    else:
        print("Skipping post performance plot - detailed file not found or specified.")
        plot_files['Engagement Rate Distribution by Post Type'] = None

    # --- Generate the HTML --- 
    report_html_path = os.path.join(output_dir, 'dashboard_report.html')
    generate_html_report(plot_files, report_html_path)

    print("--- Full Report Generation Complete ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Instagram Dashboard Export data (CSV files).")
    parser.add_argument(
        "task",
        choices=['plot_reach', 'analyze_audience', 'plot_follows', 
                 'analyze_posts', 'plot_visits', 'plot_views', 
                 'generate_report', 'llm_analyze'], 
        help="The analysis task to perform."
    )
    parser.add_argument(
        '--data-dir', default='dashboard_export',
        help='Directory containing the dashboard export CSV files'
    )
    parser.add_argument(
        '--output-dir', default='dashboard_plots',
        help='Directory to save generated plots'
    )
    parser.add_argument(
        '--post-file', default=None, 
        help='Filename of the detailed post performance CSV (optional, will try to auto-detect)'
    )
    # Add config file argument
    parser.add_argument(
        '--config', 
        default='config.json', 
        help='Path to the JSON configuration file (default: config.json)'
    )
    # LLM arguments (--llm-url, etc.) are removed

    args = parser.parse_args()

    # Load config early if needed by the task
    config = None
    if args.task == 'llm_analyze':
        config = load_config(args.config)
        if config is None:
            print("Exiting due to config loading failure.")
            exit() # Exit if config loading failed for required task

    # Find post file if needed
    post_file_needed = args.task in ['analyze_posts', 'generate_report', 'llm_analyze']
    if post_file_needed and not args.post_file:
        try:
            for f in os.listdir(args.data_dir):
                if f.endswith('.csv') and re.match(r'^[A-Za-z]{3}-\d{2}-\d{4}_', f):
                    args.post_file = f
                    print(f"Automatically detected post performance file: {args.post_file}")
                    break
            if args.task == 'analyze_posts' and not args.post_file:
                 print("Error: Could not automatically find the post performance CSV file for 'analyze_posts'. Please specify it using --post-file.")
                 exit()
        except FileNotFoundError:
             print(f"Error: Data directory '{args.data_dir}' not found.")
             if args.task == 'analyze_posts': exit() 
        except Exception as e:
             print(f"Error finding post performance file: {e}")
             if args.task == 'analyze_posts': exit() 

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Execute requested task ---
    if args.task == 'plot_reach':
        reach_file = os.path.join(args.data_dir, 'Reach.csv')
        print(f"Attempting to plot reach from {reach_file}...")
        data_df = load_timeseries_csv(reach_file, 'Reach')
        if data_df is not None:
            plot_filepath = os.path.join(args.output_dir, 'reach_trend.png')
            plot_timeseries_trend(data_df, 'Reach', 'Instagram Reach Over Time', 'Daily Reach', plot_filepath)
        else: print("Skipping plot due to data loading errors.")
        print("Reach plotting complete.")
    elif args.task == 'analyze_audience':
        audience_file = os.path.join(args.data_dir, 'Audience.csv')
        print(f"Attempting to analyze audience from {audience_file}...")
        audience_data = load_audience_data(audience_file)
        if audience_data:
            if 'age_gender' in audience_data:
                plot_filepath = os.path.join(args.output_dir, 'audience_age_gender.png')
                plot_age_gender(audience_data['age_gender'], plot_filepath)
            if 'top_cities' in audience_data:
                plot_filepath = os.path.join(args.output_dir, 'audience_top_cities.png')
                plot_top_locations(audience_data['top_cities'], 'City', plot_filepath)
            if 'top_countries' in audience_data:
                plot_filepath = os.path.join(args.output_dir, 'audience_top_countries.png')
                plot_top_locations(audience_data['top_countries'], 'Country', plot_filepath)
        else: print("Could not load audience data for analysis.")
        print("Audience analysis complete.")
    elif args.task == 'plot_follows':
        follows_file = os.path.join(args.data_dir, 'Follows.csv')
        print(f"Attempting to plot follows from {follows_file}...")
        data_df = load_timeseries_csv(follows_file, 'Follows')
        if data_df is not None:
            plot_filepath = os.path.join(args.output_dir, 'follows_trend.png')
            plot_timeseries_trend(data_df, 'Follows', 'Instagram Daily Follows Over Time', 'Daily Follows', plot_filepath)
        else: print("Skipping plot due to data loading errors.")
        print("Follows plotting complete.")
    elif args.task == 'plot_visits':
        visits_file = os.path.join(args.data_dir, 'Visits.csv')
        print(f"Attempting to plot visits from {visits_file}...")
        data_df = load_timeseries_csv(visits_file, 'Visits')
        if data_df is not None:
            plot_filepath = os.path.join(args.output_dir, 'visits_trend.png')
            plot_timeseries_trend(data_df, 'Visits', 'Instagram Profile Visits Over Time', 'Daily Visits', plot_filepath)
        else: print("Skipping plot due to data loading errors.")
        print("Visits plotting complete.")
    elif args.task == 'plot_views':
        views_file = os.path.join(args.data_dir, 'Views.csv')
        print(f"Attempting to plot views from {views_file}...")
        data_df = load_timeseries_csv(views_file, 'Views')
        if data_df is not None:
            plot_filepath = os.path.join(args.output_dir, 'views_trend.png')
            plot_timeseries_trend(data_df, 'Views', 'Instagram Content Views Over Time', 'Daily Views', plot_filepath)
        else: print("Skipping plot due to data loading errors.")
        print("Views plotting complete.")
    elif args.task == 'analyze_posts':
        if not args.post_file:
            print("Error: Post performance file needed for 'analyze_posts' but not found or specified.")
        else:
            post_perf_file = os.path.join(args.data_dir, args.post_file)
            print(f"Attempting to analyze post performance from {post_perf_file}...")
            posts_df_raw = load_post_performance_data(post_perf_file)
            if posts_df_raw is not None:
                posts_df_analyzed, post_summary = analyze_and_summarize_post_performance(posts_df_raw)
                if posts_df_analyzed is not None:
                    plot_filepath = os.path.join(args.output_dir, 'engagement_rate_by_type.png')
                    plot_post_engagement_distribution(posts_df_analyzed, plot_filepath)
            else:
                print("Skipping analysis due to data loading errors.")
            print("Post performance analysis complete.")
    elif args.task == 'generate_report':
         generate_full_report(args.data_dir, args.output_dir, args.post_file)
    
    elif args.task == 'llm_analyze':
         # Config should have been loaded earlier
         perform_llm_analysis(config, args.data_dir, args.output_dir, args.post_file)

    else:
        print(f"Error: Unknown task '{args.task}'")
        parser.print_help()

    print("\nDashboard analysis script finished.") 