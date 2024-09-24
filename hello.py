import streamlit as st
import pandas as pd
import os
import base64
from datetime import datetime
import time
import csv
from ai import instruct
import streamlit.components.v1 as components

# ------------------------------
# Configuration and Constants
# ------------------------------

# Flag to determine if the app is in internal (development) mode or external (viewing) mode
IS_INTERNAL = True  # Set to True for internal use, False for external users
PROMPT = f"""Generate a complex single file HTML web application for: {{selected_idea}}. 
It must be a single page application with no redirects.
Provide the full code for the prompt with no placeholders.
Ensure to delimit the file with <!DOCTYPE html>.
Make it as complex and impressive as you are capable of handling.
Blow away the user with your advanced capabilities."""
# Define available AI models
MODELS = [
    'openai/gpt-4o',
    'google/gemini-pro-1.5-exp',
    'qwen/qwen-2.5-72b-instruct',
    'anthropic/claude-3.5-sonnet',
    'openai/o1-mini-2024-09-12',
    'meta-llama/llama-3.1-405b-instruct',
    "deepseek/deepseek-chat"
]

# Define available Prompts
WEB_APP_IDEAS = [
    'Monetizable and original SaaS',
    'Interactive Particle Physics Simulator',
    'Advanced Fractal Explorer with Custom Algorithms',
    'Complex 3D Graphing Calculator',
    'Procedural City Generator with Customizable Parameters',
    'In-Browser Neural Network Trainer and Visualizer',
    'Advanced Cellular Automaton with Multiple Rulesets',
    'Interactive Fourier Transform Visualizer and Synthesizer',
    'Realistic Fluid Dynamics Simulator',
    'Complex Pathfinding Algorithm Visualizer',
    'Interactive Mandelbrot Set Explorer with Deep Zoom',
    'Procedural Terrain Generator with Erosion Simulation',
    'Advanced Sorting Algorithm Visualizer with Performance Metrics',
    'Interactive Lindenmayer System (L-System) Fractal Generator'
]

# Define feedback options
FEEDBACK_OPTIONS = ["thumbs", "faces", "stars"]

# Directory to save generated HTML files
GENERATED_APPS_DIR = 'generated_apps'
os.makedirs(GENERATED_APPS_DIR, exist_ok=True)

# Files to store benchmark results and feedback
BENCHMARK_FILE = 'benchmark_results.csv'
FEEDBACK_FILE = 'feedback_results.csv'

# Initialize the benchmark CSV file if it doesn't exist
if not os.path.isfile(BENCHMARK_FILE):
    with open(BENCHMARK_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'AI Model', 'Prompt', 'Temperature', 'Time Taken (s)', 'HTML File'])

# Initialize the feedback CSV file if it doesn't exist
if not os.path.isfile(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'AI Model', 'Prompt', 'HTML File', 'Rating'])

# ------------------------------
# Helper Functions
# ------------------------------

def get_download_link(file_path, file_name):
    """
    Generates a download link for the given file.

    Parameters:
    - file_path (str): Path to the file.
    - file_name (str): Name to display for the download link.

    Returns:
    - str: HTML anchor tag with the download link.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()  # Encode to base64
        href = f'<a href="data:text/html;base64,{b64}" download="{file_name}">Download</a>'
        return href
    except Exception as e:
        return f"Error generating download link: {e}"

def log_benchmark(date, model, idea, temperature, time_taken, file_name):
    """
    Logs the benchmark results to a CSV file.

    Parameters:
    - date (str): The date of the benchmark.
    - model (str): The AI model used.
    - idea (str): The Prompt.
    - temperature (float): Temperature used for generation.
    - time_taken (float): Time taken for the instruct call.
    - file_name (str): The name of the generated HTML file.
    """
    with open(BENCHMARK_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([date, model, idea, temperature, f"{time_taken:.2f}", file_name])

def log_feedback(date, model, idea, file_name, rating):
    """
    Logs the user feedback to a CSV file.

    Parameters:
    - date (str): The date of the feedback.
    - model (str): The AI model used.
    - idea (str): The Prompt.
    - file_name (str): The name of the HTML file.
    - rating (str): The rating provided by the user.
    """
    with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([date, model, idea, file_name, rating])

def load_benchmark_results():
    """
    Loads the benchmark results from the CSV file into a DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing benchmark results.
    """
    return pd.read_csv(BENCHMARK_FILE)

def load_feedback_results():
    """
    Loads the feedback results from the CSV file into a DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing feedback results.
    """
    if os.path.isfile(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame()

# Add this function to render HTML files
def render_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=1000, scrolling=False)

# ------------------------------
# Streamlit App Configuration
# ------------------------------

st.set_page_config(
    page_title="Web App LLM Benchmark",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
if 'view_state' not in st.session_state:
    st.session_state.view_state = 'table'
if 'selected_result' not in st.session_state:
    st.session_state.selected_result = None

# ------------------------------
# App Layout
# ------------------------------

st.title("AI-Powered Web App Benchmark Generator 📊")

st.markdown("""
Welcome to the **AI Web App Benchmark Generator**! This application is an LLM benchmark for different AI models by generating single .html file based web applications based on a series of selected ideas. 
""")

st.markdown(f"""*Prompt*: {PROMPT}""")

# Sidebar selections
if IS_INTERNAL:
    st.sidebar.subheader("Run Benchmark")
    selected_idea = st.sidebar.selectbox("Select Prompt", WEB_APP_IDEAS)
    temperature = st.sidebar.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    run_benchmark_button = st.sidebar.button("Run Benchmark")
else:
    selected_idea = None
    temperature = None
    run_benchmark_button = False

# Feedback configuration (available to all users)
#feedback_option = st.sidebar.selectbox("Select Feedback Style", FEEDBACK_OPTIONS, index=0)

# Run Benchmark Button (only visible to internal users)
if IS_INTERNAL and run_benchmark_button:
    with st.spinner("Generating web apps across all models..."):
        for model in MODELS:
            start_time = time.time()
            # Create the prompt with <html> tags as per instructions
            prompt = f"""Generate a complex single file HTML web application for a {selected_idea}. 
It must be a single page application with no redirects.
Provide the full code for the prompt with no placeholders.
Ensure to delimit the file with <!DOCTYPE html>.
Make it as complex and impressive as you are capable of handling.
Blow away the user with your advanced capabilities."""
            try:
                # Call the instruct method from the ai module with the specified temperature
                generated_html = instruct(prompt, model=model, temperature=temperature)
            except Exception as e:
                st.error(f"An error occurred while generating the web app with model {model}: {e}")
                continue  # Skip to the next model

            end_time = time.time()
            time_taken = end_time - start_time
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if the generated HTML is wrapped in ```html and ``` tags
            html_start = generated_html.find('```html')
            html_end = generated_html.rfind('```')
            
            if html_start != -1 and html_end != -1 and html_end > html_start:
                # Extract the HTML content from within the code block
                generated_html = generated_html[html_start + 7:html_end].strip()
            else:
                # If not in code block, check for <!DOCTYPE html>
                doctype_index = generated_html.find('<!DOCTYPE html>')
                if doctype_index != -1:
                    generated_html = generated_html[doctype_index:]
                else:
                    # If <!DOCTYPE html> is not found, wrap the content with proper HTML structure
                    generated_html = f"""<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>{selected_idea}</title>
                </head>
                <body>
                {generated_html}
                </body>
                </html>
                """

            # Define the file name and path
            safe_idea = selected_idea.replace(' ', '_').lower()
            safe_model = model.replace('/', '_').lower()
            file_name = f"{safe_idea}_{safe_model}.html"
            file_path = os.path.join(GENERATED_APPS_DIR, file_name)

            # Save the generated HTML to the file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_html)
            except Exception as e:
                st.error(f"Failed to save the HTML file for model {model}: {e}")
                continue  # Skip to the next model

            # Log the benchmark results
            log_benchmark(current_date, model, selected_idea, temperature, time_taken, file_name)

        st.success(f"Benchmark completed for '{selected_idea}' across all models!")

# ------------------------------
# Display Benchmark Results
# ------------------------------

st.markdown("---")
st.subheader("Benchmark Results")

# Load existing benchmark results
benchmark_df = load_benchmark_results()

if not benchmark_df.empty:
    if st.session_state.view_state == 'table':
        # Generate download links for HTML files
        benchmark_df['Download Link'] = benchmark_df.apply(
            lambda row: get_download_link(os.path.join(GENERATED_APPS_DIR, row['HTML File']), row['HTML File']),
            axis=1
        )

        # Add feedback information to the DataFrame
        feedback_df = load_feedback_results()
        if not feedback_df.empty:
            feedback_summary = feedback_df.groupby('HTML File').agg({
                'Rating': lambda x: f"👍 {sum(x == 'Thumbs Up')} | 👎 {sum(x == 'Thumbs Down')}"
            })
            benchmark_df = benchmark_df.merge(feedback_summary, on='HTML File', how='left')
        else:
            benchmark_df['Rating'] = 'No feedback yet'

        # Define columns
        columns = ['Date', 'AI Model', 'Prompt', 'Temperature', 'Time Taken (s)', 'Download Link', 'Rating', 'View']
        
        # Create column headers with filters
        header_cols = st.columns(len(columns))
        for col, column_name in zip(header_cols, columns):
            with col:
                
                st.write(f"**{column_name}**")
                if "Download" not in column_name and "View" not in column_name and "Rating" not in column_name:

                    with st.popover(f"Filter {column_name}"):
                        if column_name == 'Date':
                            # Convert string dates to datetime objects
                            benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
                            min_date = benchmark_df['Date'].min().date()
                            max_date = benchmark_df['Date'].max().date()
                            
                            start_date = st.date_input('From', value=min_date, min_value=min_date, max_value=max_date)
                            end_date = st.date_input('To', value=max_date, min_value=min_date, max_value=max_date)
                            
                            benchmark_df = benchmark_df[(benchmark_df['Date'].dt.date >= start_date) & 
                                                        (benchmark_df['Date'].dt.date <= end_date)]
                        elif column_name == 'AI Model':
                            models = st.multiselect('Select Models', options=benchmark_df['AI Model'].unique())
                            if models:
                                benchmark_df = benchmark_df[benchmark_df['AI Model'].isin(models)]
                        elif column_name == 'Prompt':
                            ideas = st.multiselect('Select Ideas', options=benchmark_df['Prompt'].unique())
                            if ideas:
                                benchmark_df = benchmark_df[benchmark_df['Prompt'].isin(ideas)]
                        elif column_name == 'Temperature':
                            min_temp = st.number_input('Min Temperature', value=benchmark_df['Temperature'].min(), step=0.1)
                            max_temp = st.number_input('Max Temperature', value=benchmark_df['Temperature'].max(), step=0.1)
                            benchmark_df = benchmark_df[(benchmark_df['Temperature'] >= min_temp) & 
                                                        (benchmark_df['Temperature'] <= max_temp)]
                        elif column_name == 'Time Taken (s)':
                            min_time = st.number_input('Min Time (s)', value=benchmark_df['Time Taken (s)'].min(), step=0.1)
                            max_time = st.number_input('Max Time (s)', value=benchmark_df['Time Taken (s)'].max(), step=0.1)
                            benchmark_df = benchmark_df[(benchmark_df['Time Taken (s)'] >= min_time) & 
                                                        (benchmark_df['Time Taken (s)'] <= max_time)]
                        elif column_name == 'Rating':
                            ratings = st.multiselect('Select Ratings', options=['Thumbs Up', 'Thumbs Down'])
                            if ratings:
                                benchmark_df = benchmark_df[benchmark_df['Rating'].apply(lambda x: any(rating in x for rating in ratings))]

        # Display data rows
        for idx, row in benchmark_df.iterrows():
            cols = st.columns(len(columns))
            cols[0].write(row['Date'])
            cols[1].write(row['AI Model'])
            cols[2].write(row['Prompt'])
            cols[3].write(f"{row['Temperature']:.2f}")
            cols[4].write(f"{row['Time Taken (s)']:.2f}")
            cols[5].markdown(row['Download Link'], unsafe_allow_html=True)
            cols[6].write(row['Rating'])
            if cols[7].button("View", key=f"view_button_{idx}"):
                st.session_state.view_state = 'app'
                st.session_state.selected_result = idx
                st.rerun()

    elif st.session_state.view_state == 'app':
        # Display back button
        if st.button("← Back to Results"):
            st.session_state.view_state = 'table'
            st.session_state.selected_result = None
            st.rerun()

        # Display the selected app
        if st.session_state.selected_result is not None:
            row = benchmark_df.iloc[st.session_state.selected_result]
            st.subheader(f"Viewing: {row['AI Model']} - {row['Prompt']}")
            file_path = os.path.join(GENERATED_APPS_DIR, row['HTML File'])
            render_html_file(file_path)
            
            # Feedback section
            st.write("### Provide Feedback")
            rating = st.radio("Do you like this web app?", ["👍 Thumbs Up", "👎 Thumbs Down"], key=f"rating_{st.session_state.selected_result}")
            if st.button("Submit Feedback", key=f"submit_{st.session_state.selected_result}"):
                log_feedback(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                             row['AI Model'], 
                             row['Prompt'], 
                             row['HTML File'], 
                             "Thumbs Up" if rating == "👍 Thumbs Up" else "Thumbs Down")
                st.success("Thank you for your feedback!")
                st.rerun()

else:
    st.info("No benchmark results available yet. Internal users can run a benchmark from the sidebar!")

# ------------------------------
# Display Feedback Results (Optional)
# ------------------------------

# Only show aggregated feedback in table view
if st.session_state.view_state == 'table':
    st.markdown("---")
    st.subheader("Aggregated Feedback")

    feedback_df = load_feedback_results()

    if not feedback_df.empty:
        # Aggregate feedback
        aggregated_feedback = feedback_df.groupby(['AI Model', 'Prompt', 'HTML File'])['Rating'].value_counts().unstack().fillna(0)
        st.dataframe(aggregated_feedback)
    else:
        st.info("No feedback has been submitted yet.")
