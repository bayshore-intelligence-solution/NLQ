import streamlit as st
import pandas as pd
import google.generativeai as genai
import altair as alt # Import Altair
import io
import traceback
import re
import json # For parsing chart config

# --- Configuration ---
st.set_page_config(page_title="CSV Chatbot with Gemini", layout="wide")
st.title("📊 Chat with your CSV Data (Advanced Charts)")
st.caption("Upload a CSV, ask questions, and get back dataframes and interactive Altair charts.")

# --- Gemini API Key ---
# Use st.secrets for deployment, otherwise use sidebar input
try:
    api_key = "AIzaSyCEN2XmokVA7It9CuWECaMYRx_puAloe6w" #st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.sidebar.warning("Google API Key not found in secrets. Please enter it below.")
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

if not api_key:
    st.info("Please provide your Google API Key in the sidebar to continue.")
    st.stop()

# Configure the Gemini API
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Or use 'gemini-pro'
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- Function Definitions ---

# def get_df_info(df):
#     """Gets schema information from the dataframe."""
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     info = buffer.getvalue()
#     # Also include column names and head for better context
#     return f"DataFrame Info:\n{info}\n\nColumn Names: {df.columns.tolist()}\n\nFirst 5 rows:\n{df.head().to_string()}"
def get_df_info(df):
    """Gets schema information and unique values of categorical columns from the dataframe."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    categorical_info = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[col].unique().tolist()
        # Limit the number of unique values displayed to avoid overwhelming the prompt
        if len(unique_values) > 20:
            categorical_info[col] = f"Too many unique values (> 20), showing first 20: {unique_values[:20]}"
        else:
            categorical_info[col] = unique_values

    categorical_str = "\nCategorical Column Unique Values:\n" + json.dumps(categorical_info, indent=2)

    return f"DataFrame Info:\n{info}\n\nColumn Names: {df.columns.tolist()}\nData Description {df.describe()}\n\nFirst 5 rows:\n{df.head().to_string()}\n{categorical_str}"

def generate_gemini_code(query, df_info):
    """Generates Python code and chart config using Gemini."""
    prompt = f"""
    You are a data analysis expert specializing in Python Pandas and Altair visualization.
    Your task is to generate Python code to answer a user's query based on a pandas DataFrame and suggest an appropriate Altair chart configuration.

    The DataFrame is already loaded into a variable named `df`.

    DataFrame Information:
    {df_info}

    User Query: "{query}"

    Instructions:
    1.  Generate Python code using the `df` variable to answer the query.
    2.  The code MUST produce a resulting pandas DataFrame assigned to a variable named `result_df`.
    3.  **Crucially**: Include a JSON configuration object within a comment `# chart_config: <json_object>` on a new line after the pandas code. This JSON will define an Altair chart based on `result_df`.
    4.  The JSON object should have the following structure:
        * `"type"`: (string) Chart type. Supported: `"bar"`, `"line"`, `"scatter"`, `"area"`, `"table"`, `"combined"`, `"pie"`.
        * `"x"`: (string, optional) Column name for the X-axis.
        * `"y"`: (string or list of strings, optional) Column name(s) for the Y-axis.
        * `"color"`: (string, optional) Column name to use for color encoding (hue/grouping).
        * `"tooltip"`: (list of strings, optional) Columns to show in tooltips. If omitted, default Altair tooltips used.
        * `"layers"`: (list of objects, REQUIRED if type is "combined") Defines layers for combined charts. Each object in the list follows the same structure as the main config (e.g., `{{"type": "bar", "x": "colA", "y": "colB"}}, {{"type": "line", "x": "colA", "y": "colC"}}`). Color can be defined at the top level or per layer.
    5.  Choose the chart type and columns (`x`, `y`, `color`) that best visualize the answer to the user's query based on the `result_df`.
    6.  Use `"table"` if no other chart type is suitable or if the result is best shown as a table. Omit x, y, color for tables.
    7.  Ensure the generated Python code is safe and focuses only on data analysis. Do not include code for reading files or displaying outputs (like print or st.write).
    8.  Output **only the Python code block** followed immediately by the `# chart_config:` comment with the JSON object. Do not include explanations or markdown formatting.

    Example generated code:
    ```python
    # Example 1: Simple filtering, display as table
    result_df = df[df['Sales'] > 1000].copy()
    # chart_config: {{"type": "table"}}

    # Example 2: Grouping and aggregation, bar chart with color
    result_df = df.groupby(['Region', 'Category'])['Profit'].sum().reset_index()
    # chart_config: {{"type": "bar", "x": "Region", "y": "Profit", "color": "Category", "tooltip": ["Region", "Category", "Profit"]}}

    # Example 3: Time series line plot
    result_df = df.groupby('OrderDate')['Sales'].sum().reset_index()
    result_df['OrderDate'] = pd.to_datetime(result_df['OrderDate']) # Ensure datetime
    # chart_config: {{"type": "line", "x": "OrderDate", "y": "Sales", "tooltip": ["OrderDate", "Sales"]}}

    # Example 4: Combined bar and line chart
    result_df = df.groupby('Month').agg(TotalSales=('Sales', 'sum'), AverageQuantity=('Quantity', 'mean')).reset_index()
    # chart_config: {{"type": "combined", "x": "Month", "tooltip": ["Month", "TotalSales", "AverageQuantity"], "layers": [{{"type": "bar", "y": "TotalSales"}}, {{"type": "line", "y": "AverageQuantity", "color_override": "red"}}]}}
    # Note: In combined charts, 'x' and 'tooltip' are often defined at the top level. 'y' is usually per-layer. 'color_override' can set a specific color for a layer mark.

    # Example 5: Scatter plot with color encoding
    result_df = df[['Sales', 'Profit', 'Segment']].copy()
    # chart_config: {{"type": "scatter", "x": "Sales", "y": "Profit", "color": "Segment", "tooltip": ["Sales", "Profit", "Segment"]}}
    ```

    Now, generate the Python code and the chart configuration comment for the user query:
    """
    try:
        response = model.generate_content(prompt)
        # Clean the response to extract code and config
        full_response = response.text.strip()

        # Find the chart config comment
        config_match = re.search(r"#\s*chart_config:\s*({.*})", full_response, re.DOTALL)

        if config_match:
            config_json_str = config_match.group(1)
            # Extract the code part before the comment
            code = full_response[:config_match.start()].strip()
        else:
            # Fallback if config is not found (shouldn't happen with good prompt)
            code = full_response
            config_json_str = '{"type": "table"}' # Default to table

        # Clean the code block markers if present
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.endswith("```"):
            code = code[:-len("```")].strip()

        return code, config_json_str

    except Exception as e:
        st.error(f"Error generating code/config with Gemini: {e}")
        traceback.print_exc()
        return None, None

def execute_code(code, config_json_str, df):
    """Executes the generated Python code and parses chart config."""
    local_vars = {"df": df, "pd": pd}
    global_vars = {}
    result_df = None
    chart_config = None

    try:
        # Parse the chart configuration JSON
        try:
            chart_config = json.loads(config_json_str)
        except json.JSONDecodeError as json_e:
            st.warning(f"Could not parse chart configuration JSON: {json_e}")
            st.text(f"Received config string: {config_json_str}")
            chart_config = {"type": "table"} # Fallback

        # Execute the pandas code
        if not code:
             st.warning("Generated code was empty.")
             return None, chart_config # Return config even if code is empty

        exec(code, global_vars, local_vars)

        # Retrieve the result DataFrame
        result_df = local_vars.get("result_df", None)

        if result_df is None:
             st.warning("Generated code did not produce a DataFrame named 'result_df'.")
             # Attempt to find *any* DataFrame in local_vars as a fallback
             for var_name, var_value in local_vars.items():
                 if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                     result_df = var_value
                     st.info(f"Using DataFrame named '{var_name}' as result.")
                     break

        if not isinstance(result_df, pd.DataFrame):
             st.warning(f"Could not find a resulting pandas DataFrame. Last result type: {type(result_df)}")
             # Attempt to display non-DataFrame result if it exists
             if result_df is not None:
                 st.write("Non-DataFrame result from code execution:")
                 st.write(result_df)
             return None, chart_config # Return suggestion even if df is wrong

        return result_df, chart_config

    except Exception as e:
        st.error(f"Error executing generated code:")
        # st.code(traceback.format_exc(), language='text')
        return None, chart_config # Return config even if execution fails

def display_altair_chart(df, config):
    """Displays an Altair chart based on the configuration."""
    if not config or not isinstance(df, pd.DataFrame) or df.empty:
        st.write("No chart configuration provided or data is unsuitable for charting.")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df) # Show table if data exists but config is bad
        return

    chart_type = config.get("type", "table") # Default to table if type missing
    x_col = config.get("x")
    y_col = config.get("y")
    color_col = config.get("color")
    tooltip = config.get("tooltip") # Use user-defined or let Altair decide
    layers = config.get("layers")

    try:
        

        if chart_type == "table":
            # st.dataframe(df)
            return

        if not x_col and chart_type != "table":
             st.write(f"Suggested Chart ({chart_type.capitalize()}):")
             # Try to infer a reasonable x-column if missing, e.g., index or first column
             if df.index.name:
                 x_col = df.index.name
                 df = df.reset_index() # Altair works better with columns
                 st.info(f"Inferred X-axis: '{x_col}' (from index)")
             elif len(df.columns) > 0:
                 x_col = df.columns[0]
                 st.info(f"Inferred X-axis: '{x_col}' (first column)")
             else:
                 st.warning("Cannot create chart: No columns found in the data.")
                 st.dataframe(df)
                 return

        # Base chart definition (common properties)
        base = alt.Chart(df).encode(
            tooltip=tooltip # Use specified tooltips or Altair defaults if None
        )

        # Handle combined charts
        if chart_type == "combined" and layers:
            chart_layers = []
            for i, layer_config in enumerate(layers):
                layer_type = layer_config.get("type")
                layer_y = layer_config.get("y")
                layer_x = layer_config.get("x", x_col) # Inherit top-level x if not specified
                layer_color_col = layer_config.get("color", color_col) # Inherit top-level color
                layer_color_override = layer_config.get("color_override") # Specific color like "red"

                if not layer_type or not layer_y:
                    st.warning(f"Skipping combined layer {i+1}: 'type' or 'y' missing.")
                    continue

                # Encoding for this layer
                encoding = {"x": alt.X(layer_x), "y": alt.Y(layer_y)}
                if layer_color_col and layer_color_col in df.columns:
                    encoding["color"] = alt.Color(layer_color_col)
                elif layer_color_override:
                     encoding["color"] = alt.value(layer_color_override) # Set specific color

                # Create mark for this layer
                mark = None
                if layer_type == "bar":
                    mark = base.mark_bar().encode(**encoding)
                elif layer_type == "line":
                    mark = base.mark_line().encode(**encoding)
                elif layer_type == "area":
                    mark = base.mark_area().encode(**encoding)
                elif layer_type == "scatter":
                     mark = base.mark_circle().encode(**encoding) # Use circle for scatter
                # Add more layer types if needed

                if mark:
                    chart_layers.append(mark)

            if chart_layers:
                # Layer the charts
                final_chart = alt.layer(*chart_layers).resolve_scale(
                    y='independent' # Allow independent y-axes if needed
                ).interactive() # Make combined chart interactive
                st.altair_chart(final_chart, use_container_width=True)
            else:
                st.warning("Could not create combined chart from layers.")
                st.dataframe(df)

        # Handle single-type charts
        else:
            encoding = {}
            if x_col and x_col in df.columns:
                encoding["x"] = alt.X(x_col)
            if y_col:
                 # Handle single or multiple y columns
                 if isinstance(y_col, list):
                      # Might need specific handling for multiple Ys depending on chart type
                      # For simplicity, using the first one for now, or requires specific logic
                      if y_col and y_col[0] in df.columns:
                           encoding["y"] = alt.Y(y_col[0])
                      else:
                           st.warning(f"Invalid Y column(s): {y_col}")
                           st.dataframe(df); return
                 elif y_col in df.columns:
                      encoding["y"] = alt.Y(y_col)
                 else:
                      st.warning(f"Invalid Y column: {y_col}")
                      st.dataframe(df); return

            if color_col and color_col in df.columns:
                encoding["color"] = alt.Color(color_col)

            # Apply encoding to base chart
            chart = base.encode(**encoding)

            # Select mark based on type
            if chart_type == "bar":
                final_chart = chart.mark_bar().interactive()
            elif chart_type == "line":
                final_chart = chart.mark_line().interactive()
            elif chart_type == "scatter":
                final_chart = chart.mark_circle().interactive() # Use circle for scatter
            elif chart_type == "area":
                final_chart = chart.mark_area().interactive()
            else:
                st.warning(f"Unsupported single chart type: {chart_type}. Displaying table.")
                st.dataframe(df)
                return

            st.altair_chart(final_chart, use_container_width=True)

    except Exception as e:
        print(f"Error displaying Altair chart:")
        # st.code(traceback.format_exc(), language='text')
        # st.dataframe(df) # Fallback to showing the dataframe

# --- Streamlit App Logic ---

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "df_info" not in st.session_state:
    st.session_state.df_info = None

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("1. Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataframe = df
            st.session_state.df_info = get_df_info(df)
            st.success("CSV file loaded successfully!")
            st.dataframe(df.head(), height=200)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.session_state.dataframe = None
            st.session_state.df_info = None

    if st.session_state.dataframe is not None:
        st.header("Dataframe Info")
        st.text_area("Schema", st.session_state.df_info, height=300, disabled=True)

# --- Main Area: Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "text" in message:
            st.markdown(message["text"])
        # if "code" in message:
            # st.code(message["code"], language="python")
        # Display dataframe and chart from stored data if available
        if "dataframe" in message and isinstance(message["dataframe"], pd.DataFrame):
            st.dataframe(message["dataframe"])
        if "chart_config" in message and message["chart_config"] and "dataframe" in message and isinstance(message["dataframe"], pd.DataFrame):
            # Re-display chart using stored config and df
            # Note: This re-renders the chart every time, could be optimized if needed
            display_altair_chart(message["dataframe"], message["chart_config"])
        elif "chart_config" in message and message["chart_config"]:
             # Indicate a chart was intended if df isn't stored/valid
             st.markdown(f"_Chart intended ({message['chart_config'].get('type', 'unknown')})_")

# Accept user input
if prompt := st.chat_input("Ask a question about your data..."):
    if st.session_state.dataframe is None:
        st.warning("Please upload a CSV file first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using Gemini
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        generated_code, chart_config_json = generate_gemini_code(prompt, st.session_state.df_info)

        bot_response_message = {"role": "assistant"}

        if generated_code is not None and chart_config_json is not None:
            # st.code(generated_code, language="python") # Show the generated code
            # st.caption("Chart Configuration Suggestion:")
            # st.json(chart_config_json) # Show the suggested JSON config
            bot_response_message["code"] = generated_code
            # Don't store raw JSON string, store parsed dict after execution

            message_placeholder.markdown("Executing generated code...")
            result_df, chart_config = execute_code(generated_code, chart_config_json, st.session_state.dataframe.copy())

            bot_response_message["chart_config"] = chart_config # Store parsed config

            if result_df is not None:
                message_placeholder.markdown("Here's the result:")
                st.dataframe(result_df)
                # Store dataframe in history *only if it's not too large* (optional optimization)
                # For simplicity, storing it for now. Consider size limits in production.
                bot_response_message["dataframe"] = result_df

                # Display chart if config is valid and df exists
                if chart_config:
                    display_altair_chart(result_df, chart_config)
                else:
                    st.write("No valid chart configuration generated.")

            else:
                message_placeholder.error("Could not retrieve results from the generated code.")
                bot_response_message["text"] = "Sorry, I couldn't retrieve results from the generated code."
                # Display table based on fallback config if execution failed but config was parsed
                if chart_config and chart_config.get("type") == "table":
                     st.write("Displaying empty table based on fallback config.")
                     st.dataframe(pd.DataFrame()) # Show empty df

        else:
            message_placeholder.error("Sorry, I couldn't generate the code or chart configuration.")
            bot_response_message["text"] = "Sorry, I couldn't generate the code or chart configuration for your query."

        st.session_state.messages.append(bot_response_message)

