import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Tourism Analytics Dashboard", 
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1e88e5;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #e53935;
    margin: 1.5rem 0 1rem 0;
}
.insight-box {
    background-color: #e3f2fd;
    border-left: 5px solid #1e88e5;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
.metric-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.context-box {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏨 Interactive Tourism Analytics Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>📊 Dashboard Overview</strong><br>
This interactive dashboard analyzes tourism data across different regions and initiative types. 
The visualizations help identify patterns in tourism development and infrastructure distribution.
Use the sidebar controls to filter data and explore different perspectives on tourism metrics.
</div>
""", unsafe_allow_html=True)

# Default dataset URL
DEFAULT_CSV_URL = "https://linked.aub.edu.lb/pkgcube/data/551015b5649368dd2612f795c2a9c2d8_20240902_115953.csv"

@st.cache_data
def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)

def find_col(df, candidates):
    """Return first column in df whose name contains any of the candidate substrings (case-insensitive)."""
    if df is None:
        return None
    cols = df.columns.tolist()
    for cand in candidates:
        cand_l = cand.lower()
        for original in cols:
            if cand_l in original.lower():
                return original
    return None

# Sidebar for data loading and controls
st.sidebar.markdown("## 📁 Data Source")
use_url = st.sidebar.checkbox("Load dataset from URL", value=True)

df = None
err = None

if use_url:
    st.sidebar.caption("📡 Loading from online source...")
    df, err = load_data_from_url(DEFAULT_CSV_URL)
    if err:
        st.sidebar.error(f"❌ Could not load from URL: {err}")

if df is None:
    uploaded = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")

if df is None:
    st.warning("⚠️ No data loaded yet. Please enable URL loading or upload a CSV file using the sidebar.")
    st.stop()

# Data info
st.sidebar.markdown("## 📊 Dataset Information")
st.sidebar.metric("Total Rows", f"{df.shape[0]:,}")
st.sidebar.metric("Total Columns", f"{df.shape[1]}")

# Expandable data preview
with st.expander("🔍 Preview Dataset (First 5 Rows)"):
    st.dataframe(df.head(), use_container_width=True)

with st.expander("📋 Column Names"):
    cols_df = pd.DataFrame({"Column Names": df.columns.tolist()})
    st.dataframe(cols_df, use_container_width=True)

# Column detection
col_initiative = find_col(df, [
    "Existence of initiatives", "Existence of initiativ", "existence of initiativ", 
    "initiatives and projects", "initiatives"
])

col_tourism_index = find_col(df, ["Tourism Index", "Tourism_Index", "tourism index"])
col_total_hotels = find_col(df, ["Total number of hotels", "Total number of hotel", "total hotels", "total number"])
col_governorate = find_col(df, ["Governorate", "governorate", "Region", "region", "Mohafazat", "mohafazat"])

# Numeric columns for analysis
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
preferred_metrics = []
if col_tourism_index and col_tourism_index in numeric_cols:
    preferred_metrics.append(col_tourism_index)
if col_total_hotels and col_total_hotels in numeric_cols:
    preferred_metrics.append(col_total_hotels)
for c in numeric_cols:
    if c not in preferred_metrics:
        preferred_metrics.append(c)

# Sidebar controls
st.sidebar.markdown("## 🎛️ Interactive Controls")

# Governorate filter
governorate_choice = None
if col_governorate:
    uniq_gov = sorted(df[col_governorate].dropna().unique().tolist())
    governorate_choice = st.sidebar.multiselect(
        f"🗺️ Filter by {col_governorate}",
        options=uniq_gov,
        default=uniq_gov,
        help="Select regions to include in the analysis"
    )

# Initiative filter
selected_initiatives = None
if col_initiative:
    uniq_init = sorted(df[col_initiative].dropna().unique().tolist())
    selected_initiatives = st.sidebar.multiselect(
        "🏗️ Initiative Status Filter",
        options=uniq_init,
        default=uniq_init,
        help="Filter by existence of tourism initiatives"
    )
else:
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        selected_cat = st.sidebar.selectbox("📊 Choose categorical column", options=[None] + categorical_cols)
        if selected_cat:
            col_initiative = selected_cat
            uniq_init = sorted(df[col_initiative].dropna().unique().tolist())
            selected_initiatives = st.sidebar.multiselect(f"Filter by {selected_cat}", options=uniq_init, default=uniq_init)

# Metric and aggregation selection
if preferred_metrics:
    metric = st.sidebar.selectbox("📈 Select Metric to Analyze", preferred_metrics, 
                                help="Choose the numeric variable for analysis")
    agg_func = st.sidebar.selectbox("🔢 Aggregation Method", 
                                  ["mean", "median", "sum", "count"], 
                                  index=0,
                                  help="How to aggregate the metric by groups")
else:
    st.error("❌ No numeric columns found in the dataset!")
    st.stop()

# Apply filters
df_filtered = df.copy()
if governorate_choice is not None and len(governorate_choice) > 0:
    df_filtered = df_filtered[df_filtered[col_governorate].isin(governorate_choice)]
if col_initiative and selected_initiatives is not None and len(selected_initiatives) > 0:
    df_filtered = df_filtered[df_filtered[col_initiative].isin(selected_initiatives)]

# Display filter summary
if len(df_filtered) != len(df):
    st.info(f"📊 Showing {len(df_filtered):,} out of {len(df):,} total records based on your filters")

# Key metrics display
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_records = len(df_filtered)
    st.metric("📊 Filtered Records", f"{total_records:,}")

with col2:
    if col_governorate and governorate_choice:
        regions_count = len(governorate_choice)
        st.metric("🗺️ Selected Regions", regions_count)

with col3:
    if metric in df_filtered.columns:
        avg_metric = df_filtered[metric].mean()
        st.metric(f"📈 Avg {metric}", f"{avg_metric:.2f}" if not pd.isna(avg_metric) else "N/A")

with col4:
    if selected_initiatives:
        init_count = len(selected_initiatives)
        st.metric("🏗️ Initiative Types", init_count)

# Visualization 1: Aggregated Analysis
st.markdown('<div class="sub-header">📊 Visualization 1: Regional Analysis by Initiative Status</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This visualization shows how the selected metric varies across different initiative statuses.
It helps identify which regions or initiative types perform better in terms of the chosen tourism metric.
</div>
""", unsafe_allow_html=True)

if col_initiative and not df_filtered.empty:
    # Compute aggregation
    if agg_func == "mean":
        agg_df = df_filtered.groupby(col_initiative)[metric].mean().reset_index()
    elif agg_func == "median":
        agg_df = df_filtered.groupby(col_initiative)[metric].median().reset_index()
    elif agg_func == "sum":
        agg_df = df_filtered.groupby(col_initiative)[metric].sum().reset_index()
    else:  # count
        agg_df = df_filtered.groupby(col_initiative)[metric].count().reset_index()
    
    agg_df = agg_df.sort_values(by=metric, ascending=False)
    
    # Create enhanced bar chart
    fig1 = px.bar(
        agg_df, 
        x=col_initiative, 
        y=metric,
        title=f"{agg_func.title()} of {metric} by {col_initiative}",
        labels={col_initiative: col_initiative, metric: f"{agg_func.title()} {metric}"},
        color=metric,
        color_continuous_scale="viridis",
        text=metric
    )
    
    # Customize the chart
    fig1.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig1.update_layout(
        height=500,
        xaxis={'categoryorder': 'total descending'},
        showlegend=False
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Show insights
    st.markdown(f"""
    <div class="insight-box">
    <strong>💡 Key Insights:</strong><br>
    • Highest {agg_func}: <strong>{agg_df.iloc[0][col_initiative]}</strong> with {agg_df.iloc[0][metric]:.2f}<br>
    • Lowest {agg_func}: <strong>{agg_df.iloc[-1][col_initiative]}</strong> with {agg_df.iloc[-1][metric]:.2f}<br>
    • Range: {agg_df[metric].max() - agg_df[metric].min():.2f} ({metric})
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Cannot create visualization - no valid categorical column or empty dataset.")

# Visualization 2: Distribution Analysis
st.markdown('<div class="sub-header">📈 Visualization 2: Distribution Analysis</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This box plot shows the distribution and variability of the selected metric across different categories.
It helps identify outliers, quartiles, and overall spread of the data.
</div>
""", unsafe_allow_html=True)

# Choose distribution column
dist_candidates = []
if col_total_hotels and col_total_hotels in numeric_cols:
    dist_candidates.append(col_total_hotels)
if metric not in dist_candidates:
    dist_candidates.append(metric)
for n in numeric_cols:
    if n not in dist_candidates:
        dist_candidates.append(n)

dist_col = st.selectbox("🎯 Select column for distribution analysis:", dist_candidates, 
                       help="Choose which numeric variable to analyze the distribution of")

if col_initiative and dist_col:
    # Prepare data for box plot
    df_box = df_filtered[[col_initiative, dist_col]].copy()
    df_box = df_box.dropna()
    
    # Ensure numeric
    try:
        df_box[dist_col] = pd.to_numeric(df_box[dist_col], errors='coerce')
        df_box = df_box.dropna(subset=[dist_col])
    except:
        pass
    
    if not df_box.empty:
        # Create enhanced box plot
        fig2 = px.box(
            df_box, 
            x=col_initiative, 
            y=dist_col,
            title=f"Distribution of {dist_col} by {col_initiative}",
            labels={col_initiative: col_initiative, dist_col: dist_col},
            color=col_initiative,
            points="outliers"  # Show outliers
        )
        
        # Add mean markers
        mean_values = df_box.groupby(col_initiative)[dist_col].mean().reset_index()
        fig2.add_scatter(
            x=mean_values[col_initiative],
            y=mean_values[dist_col],
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            name='Mean',
            showlegend=True
        )
        
        fig2.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Distribution insights
        stats = df_box.groupby(col_initiative)[dist_col].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(stats, use_container_width=True)
        
    else:
        st.warning("⚠️ No valid numeric data available for the selected distribution column and filters.")

# Additional Analysis Section
st.markdown('<div class="sub-header">🔍 Additional Interactive Analysis</div>', 
            unsafe_allow_html=True)

analysis_type = st.selectbox(
    "Choose additional analysis:",
    ["Summary Statistics", "Correlation Analysis", "Top Performers"]
)

if analysis_type == "Summary Statistics":
    st.markdown("### 📊 Comprehensive Statistics")
    numeric_summary = df_filtered[numeric_cols].describe()
    st.dataframe(numeric_summary, use_container_width=True)

elif analysis_type == "Correlation Analysis" and len(numeric_cols) > 1:
    st.markdown("### 🔗 Correlation Matrix")
    corr_matrix = df_filtered[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        aspect='auto',
        color_continuous_scale='RdBu',
        title='Correlation Matrix of Numeric Variables'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_type == "Top Performers":
    st.markdown("### 🏆 Top Performers")
    if col_governorate and metric:
        top_regions = df_filtered.groupby(col_governorate)[metric].mean().nlargest(5)
        col_a, col_b = st.columns(2)
        with col_a:
            st.bar_chart(top_regions)
        with col_b:
            st.write("**Top 5 Regions:**")
            for region, value in top_regions.items():
                st.write(f"• {region}: {value:.2f}")

# Footer with insights and instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Use This Dashboard:

1. **Data Filtering**: Use sidebar controls to filter by region and initiative status
2. **Metric Selection**: Choose different tourism metrics to analyze various aspects
3. **Aggregation Methods**: Switch between mean, median, sum, and count for different perspectives
4. **Distribution Analysis**: Examine data spread and identify outliers using box plots

### 💡 Key Design Decisions:

- **Interactive Filtering**: All visualizations update dynamically based on your selections
- **Multiple Perspectives**: Bar charts show aggregated values while box plots reveal distributions
- **Smart Column Detection**: Automatically finds relevant tourism columns in your dataset
- **Statistical Insights**: Provides summary statistics and correlation analysis for deeper understanding
""")

# Export functionality
if st.button("📥 Export Filtered Data"):
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"filtered_tourism_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
