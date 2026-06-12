import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page setup
st.set_page_config(
    page_title="Black Friday Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Curated cool-tone palette (strictly no orange or red)
COOL_PALETTE = [
    '#00ADB5',  # Bright Teal
    '#58A6FF',  # Soft Blue
    '#1F6FEB',  # Royal Blue
    '#BC8CFF',  # Soft Purple
    '#00D2FC',  # Cyan
    '#8B949E',  # Slate Grey
    '#38BDF8',  # Sky Blue
    '#0ea5e9',  # Cyan-Blue
    '#0284C7',  # Medium Blue
    '#1e40af',  # Dark Blue
    '#3b82f6',  # Bright Blue
    '#6366f1',  # Indigo
    '#a855f7'   # Lavender
]

# Custom premium styling for modern dark interface (GitHub Dark / VSCode style)
st.markdown("""
    <style>
    /* Main body and backgrounds */
    .stApp {
        background-color: #0D1117;
        color: #C9D1D9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D !important;
        padding-top: 20px;
    }
    [data-testid="stSidebar"] .stMultiSelect {
        background-color: #0D1117 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #F0F6FC !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #30363D;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    /* Expander inside sidebar */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #0D1117 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
        color: #C9D1D9 !important;
        font-size: 14px !important;
    }
    
    /* Tab controls */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161B22;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #30363D;
        margin-top: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #0D1117;
        border-radius: 6px;
        color: #8B949E;
        border: 1px solid #30363D;
        padding: 4px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F0F6FC;
        border-color: #8B949E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262D !important;
        color: #F0F6FC !important;
        border: 1px solid #00ADB5 !important;
        box-shadow: 0 0 10px rgba(0, 173, 181, 0.2);
    }
    
    /* Metric Cards with glassmorphism/hover effects */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 173, 181, 0.2);
        border-color: #00ADB5;
    }
    .metric-label {
        font-size: 11px;
        color: #8B949E;
        text-transform: uppercase;
        margin-bottom: 6px;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #00ADB5;
    }
    
    /* Action Strategy Box */
    .recommendation-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-left: 4px solid #00ADB5;
        padding: 20px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Caption helper text styling */
    .chart-caption {
        font-size: 13px;
        color: #8B949E;
        margin-top: 8px;
        margin-bottom: 25px;
        line-height: 1.4;
        border-left: 2px solid #30363D;
        padding-left: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Data Loading & Caching -----------------

@st.cache_data
def load_data():
    path = r"C:\Users\FATHIR\.cache\kagglehub\datasets\noopurbhatt\retail-black-friday-sales-dataset\versions\1"
    filepath = os.path.join(path, 'retail_black_friday_sales_100k.csv')
    df = pd.read_csv(filepath)
    df.rename(columns={'purchase_date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def get_rfm_segmented_df(df):
    max_date = df['date'].max()
    rfm = df.groupby('customer_id').agg(
        recency=('date', lambda x: (max_date - x.max()).total_seconds() / 86400.0),
        frequency=('transaction_id', 'count'),
        monetary=('purchase_amount', 'sum')
    ).reset_index()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    rfm_scaled = pd.DataFrame(scaled_features, columns=['recency_scaled', 'frequency_scaled', 'monetary_scaled'])

    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_names = {
        2: 'Champions/VIPs',
        3: 'Loyal Shoppers',
        1: 'Occasional Shoppers',
        0: 'At-Risk'
    }
    rfm['segment'] = rfm['cluster'].map(cluster_names)
    
    customer_segments = rfm[['customer_id', 'segment', 'recency', 'frequency', 'monetary']]
    df_merged = df.merge(customer_segments, on='customer_id', how='left')
    return df_merged, rfm

try:
    base_df = load_data()
    df_merged, rfm_df = get_rfm_segmented_df(base_df)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ----------------- Sidebar Filters -----------------

st.sidebar.markdown("### Dashboard Filters")

# Unique values extraction for dropdowns
all_cities = sorted(df_merged['city'].unique())
all_ages = sorted(df_merged['age_group'].unique())
all_categories = sorted(df_merged['product_category'].unique())
all_payments = sorted(df_merged['payment_method'].unique())

# Gender Filter (Horizontal Radio button for clean single-click action)
gender_opt = st.sidebar.radio("Gender", options=["All Genders", "Female", "Male"], horizontal=True)

# City Filter (Horizontal Radio button for quick geographic toggling)
city_opt = st.sidebar.radio("City Category", options=["All Cities", "City A", "City B", "City C"], horizontal=True)

# Age Group Filter (Clean selectbox dropdown, defaults to all)
age_opt = st.sidebar.selectbox("Age Group", options=["All Ages"] + all_ages)

# Product Category Filter (Clean selectbox dropdown, defaults to all)
category_opt = st.sidebar.selectbox("Product Category", options=["All Categories"] + [str(c) for c in all_categories])

# Payment Method Filter (Clean selectbox dropdown, defaults to all)
payment_opt = st.sidebar.selectbox("Payment Method", options=["All Payments"] + all_payments)

# Apply filters step-by-step to the merged dataset
df_filtered = df_merged.copy()

if gender_opt == "Female":
    df_filtered = df_filtered[df_filtered['gender'] == 'F']
elif gender_opt == "Male":
    df_filtered = df_filtered[df_filtered['gender'] == 'M']

if city_opt != "All Cities":
    city_val = city_opt.replace("City ", "")
    df_filtered = df_filtered[df_filtered['city'] == city_val]

if age_opt != "All Ages":
    df_filtered = df_filtered[df_filtered['age_group'] == age_opt]

if category_opt != "All Categories":
    try:
        if float(category_opt).is_integer():
            cat_val = int(float(category_opt))
        else:
            cat_val = float(category_opt)
    except ValueError:
        cat_val = category_opt
    df_filtered = df_filtered[df_filtered['product_category'] == cat_val]

if payment_opt != "All Payments":
    df_filtered = df_filtered[df_filtered['payment_method'] == payment_opt]

# Add a spacer
st.sidebar.markdown("<br>", unsafe_allow_html=True)

if st.sidebar.button("Reset Filters", use_container_width=True):
    st.rerun()

# ----------------- Header & Context Card -----------------

st.markdown("""
    <div style="background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <h2 style="margin-top: 0; color: #00ADB5; font-size: 22px; font-weight: 600;">Black Friday Sales Performance & Customer Segmentation Dashboard</h2>
        <p style="margin-bottom: 0; color: #8B949E; font-size: 14px; line-height: 1.5;">
            This interactive dashboard is designed to analyze 100,000 retail transaction records collected during the Black Friday promotional window (November 24 to December 1, 2025). 
            It translates complex tabular data into actionable business intelligence by monitoring revenue peaks, analyzing promotional discount performance, and classifying customer profiles using K-Means RFM (Recency, Frequency, Monetary) clustering. 
            Use the collapsible expander filters on the left panel (sidebar) to isolate specific demographics, cities, or payment channels.
        </p>
    </div>
""", unsafe_allow_html=True)

num_records = len(df_filtered)
pct_active = (num_records / len(df_merged)) * 100
st.info(f"Showing **{num_records:,}** of **{len(df_merged):,}** transactional records ({pct_active:.1f}% of total dataset)")

# ----------------- Tabs Layout -----------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Business Overview", 
    "Product & Discount Insights", 
    "Customer RFM Segments",
    "Demographics & Payments"
])

# ================= TAB 1: BUSINESS OVERVIEW =================
with tab1:
    st.subheader("Overview of Business Metrics & Trends")
    st.markdown("<p style='color: #8B949E; margin-bottom: 20px;'>Quick KPI summaries and daily sales performance trends across the target dates.</p>", unsafe_allow_html=True)
    
    # 1. KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    
    total_rev = df_filtered['purchase_amount'].sum()
    total_tx = df_filtered['transaction_id'].nunique()
    unique_cust = df_filtered['customer_id'].nunique()
    avg_order = df_filtered['purchase_amount'].mean()
    avg_disc = df_filtered['discount_pct'].mean()
    total_qty = df_filtered['quantity'].sum()
    
    with kpi_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Revenue</div><div class="metric-value">${total_rev/1e6:.2f}M</div></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Transactions</div><div class="metric-value">{total_tx:,}</div></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Customers</div><div class="metric-value">{unique_cust:,}</div></div>', unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Order Size</div><div class="metric-value">${avg_order:.2f}</div></div>', unsafe_allow_html=True)
    with kpi_col5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Discount</div><div class="metric-value">{avg_disc:.1f}%</div></div>', unsafe_allow_html=True)
    with kpi_col6:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Items Sold</div><div class="metric-value">{total_qty:,}</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Daily Sales Trend
    df_daily = df_filtered.groupby(pd.Grouper(key='date', freq='D')).agg(
        revenue=('purchase_amount', 'sum'),
        transactions=('transaction_id', 'count')
    ).reset_index()
    
    df_daily['date_str'] = df_daily['date'].dt.strftime('%Y-%m-%d')
    
    fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_daily.add_trace(
        go.Bar(
            x=df_daily['date_str'], 
            y=df_daily['revenue'], 
            name="Revenue ($)", 
            marker_color='#1F6FEB', 
            opacity=0.85
        ),
        secondary_y=False
    )
    
    fig_daily.add_trace(
        go.Scatter(
            x=df_daily['date_str'], 
            y=df_daily['transactions'], 
            name="Transactions", 
            line=dict(color='#00D2FC', width=3), 
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    bf_day_str = '2025-11-28'
    if bf_day_str in df_daily['date_str'].values:
        fig_daily.add_vline(x=bf_day_str, line_dash="dash", line_color="#00D2FC")
        fig_daily.add_annotation(
            x=bf_day_str,
            y=float(df_daily['revenue'].max()),
            text="Black Friday",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-20,
            font=dict(color="#00D2FC", size=10, weight="bold"),
            arrowcolor="#00D2FC"
        )
        
    fig_daily.update_layout(
        title_text="Daily Revenue vs Transaction Volume",
        xaxis_title="Purchase Date",
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    fig_daily.update_yaxes(title_text="Revenue ($)", secondary_y=False, gridcolor="#21262D")
    fig_daily.update_yaxes(title_text="Transaction Count", secondary_y=True)
    
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> The blue bar heights indicate total sales revenue per day, and the cyan line plots transaction counts. The cyan vertical dashed line indicates Black Friday. This chart evaluates whether Black Friday generated a massive spike compared to neighboring dates or if promotional sales were distributed evenly.</p>", unsafe_allow_html=True)
    
    # 3. Black Friday vs Non-Black Friday Performance
    st.subheader("Black Friday vs. Non-Black Friday Performance Comparison")
    
    df_bf = df_filtered.copy()
    df_bf['period'] = np.where(df_bf['is_black_friday'] == 1, 'Black Friday', 'Non-Black Friday')
    
    bf_summary = df_bf.groupby('period').agg(
        transactions=('transaction_id', 'count'),
        revenue=('purchase_amount', 'sum'),
        avg_purchase=('purchase_amount', 'mean')
    ).reset_index()
    
    bf_summary['days'] = bf_summary['period'].map({
        'Black Friday': 1,
        'Non-Black Friday': df_bf[df_bf['is_black_friday'] == 0]['date'].dt.date.nunique()
    })
    
    bf_summary['avg_daily_revenue'] = bf_summary['revenue'] / bf_summary['days']
    bf_summary['avg_daily_transactions'] = bf_summary['transactions'] / bf_summary['days']
    
    col_bf1, col_bf2, col_bf3 = st.columns(3)
    
    with col_bf1:
        fig_bf_rev = px.bar(bf_summary, x='period', y='avg_daily_revenue', color='period',
                            color_discrete_map={'Black Friday': '#00D2FC', 'Non-Black Friday': '#1F6FEB'},
                            title="Average Daily Revenue ($)", labels={'avg_daily_revenue': 'Daily Revenue ($)'})
        fig_bf_rev.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_bf_rev.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig_bf_rev, use_container_width=True)
        
    with col_bf2:
        fig_bf_tx = px.bar(bf_summary, x='period', y='avg_daily_transactions', color='period',
                           color_discrete_map={'Black Friday': '#00D2FC', 'Non-Black Friday': '#1F6FEB'},
                           title="Average Daily Transactions", labels={'avg_daily_transactions': 'Daily Transactions'})
        fig_bf_tx.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_bf_tx.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig_bf_tx, use_container_width=True)
        
    with col_bf3:
        fig_bf_avg = px.bar(bf_summary, x='period', y='avg_purchase', color='period',
                            color_discrete_map={'Black Friday': '#00D2FC', 'Non-Black Friday': '#1F6FEB'},
                            title="Average Transaction Value ($)", labels={'avg_purchase': 'Average Value ($)'})
        fig_bf_avg.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_bf_avg.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig_bf_avg, use_container_width=True)
        
    st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> These comparative charts show how Black Friday stacks up against the average of all other non-Black Friday dates in terms of average daily revenue (left), transaction volumes (middle), and order sizes (right). This determines whether the event drove higher value or simply more transactions.</p>", unsafe_allow_html=True)

# ================= TAB 2: PRODUCT & DISCOUNT INSIGHTS =================
with tab2:
    st.subheader("Product Category Sales & Discount Optimization")
    st.markdown("<p style='color: #8B949E; margin-bottom: 20px;'>Detailed charts studying which product divisions generate the highest sales share and the optimal discount strategies.</p>", unsafe_allow_html=True)
    
    col_pr1, col_pr2 = st.columns(2)
    
    with col_pr1:
        st.subheader("Revenue Share by Product Category")
        cat_rev = df_filtered.groupby('product_category')['purchase_amount'].sum().reset_index()
        fig_donut = px.pie(cat_rev, values='purchase_amount', names='product_category', hole=0.4,
                           color_discrete_sequence=COOL_PALETTE)
        fig_donut.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> This donut chart represents the revenue contribution of each product category. Hover over segments to see exact percentages and dollar values. It helps isolate top-performing divisions (like Electronics) from low-performers.</p>", unsafe_allow_html=True)
        
    with col_pr2:
        st.subheader("Optimal Discount Threshold Analysis")
        disc_agg = df_filtered.groupby('discount_pct').agg(
            revenue=('purchase_amount', 'sum'),
            quantity_sold=('quantity', 'sum')
        ).reset_index().sort_values('discount_pct')
        
        fig_disc = make_subplots(specs=[[{"secondary_y": True}]])
        fig_disc.add_trace(
            go.Bar(x=disc_agg['discount_pct'].astype(str) + '%', y=disc_agg['revenue'], 
                   name="Revenue ($)", marker_color='#1F6FEB', opacity=0.85),
            secondary_y=False
        )
        fig_disc.add_trace(
            go.Scatter(x=disc_agg['discount_pct'].astype(str) + '%', y=disc_agg['quantity_sold'], 
                       name="Qty Sold", line=dict(color='#00D2FC', dash='dash', width=2), marker=dict(size=6)),
            secondary_y=True
        )
        fig_disc.update_layout(
            title="Revenue and Quantity Sold by Discount Level",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.01, y=0.99)
        )
        fig_disc.update_yaxes(title_text="Total Revenue ($)", secondary_y=False, gridcolor="#21262D")
        fig_disc.update_yaxes(title_text="Quantity Items Sold", secondary_y=True)
        st.plotly_chart(fig_disc, use_container_width=True)
        st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> This graph maps the revenue (bars) and quantity sold (line) against discount percentages to determine the 'sweet spot' for promotions—maximizing volume without unnecessarily sacrificing margins.</p>", unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("Purchase Amount Distribution by Discount Level")
    df_filtered['period'] = np.where(df_filtered['is_black_friday'] == 1, 'Black Friday', 'Non-Black Friday')
    
    fig_box = px.box(
        df_filtered, 
        x='discount_pct', 
        y='purchase_amount', 
        color='period', 
        color_discrete_map={'Black Friday': '#00D2FC', 'Non-Black Friday': '#1F6FEB'},
        labels={'discount_pct': 'Discount Percentage', 'purchase_amount': 'Purchase Amount ($)'},
        points=False
    )
    fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig_box.update_yaxes(gridcolor="#21262D")
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> These side-by-side boxplots show how transaction values vary across discount levels. It evaluates if higher discounts are applied to higher or lower-value purchases, and compares the spread of spending between Black Friday (cyan) and regular days (blue).</p>", unsafe_allow_html=True)

# ================= TAB 3: CUSTOMER RFM SEGMENTS =================
with tab3:
    st.subheader("Behavioral Customer Segmentation (K-Means RFM)")
    st.markdown("<p style='color: #8B949E; margin-bottom: 20px;'>Advanced customer classification using K-Means Clustering on customer Recency, Frequency, and Monetary parameters.</p>", unsafe_allow_html=True)
    
    filtered_customers = df_filtered['customer_id'].unique()
    rfm_filtered = rfm_df[rfm_df['customer_id'].isin(filtered_customers)]
    
    st.subheader("Customer Cluster Profiles Summary")
    
    seg_summary = rfm_filtered.groupby('segment').agg(
        customer_count=('customer_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean')
    ).reset_index()
    
    seg_summary['pct_customers'] = (seg_summary['customer_count'] / len(rfm_filtered) * 100).round(2)
    seg_summary = seg_summary.sort_values(by='avg_monetary', ascending=False)
    
    st.dataframe(
        seg_summary.rename(columns={
            'segment': 'Segment Name',
            'customer_count': 'Total Customers',
            'avg_recency': 'Avg Recency (Days ago)',
            'avg_frequency': 'Avg Frequency (Transactions)',
            'avg_monetary': 'Avg Spend ($)',
            'pct_customers': 'Customers Share (%)'
        }).set_index('Segment Name').style.format({
            'Total Customers': '{:,}',
            'Avg Recency (Days ago)': '{:.2f}',
            'Avg Frequency (Transactions)': '{:.2f}',
            'Avg Spend ($)': '${:,.2f}',
            'Customers Share (%)': '{:.2f}%'
        }),
        use_container_width=True
    )
    st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> This summary details the average customer metrics for the 4 clusters identified by K-Means. It exposes segments with short recency, high frequency, and high monetary spend (VIPs) compared to dormant and low-value groups (At-Risk).</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Interactive 3D Customer Segments Plot")
    
    sample_size = min(len(rfm_filtered), 5000)
    rfm_sample = rfm_filtered.sample(sample_size, random_state=42) if len(rfm_filtered) > 5000 else rfm_filtered
    
    fig_3d = px.scatter_3d(
        rfm_sample, 
        x='recency', 
        y='frequency', 
        z='monetary', 
        color='segment',
        hover_data=['customer_id'],
        labels={'recency': 'Recency (Days)', 'frequency': 'Frequency (Qty)', 'monetary': 'Monetary (Spend $)'},
        color_discrete_map={
            'Champions/VIPs': '#BC8CFF',
            'Loyal Shoppers': '#00ADB5',
            'Occasional Shoppers': '#58A6FF',
            'At-Risk': '#8B949E'
        }
    )
    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene = dict(
            xaxis = dict(title='Recency (Days)', gridcolor="#21262D"),
            yaxis = dict(title='Frequency', gridcolor="#21262D"),
            zaxis = dict(title='Spend ($)', gridcolor="#21262D")
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> A 3D spatial plot mapping customers based on Recency, Frequency, and Monetary scores. Every dot represents one customer colored by cluster. Use your mouse to rotate and zoom in on specific regions, and hover over dots to see individual customer IDs.</p>", unsafe_allow_html=True)
    
    st.subheader("Customer Segmentation Marketing Strategies")
    
    segment_select = st.selectbox("Select Customer Cluster to view action plan:", 
                                   options=['Champions/VIPs', 'Loyal Shoppers', 'Occasional Shoppers', 'At-Risk'])
    
    if segment_select == 'Champions/VIPs':
        st.markdown("""
            <div class="recommendation-box">
            <h4>VIP & Champions Strategy</h4>
            <p><strong>Profile:</strong> Very active, purchases frequently, spends highly.</p>
            <ul>
                <li>Offer exclusive previews of next-season launches.</li>
                <li>Assign dedicated premium support channels.</li>
                <li>Implement high-tier milestone rewards (e.g., free shipping, custom packaging).</li>
                <li>Avoid aggressive discounts; these customers buy for quality and exclusivity, not cheap prices.</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
    elif segment_select == 'Loyal Shoppers':
        st.markdown("""
            <div class="recommendation-box" style="border-left-color: #00ADB5;">
            <h4>Loyal Shoppers Strategy</h4>
            <p><strong>Profile:</strong> Recent purchases, consistent shoppers, solid monetary contribution.</p>
            <ul>
                <li>Introduce cross-selling schemes (e.g., recommend accessories for products already purchased).</li>
                <li>Offer points-based reward multipliers for their next purchases.</li>
                <li>Collect customer feedback to build brand advocacy.</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
    elif segment_select == 'Occasional Shoppers':
        st.markdown("""
            <div class="recommendation-box" style="border-left-color: #58A6FF;">
            <h4>Active Occasional Shoppers Strategy</h4>
            <p><strong>Profile:</strong> Recent transactions but low buying frequency and lower monetary spend.</p>
            <ul>
                <li>Implement basket-size triggers (e.g., "Spend $50 more to get 10% off").</li>
                <li>Send targeted notifications about stock limits of items they viewed.</li>
                <li>Offer bundles of low-value and high-value accessories.</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
    elif segment_select == 'At-Risk':
        st.markdown("""
            <div class="recommendation-box" style="border-left-color: #8B949E;">
            <h4>At-Risk / Churn Prevention Strategy</h4>
            <p><strong>Profile:</strong> Long time since last transaction, low frequency, and low spend.</p>
            <ul>
                <li>Send automated email win-back alerts (e.g., "We miss you, here is a 20% coupon").</li>
                <li>Run remarketing campaigns on Google/Social platforms highlighting trending seasonal items.</li>
                <li>Clean mailing list if no activity occurs in the next 30 days to save server costs.</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: DEMOGRAPHICS & PAYMENTS =================
with tab4:
    st.header("Demographic Groups & Payment Channels Analysis")
    st.markdown("<p style='color: #8B949E; margin-bottom: 20px;'>Analysis of geographic city revenue, age segmentations, gender divisions, and payment channel performance.</p>", unsafe_allow_html=True)
    
    # 1. City Sales Analysis
    st.subheader("Sales Breakdown by City")
    city_summary = df_filtered.groupby('city').agg(
        revenue=('purchase_amount', 'sum'),
        transactions=('transaction_id', 'count')
    ).reset_index().sort_values('revenue', ascending=True)
    
    fig_city = make_subplots(rows=1, cols=2, subplot_titles=("Revenue by City ($)", "Transactions by City"))
    
    fig_city.add_trace(
        go.Bar(y=city_summary['city'], x=city_summary['revenue'], orientation='h', 
               marker_color='#58A6FF', name="Revenue ($)"),
        row=1, col=1
    )
    fig_city.add_trace(
        go.Bar(y=city_summary['city'], x=city_summary['transactions'], orientation='h', 
               marker_color='#00ADB5', name="Transactions"),
        row=1, col=2
    )
    fig_city.update_layout(height=400, showlegend=False, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig_city.update_xaxes(gridcolor="#21262D", row=1, col=1)
    fig_city.update_xaxes(gridcolor="#21262D", row=1, col=2)
    st.plotly_chart(fig_city, use_container_width=True)
    st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> A side-by-side ranking of revenue contribution (left) and transaction volumes (right) for each city. It helps isolate top geographical markets from low-performing cities.</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Age Group and Gender Side-by-Side
    col_dg1, col_dg2 = st.columns(2)
    
    with col_dg1:
        st.subheader("Sales Distribution by Age Group")
        age_summary = df_filtered.groupby('age_group')['purchase_amount'].sum().reset_index()
        fig_age = px.bar(age_summary, x='age_group', y='purchase_amount', color='age_group',
                         labels={'purchase_amount': 'Total Spend ($)'},
                         color_discrete_sequence=COOL_PALETTE)
        fig_age.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_age.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> Total customer spend contribution by age bracket. Exposes which generation contributes the highest monetary value.</p>", unsafe_allow_html=True)
        
    with col_dg2:
        st.subheader("Sales Distribution by Gender")
        gender_summary = df_filtered.groupby('gender')['purchase_amount'].sum().reset_index()
        fig_gen = px.bar(gender_summary, x='gender', y='purchase_amount', color='gender',
                         labels={'purchase_amount': 'Total Spend ($)'},
                         color_discrete_sequence=['#00ADB5', '#58A6FF'])
        fig_gen.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_gen.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig_gen, use_container_width=True)
        st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> Total customer spend contribution split by gender.</p>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 3. Payment Methods
    st.subheader("Popularity of Payment Methods")
    pay_summary = df_filtered.groupby('payment_method').agg(
        transactions=('transaction_id', 'count'),
        avg_spend=('purchase_amount', 'mean')
    ).reset_index().sort_values('transactions', ascending=False)
    
    fig_pay = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pay.add_trace(
        go.Bar(x=pay_summary['payment_method'], y=pay_summary['transactions'], 
               name="Transactions Count", marker_color='#BC8CFF', opacity=0.85),
        secondary_y=False
    )
    fig_pay.add_trace(
        go.Scatter(x=pay_summary['payment_method'], y=pay_summary['avg_spend'], 
                   name="Average Order Size ($)", marker=dict(size=8), line=dict(color='#C9D1D9', width=3)),
        secondary_y=True
    )
    fig_pay.update_layout(
        title="Transaction Volumes vs Average Order Sizes across Payment Channels",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99)
    )
    fig_pay.update_yaxes(title_text="Total Transactions", secondary_y=False, gridcolor="#21262D")
    fig_pay.update_yaxes(title_text="Average Spend ($)", secondary_y=True)
    st.plotly_chart(fig_pay, use_container_width=True)
    st.markdown("<p class='chart-caption'><strong>Interpretation:</strong> The bars map total transaction volumes (left axis) while the line maps average purchase values (right axis) for each payment method. It highlights channels that process high volumes versus channels that process high-value purchases.</p>", unsafe_allow_html=True)
