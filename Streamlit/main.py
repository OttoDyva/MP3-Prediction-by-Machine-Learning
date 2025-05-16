import streamlit as st
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
from scipy import stats

# ---- Load your data ----
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "HR-Employee-Attrition.csv")
data_path = os.path.normpath(data_path)

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.error(f"Dataset not found at: {data_path}")
    df = pd.DataFrame()  # fallback empty dataframe

# ---- Set page config ----
st.set_page_config(page_title="Data Dashboard", layout="centered")

# ---- Title ----
st.title("Mini Project 3")
st.write('Group 6 - Bekhan, Otto, Victor & Patrick')

# ---- Buttons ----
col1, col2, col3, col4 = st.columns(4)

if "stage" not in st.session_state:
    st.session_state["stage"] = None

with col1:
    if st.button("Data wrangling"):
        st.session_state["stage"] = "Data wrangling"

with col2:
    if st.button("Linear regression"):
        st.session_state["stage"] = "Linear regression"

with col3:
    if st.button("Classification"):
        st.session_state["stage"] = "Classification"

with col4:
    if st.button("Clustering"):
        st.session_state["stage"] = "Clustering"

st.write("")

# ---- Show content based on selection ----
if st.session_state["stage"] is None:
    st.subheader("Welcome! Please select a module:")
    st.write("""
    - **Data wrangling:** Cleaning and preparing your data.
    - **Linear regression:** Predict continuous outcomes.
    - **Classification:** Assign labels to data points.
    - **Clustering:** Group similar data points.
    """)
else:
    stage = st.session_state["stage"]
    st.subheader(f"Stage: {stage}")

    if df.empty:
        st.error("Dataset not loaded or empty.")
    else:
        if stage == "Data wrangling":
            st.write("First we will check a glimpse of the dataset to see how it's structured:")
            st.dataframe(df.head())

            st.write("Now we will do some exploration with checking for the number of non-null values and data types:")
            st.write("RangeIndex and columns:", df.shape)
            with st.expander("Show full df.info()"):
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            
            st.subheader("Summary Statistics")
            st.dataframe(df.describe())

            st.subheader("Missing Values per Column")
            with st.expander("Press here to view"):
                missing_values = df.isnull().sum()
                st.table(missing_values)

            # Show unique values in Over18 before dropping it
            if 'Over18' in df.columns:
                st.write("Unique values in 'Over18' column before dropping:")
                st.write(df['Over18'].unique())

                df = df.drop(['Over18'], axis=1)
                st.write("Dropped 'Over18' column.")

            
            # Encode categorical object columns
            object_cols = df.select_dtypes(include='object').columns
            mapping_report = {}

            for col in object_cols:
                unique_vals = df[col].unique()
                val_to_code = {val: code for code, val in enumerate(unique_vals)}
                df[col] = df[col].map(val_to_code)
                mapping_report[col] = val_to_code

            st.subheader("Categorical to Numeric Mappings")
            with st.expander("View"):
                for col, mapping in mapping_report.items():
                    st.write(f"**Column: {col}**")
                    for k, v in mapping.items():
                        st.write(f"'{k}' → {v}")
                    st.write("---")

            # Outlier removal on skewed numeric columns
            skewed_cols = [
                'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
                'YearsSinceLastPromotion', 'NumCompaniesWorked',
                'DistanceFromHome', 'JobLevel'
            ]

            df_before = df.copy()
            rows_before = len(df)
            st.write(f"Rows before outlier removal: {rows_before}")

            removed_per_column = {}

            for col in skewed_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR

                    outliers = df[(df[col] < lower) | (df[col] > upper)]
                    removed_per_column[col] = len(outliers)

                    df = df[(df[col] >= lower) & (df[col] <= upper)]

            rows_after = len(df)
            total_removed = rows_before - rows_after

            st.write(f"Rows after outlier removal:  {rows_after}")
            st.write(f"Total rows removed:          {total_removed}")

            st.subheader("Rows removed per column")
            for col, count in removed_per_column.items():
                st.write(f"{col}: {count}")

            # Q-Q plots before and after
            st.subheader("Q–Q Plots Before and After Outlier Removal")
            for col in skewed_cols:
                if col in df.columns:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    stats.probplot(df_before[col], dist="norm", plot=axes[0])
                    axes[0].set_title(f"Before - Q–Q Plot of {col}")

                    stats.probplot(df[col], dist="norm", plot=axes[1])
                    axes[1].set_title(f"After - Q–Q Plot of {col}")

                    plt.tight_layout()
                    st.pyplot(fig)

        elif stage == "Linear regression":
            st.write("Linear regression analysis will go here.")
            # Add your linear regression code here using df

        elif stage == "Classification":
            st.write("Classification analysis will go here.")
            # Add your classification code here using df

        elif stage == "Clustering":
            st.write("Clustering analysis will go here.")
            # Add your clustering code here using df
