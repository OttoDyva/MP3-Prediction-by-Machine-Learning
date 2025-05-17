import streamlit as st
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import numpy as np
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
    st.subheader("Objective:")
    st.write('The objective of this mini project is to provide practice in data analysis and prediction by regression, classification and clustering algorithms.')

    st.subheader("Problem Statement:")
    st.write("Attrition is the rate at which employees leave their job. When attrition reaches high levels, it becomes a concern for the company. Therefore, it is important to find out why employees leave, which factors contribute to such significant decision.")

else:
    stage = st.session_state["stage"]
    st.title(f"Stage: {stage}")

    if df.empty:
        st.error("Dataset not loaded or empty.")
    else:
        if stage == "Data wrangling":
            st.write("### Basic Information")
            st.write("Shape of dataset:", df.shape)
            st.write("First few rows:")
            st.dataframe(df.head())

            st.write("#### Column Info")
            with st.expander("Show Column Info"):
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())

            st.write("### Summary Statistics")
            st.dataframe(df.describe())

            st.write("### Missing Values")
            st.dataframe(df.isnull().sum())

            # Drop rows with missing values
            df = df.dropna()

            st.write("Dropped rows with missing values. New shape:", df.shape)

            # Identify and drop constant columns
            constant_cols = [col for col in df.columns if df[col].nunique() == 1]
            st.write("Columns with only one unique value (to be removed):", constant_cols)
            df = df.drop(columns=constant_cols)

            # Columns for outlier removal
            skewed_cols = [
                'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
                'YearsSinceLastPromotion', 'NumCompaniesWorked',
                'DistanceFromHome', 'JobLevel'
            ]

            df_before = df.copy()
            rows_before = len(df)

            removed_per_column = {}

            for col in skewed_cols:
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

            st.write(f"Rows before outlier removal: {rows_before}")
            st.write(f"Rows after outlier removal: {rows_after}")
            st.write(f"Total rows removed: {total_removed}")
            st.subheader("Rows removed per column:")
            for col, count in removed_per_column.items():
                st.write(f"{col}: {count}")

            # Q–Q Plots before and after
            st.subheader("Q–Q Plots Before and After Outlier Removal")
            for col in skewed_cols:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                stats.probplot(df_before[col], dist="norm", plot=axes[0])
                axes[0].set_title(f"Before - Q–Q Plot of {col}")

                stats.probplot(df[col], dist="norm", plot=axes[1])
                axes[1].set_title(f"After - Q–Q Plot of {col}")

                plt.tight_layout()
                st.pyplot(fig)

            st.session_state["cleaned_df"] = df.copy()

        elif stage == "Linear regression":
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn import metrics
            import sklearn.metrics as sm
            from sklearn.metrics import r2_score

            if "cleaned_df" not in st.session_state:
                st.error("Please run Data wrangling first to generate the cleaned dataset.")
            else:
                df = st.session_state["cleaned_df"].copy()
                df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

            data_column_category = df.select_dtypes(exclude=[np.number]).columns
            st.write("Categorical Columns:", list(data_column_category))

            label_encoder = LabelEncoder()
            df_label_encoded = df.copy()
            for i in data_column_category:
                df_label_encoded[i] = label_encoder.fit_transform(df_label_encoded[i])
            st.write("Label Encoded Data Preview:")
            st.dataframe(df_label_encoded.head())

            st.write("### Correlation Analysis")
            st.write("We want to predict the income of a new employee. We are going to do that by linear regression. My first instinct is to check for correlations between different columns i find relevant to employees and payments.")
            st.write("By looking at this first correlation, we can see that Hourly-, Monthly- & DailyRate are not really relevant to YearsAtCompany at all, so we will discard them for now.")
            st.write("Checking correlation of income-related features with `YearsAtCompany`:")
            st.dataframe(df[['YearsAtCompany', 'MonthlyIncome', 'HourlyRate', 'MonthlyRate', 'DailyRate']].corr())


            st.write("I will now check for correlations with MonthlyIncome, to see which have the highest correlations with MonthlyIncome.")
            st.write("We can see, that JobLevel, TotalWorkingYears & YearsAtCompany have the highest correlation with MonthlyIncome.")
            st.write("Checking which features correlate most with `MonthlyIncome`:")
            corr_income = df_label_encoded.corr()['MonthlyIncome'].sort_values(ascending=False)
            st.dataframe(corr_income)

            st.write("Checking correlations with `YearsAtCompany`:")
            corr_years = df_label_encoded.corr()['YearsAtCompany'].sort_values(ascending=False)
            st.dataframe(corr_years)

            st.write("### Creating Feature Set")
            feature_cols = ['JobLevel', 'TotalWorkingYears']
            X = df_label_encoded[feature_cols]
            y = df_label_encoded['MonthlyIncome']

            st.write("Feature Columns (X):")
            st.dataframe(X.head())
            st.write("Target Column (y):")
            st.dataframe(y.head())

            st.write("### Splitting the Data")
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

            st.write(f"Training set shape: {X_train.shape}")
            st.write(f"Testing set shape: {X_test.shape}")

            st.write("### Training the Linear Regression Model")
            st.write("""
                **Creating and Training the Model**

                We create a linear regression model.  
                Then, we teach the model using the training data (`X_train` and `y_train`).  
                After training, we print the model's intercept (`b0`) and coefficients (`bi`), which show how each feature affects the prediction.""")

            linreg = LinearRegression()
            linreg.fit(X_train, y_train)

            st.write("Intercept (b0):", linreg.intercept_)
            st.write("Coefficients (bi):", list(zip(feature_cols, linreg.coef_)))

            st.write("### Model Predictions on Test Set")
            y_predicted = linreg.predict(X_test)
            st.write("First 10 Predictions vs Actual:")
            st.dataframe(pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_predicted[:10]}))

            st.write("### Mean Absolute Error (MAE)")
            st.write("The MAE is the average absolute difference between the actual and predicted values. "
             "Here, the MAE shows how far off our predictions are in terms of dollars on average.")
            mae = metrics.mean_absolute_error(y_test, y_predicted)
            st.write(f"Mean Absolute Error: {mae:.2f}")

            st.write("### Mean Squared Error (MSE)")
            st.write("MSE is the average of the squared differences between actual and predicted values. "
             "It penalizes larger errors more heavily.")
            mse = metrics.mean_squared_error(y_test, y_predicted)
            st.write(f"Mean Squared Error: {mse:.2f}")

            st.write("### Root Mean Squared Error (RMSE)")
            st.write("RMSE is the square root of the MSE, bringing the error back to the original unit scale.")
            rmse = np.sqrt(mse)
            st.write(f"Root Mean Squared Error: {rmse:.2f}")

            st.write("### Explained Variance Score")
            st.write("This metric tells us how much of the variance in MonthlyIncome is explained by our model. "
             "It ranges from 0 to 1, with 1 being perfect prediction.")
            eV = round(sm.explained_variance_score(y_test, y_predicted), 2)
            st.write(f"Explained Variance Score: {eV}")

            st.write("### R-squared (R²) Score")
            st.write("R² measures how well the model explains variability in the target. "
             "Higher R² means better fit.")
            r2 = r2_score(y_test, y_predicted)
            st.write(f"R² Score: {r2:.2f}")

            st.write("### Visualizing the Regression Results")
            st.write("This scatter plot shows how predicted incomes compare to actual values. "
             "Points closer to the diagonal line indicate better predictions.")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_predicted, color='blue')
            ax.set_title("Multiple Linear Regression")
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predictions")
            st.pyplot(fig)

            st.write("As seen in the scatter plot, predictions are somewhat scattered around the ideal line, "
             "which reflects the R² score — the model explains about 74% of the variance.")

        elif stage == "Classification":
            if "cleaned_df" not in st.session_state:
                st.error("Please run Data wrangling first to generate the cleaned dataset.")
            else:
                import seaborn as sns
                from sklearn import tree, model_selection
                from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.preprocessing import LabelEncoder

                df = st.session_state["cleaned_df"].copy()

                st.write("""
                    In this task we want to try and use `DecisionTreeClassifier` to predict the attrition of an employee.
                    We'll use the cleaned dataset, perform label encoding so we can create a correlation matrix and remove
                    columns that have too small of a correlation with `Attrition` to matter.

                    We then use one hot encoding to convert categorical values to numbers while avoiding arbitrary ranking.
                    This data is then split into a training set and a test set.""")


                # Re-encode Attrition
                st.markdown("### Re-encode Attrition Column")
                df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
                st.dataframe(df.head())

                # Label Encoding
                st.markdown("### Label Encoding (for Correlation)")
                data_column_category = df.select_dtypes(exclude=[np.number]).columns
                label_encoder = LabelEncoder()
                df_label_encoded = df.copy()
                for i in data_column_category:
                    df_label_encoded[i] = label_encoder.fit_transform(df_label_encoded[i])
                st.dataframe(df_label_encoded.head())

                # Correlation Matrix
                st.markdown("### Correlation Matrix")
                corr_matrix = df_label_encoded.corr()['Attrition']
                st.write(corr_matrix)

                cols_to_remove = corr_matrix[corr_matrix.abs() < 0.1].index.tolist()
                st.write(f"Columns to be removed due to low correlation: {cols_to_remove}")
                df = df.drop(columns=cols_to_remove)

                # One-Hot Encoding
                st.markdown("### One-Hot Encoding")
                data_column_category = df.select_dtypes(exclude=[np.number]).columns
                df_onehot_getdummies = pd.get_dummies(df[data_column_category], prefix=data_column_category, dtype=int)
                data_column_number = df.select_dtypes(include=[np.number]).columns
                df_onehot_encoded = pd.concat([df_onehot_getdummies, df[data_column_number]], axis=1)

                cols = [col for col in df_onehot_encoded.columns if col != 'Attrition'] + ['Attrition']
                df_onehot_encoded = df_onehot_encoded[cols]

                st.dataframe(df_onehot_encoded.head())

                # Split data
                X, y = df_onehot_encoded.iloc[:, :-1], df_onehot_encoded['Attrition']
                X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=7)

                # Train model
                st.markdown("### Training Decision Tree Classifier")
                classifier = DecisionTreeClassifier(max_depth=5)
                classifier.fit(X_train, y_train)

                # Visualize tree
                st.markdown("### Decision Tree Visualization")
                dot_data = tree.export_graphviz(
                    classifier, out_file=None,
                    feature_names=X.columns,
                    class_names=["No Attrition", "Attrition"],
                    filled=True, rounded=True,
                    special_characters=True
                )
                st.graphviz_chart(dot_data)

                # Evaluate model
                st.markdown("### Evaluation Metrics")
                y_testp = classifier.predict(X_test)
                acc = accuracy_score(y_test, y_testp)
                st.write("**Accuracy:**", acc)

                conf_matrix = confusion_matrix(y_test, y_testp)
                st.write("**Confusion Matrix:**")
                st.write(conf_matrix)

                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.markdown("### Confusion Matrix Summary")
                st.markdown("""
                    Out of the employees who actually stayed (class 0):
                    - 174 were correctly predicted to stay (true negatives)
                    - 10 were incorrectly predicted to leave (false positives)

                    Out of the employees who actually left (class 1):
                    - 8 were correctly predicted to leave (true positives)
                    - 23 were missed and predicted to stay (false negatives)

                    This means the model is good at identifying employees who will stay but struggles to detect those who will leave.""")

                st.markdown("### Classification Reports")
                class_names = ['No Attrition', 'Attrition']

                st.write("**Training Set Performance**")
                train_report = classification_report(y_train, classifier.predict(X_train), target_names=class_names, output_dict=True)
                st.dataframe(pd.DataFrame(train_report).transpose())

                st.write("**Test Set Performance**")
                test_report = classification_report(y_test, classifier.predict(X_test), target_names=class_names, output_dict=True)
                st.dataframe(pd.DataFrame(test_report).transpose())

                st.markdown("### Summary")
                st.markdown("""
                    Based on the evaluation metrics, the classifier performs well at identifying employees who will stay with the company,
                    achieving high precision and recall for the "No Attrition" class on both training and test data.

                    However, the model struggles to correctly identify employees who will leave ("Attrition"), with much lower recall and f1-score for this class—especially on the test set, where recall drops to 0.26. This suggests the model misses many actual attrition cases.

                    Overall, while the model is accurate for predicting retention, it is less effective at detecting attrition. Improving the model’s ability to identify employees at risk of leaving—perhaps by addressing class imbalance or tuning model parameters—could make it more useful for attrition prediction tasks.""")

        elif stage == "Clustering":
            if "cleaned_df" not in st.session_state:
                st.error("Please run Data wrangling first to generate the cleaned dataset.")
            else:
                df = st.session_state["cleaned_df"].copy()

                st.title("Clustering Analysis")

                st.write("### Data Preparation")
                numeric_data = df.select_dtypes(include=['int64', 'float64'])
                st.write(f"Using {numeric_data.shape[1]} numeric features for clustering.")

                # Standardize data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)

                st.write("### Finding Optimal Number of Clusters with Silhouette Scores")

                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                import matplotlib.pyplot as plt

                sil_scores = []
                K = range(2, 11)
                for k in K:
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                    model.fit(scaled_data)
                    score = silhouette_score(scaled_data, model.labels_)
                    sil_scores.append(score)

                silhouette_df = pd.DataFrame({'K': list(K), 'Silhouette Score': sil_scores})
                st.dataframe(silhouette_df)

                # Plot silhouette scores
                fig, ax = plt.subplots()
                ax.plot(list(K), sil_scores, 'bx-')
                ax.set_xlabel('Number of clusters (K)')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Score for Different K')
                st.pyplot(fig)

                best_k = K[sil_scores.index(max(sil_scores))]
                st.write(f"**Best number of clusters:** {best_k} (Silhouette Score: {max(sil_scores):.4f})")

                # Fit final model and assign cluster labels
                final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
                df['Cluster'] = final_kmeans.fit_predict(scaled_data)

                st.write("### PCA Visualization of Clusters")

                from sklearn.decomposition import PCA
                import seaborn as sns

                pca = PCA(n_components=2, random_state=42)
                pca_components = pca.fit_transform(scaled_data)

                pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = df['Cluster'].astype(str)  # as string for color hue

                fig2, ax2 = plt.subplots(figsize=(10, 7))
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=60, ax=ax2)
                ax2.set_title('PCA Visualization of Clusters')
                ax2.set_xlabel('Principal Component 1')
                ax2.set_ylabel('Principal Component 2')
                ax2.legend(title='Cluster')
                ax2.grid(True)
                plt.tight_layout()
                st.pyplot(fig2)

                st.write("""
                The scatterplot above uses PCA (Principal Component Analysis) to project high-dimensional employee data 
                down to two dimensions, allowing us to visualize the clusters more intuitively. 
                Each point is an employee, and each color represents the cluster they belong to. 
                While there is some overlap between clusters (indicating similarities), 
                you can still observe distinct groupings. This tells us that the KMeans model found 
                real patterns in employee characteristics, such as tenure, income, and job level.
                """)

                st.write("### Cluster Profiles (Mean Values by Cluster)")

                cluster_profile = df.groupby('Cluster').mean(numeric_only=True).T

                fig3, ax3 = plt.subplots(figsize=(14, 10))
                sns.heatmap(cluster_profile, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, ax=ax3)
                ax3.set_title("Cluster Profiles by Feature (Mean Values)")
                ax3.set_xlabel("Cluster")
                ax3.set_ylabel("Feature")
                plt.tight_layout()
                st.pyplot(fig3)

                st.write("""
                This heatmap shows the average value of each numeric feature for every cluster. 
                It helps us interpret what differentiates the groups.

                - **Cluster 0, (left hand side) ** tends to include employees with **higher MonthlyIncome**, **more TotalWorkingYears**, 
                and **higher JobLevel**. This suggests these are more **experienced or senior staff**.
                - **Cluster 1, (right side)**, on the other hand, has **younger employees**, with **lower income**, and **less tenure** — 
                likely **junior or newer hires**.

                Interestingly, features like **JobSatisfaction**, **WorkLifeBalance**, and **EnvironmentSatisfaction** 
                remain fairly consistent between clusters, indicating that the work environment is perceived similarly 
                regardless of seniority. This could be a good sign for organizational culture, and indicating a general job satisfaction
                at the company across the board.

                Overall, this clustering can help HR or business leaders tailor programs:
                - Cluster 0 might be suitable for leadership or mentoring roles.
                - Cluster 1 could benefit from more support, training, or engagement initiatives to at the very least improve retention.
                """)

