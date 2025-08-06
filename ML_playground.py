import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# =============================================
# Functions
# =============================================

# Handling missing values
def fill_missing(df, method='mean'):
    df_filled = df.copy()
    for col in df_filled.select_dtypes(include=[np.number]).columns:
        if df_filled[col].isnull().sum() > 0:
            if method == "mean":
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            elif method == "median":
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            elif method == "mode":
                df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
    return df_filled

# =============================================
# Setting default data (Kaggle Heart Disease)
# =============================================
@st.cache_data
def load_default_data():
    df = fetch_openml(name="heart-disease", version=1, as_frame=True).frame
    return df

# =============================================
# Main
# =============================================
st.set_page_config(page_title="ML Demo", layout="wide")
st.title("🤖 Machine Learning Playground 🤖")

# Sidebar
st.sidebar.header("Settings")

## Loading CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
## Reset button
# with st.sidebar:
#     st.markdown("---")
#     if st.button("🔄 Reset All", type="primary"):
#         if st.confirm("Are you seriously doing it?"):
#             keys_to_keep = ["df"]
#             for key in list(st.session_state.keys()):
#                 if key not in keys_to_keep:
#                     del st.session_state[key]
#             st.experimental_rerun()

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=0)
    dataset_name = "input data"
else:
    df = load_default_data()
    dataset_name = "default data"
df.reset_index(drop=True, inplace=True)

# Main display
# st.write("# 1.Check Input Dataset")
st.markdown('<p style="font-family:cursive; color:white; font-size: 28px; background-color: #008080;">1. Check Input Dataset</p>', unsafe_allow_html=True)
st.subheader(f"Dataset in use")
st.write(f"Sample size ---> rows: {df.shape[0]} × cols: {df.shape[1]}" )

# View dataset
col1, col2 = st.columns(2)

with col1:
    st.write("### Preview the dataset")
    st.dataframe(df.head(10))

# Check data types
with col2:
    st.write("### Data types & Missing count")
    dtype_info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str),
        "Missing": df.isnull().sum().values
    })
    st.dataframe(dtype_info.reset_index(drop=True))

# Visualize
st.write("### Missing data")
missing_counts = df.isnull().sum()
st.bar_chart(missing_counts)

# =============================================
# EDA
# =============================================
# st.write("# 2. EDA")
st.markdown('<p style="font-family:cursive; color:white; font-size: 28px; background-color: #008080;">3. EDA</p>', unsafe_allow_html=True)

# --- Select a method ---
eda_options = ["Histogram", "Barplot", "Pairplot"]
eda_choice = st.selectbox("Select a method", eda_options)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --- Histogram
if eda_choice == "Histogram":
    selected_num_col = st.selectbox("Select the feature", numeric_cols)
    fig = px.histogram(df, x=selected_num_col, nbins=20, marginal="violin")
    st.plotly_chart(fig, use_container_width=True)

# --- Barplot
elif eda_choice == "Barplot":
    if categorical_cols:
        selected_cat_col = st.selectbox("Select the feature", categorical_cols)
        fig, ax = plt.subplots(figsize=(4, 3))
        df[selected_cat_col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("There are no categorical features")

# --- Pairplot
elif eda_choice == "Pairplot":
    if len(numeric_cols) > 1:
        with st.spinner("Generating pairplot..."):
            pairplot = px.scatter_matrix(df[numeric_cols], dimensions=numeric_cols)
            pairplot.update_layout(width=800, height=800)
            st.plotly_chart(pairplot, use_container_width=False)
    else:
        st.info("Not enough numeric features for pairplot")


# =============================================
# Preprocessing
# =============================================
st.markdown('<p style="font-family:cursive; color:white; font-size: 28px; background-color: #008080;">3. Preprocessing</p>', unsafe_allow_html=True)

if "df_filled" not in st.session_state:
    st.session_state["df_filled"] = None

preprocess_cols1, preprocess_cols2 = st.columns(2)

# --- Handling missing data
with preprocess_cols1:
    st.subheader("Numeric Missing Value Handling")
    fill_method = st.selectbox(
        "Select a method for missing data",
        ["mean", "median", "mode"],
        index=0
    )

# --- Encoding categorical features
with preprocess_cols2:
    st.subheader("Categorical Encoding")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        encoding_method = st.selectbox(
            "Select encoding method for categorical variables",
            ["Label Encoding", "One-Hot Encoding"],
            index=0
        )
    else:
        st.info("No categorical columns detected.")
        encoding_method = "None (No encoding)"

# --- Exec
if st.button("Exec --> Preprocessing"):
    df_filled = df.copy()
    target_col = "target" if "target" in df.columns else None
    features_cat = [col for col in cat_cols if col != target_col]

    # --- Handling missing data
    df_filled = fill_missing(df_filled, method=fill_method)
    # --- Encoding categorical features
    if encoding_method != "None (No encoding)" and features_cat:
        if encoding_method == "Label Encoding":
            for col in features_cat:
                le = LabelEncoder()
                df_filled[col] = le.fit_transform(df_filled[col].astype(str))
        elif encoding_method == "One-Hot Encoding":
            df_filled = pd.get_dummies(df_filled, columns=features_cat, drop_first=True)

    # --- Save to session state
    st.session_state["df_filled"] = df_filled

    # --- Completed 
    st.success("Preprocessing completed successfully!")
    st.write("Processed DataFrame:")
    st.dataframe(df_filled.head())

# =============================================
# Machine Learning
# =============================================
st.markdown('<p style="font-family:cursive; color:white; font-size: 28px; background-color: #008080;">4. Model Training and Predict target</p>', unsafe_allow_html=True)

def needs_preprocessing(df):
    has_categorical = len(df.select_dtypes(exclude=[np.number]).columns) > 0
    return has_categorical

if needs_preprocessing(df):
    if "df_filled" not in st.session_state or st.session_state["df_filled"] is None:
        st.error("⚠️ Please preprocess categorical columns before training.")
        st.stop()
    else:
        df_filled = st.session_state["df_filled"]
else:
    df_filled = st.session_state["df_filled"] if st.session_state["df_filled"] is not None else df.copy()

split_cols1 = st.columns(3)
select_task = split_cols1[0].selectbox(
    "Select the task",
    ["Classification(binary)", "Regression"],
    index=0
)
model_choices = split_cols1[1].multiselect(
    "Select models",
    ["LightGBM", "XGBoost", "CatBoost"],
    default=["LightGBM"]
)
select_target = split_cols1[2].selectbox(
    "TARGET", df_filled.columns, index=list(df_filled.columns).index("target") if "target" in df_filled.columns else 0
)
# if st.session_state["df_filled"] is not None:
#     df_filled = st.session_state["df_filled"]
#     select_target = split_cols1[2].selectbox(
#         "TARGET", df_filled.columns, index=list(df_filled.columns).index("target") if "target" in df_filled.columns else 0
#     )
# else:
#     split_cols1[2].warning("⚠️ Run preprocessing first")
#     select_target = None

split_cols2 = st.columns(2)
test_size = split_cols2[0].slider("Validation data size(ratio):", 0.1, 0.5, 0.2)
random_seed = split_cols2[1].number_input("Random seed:", min_value=0, step=1, value=42)

X = df_filled.drop(columns=select_target)
y = df_filled[select_target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

results = []
roc_curves = []
pred_vs_actual = []

if st.button("Exec --> Model training"):
    for model_choice in model_choices:
        # --- Selecting models
        if select_task == "Classification(binary)":
            if "LightGBM" in model_choice: model = LGBMClassifier(random_state=random_seed)
            elif "XGBoost" in model_choice: model = XGBClassifier(eval_metric='logloss', random_state=random_seed)
            else: model = CatBoostClassifier(verbose=0, random_state=random_seed)
        else:
            if "LightGBM" in model_choice: model = LGBMRegressor(random_state=random_seed)
            elif "XGBoost" in model_choice: model = XGBRegressor(random_state=random_seed)
            else: model = CatBoostRegressor(verbose=0, random_state=random_seed)

        # --- Training
        model.fit(X_train, y_train)

        # --- Evaluation
        if select_task == "Classification(binary)":
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            results.append({"Model": model_choice, "Metric": "AUC", "Score": auc})

            # ROC
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_curves.append((model_choice, fpr, tpr, auc))
        else:
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            results.append({"Model": model_choice, "Metric": "MAE", "Score": mae})
            pred_vs_actual.append((model_choice, y_val, y_pred))

    # --- Visualization
    st.write("### Model Performance")
    eval_cols = st.columns(2)
    results_df = pd.DataFrame(results)

    if select_task == "Classification(binary)" and roc_curves:
        # Bar plot
        fig_bar = px.bar(
            results_df, x="Model", y="Score", color="Model",
            text=results_df["Score"].round(3)
        )
        fig_bar.update_layout(
            yaxis=dict(title="AUC", range=[0.5, 1.0]),
            xaxis_title="Model", showlegend=False
        )
        fig_bar.update_traces(textposition='outside')
        eval_cols[0].write("#### AUC Scores")
        eval_cols[0].plotly_chart(fig_bar, use_container_width=False)

        # ROC
        fig_roc = go.Figure()
        for model_name, fpr, tpr, auc in roc_curves:
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{model_name} (AUC={auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name="Random"))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        eval_cols[1].write("#### ROC curve")
        eval_cols[1].plotly_chart(fig_roc, use_container_width=False)

    elif select_task == "Regression":
        # Bar plot
        fig_bar = px.bar(
            results_df, x="Model", y="Score", color="Model",
            text=results_df["Score"].round(3)
        )
        fig_bar.update_layout(yaxis=dict(title="MAE"), xaxis_title="Model", showlegend=False)
        fig_bar.update_traces(textposition='outside')
        eval_cols[0].write("#### MAE Scores")
        eval_cols[0].plotly_chart(fig_bar, use_container_width=False)

        # Pred vs Actual
        fig_scatter = go.Figure()
        for model_name, y_true, y_pred in pred_vs_actual:
            fig_scatter.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name=model_name, opacity=0.6))
        fig_scatter.add_trace(go.Scatter(
            x=[y_val.min(), y_val.max()], y=[y_val.min(), y_val.max()],
            mode="lines", line=dict(dash="dash", color="red"), name="Ideal"
        ))
        fig_scatter.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
        eval_cols[1].write("#### Predicted vs Actual")
        eval_cols[1].plotly_chart(fig_scatter, use_container_width=False)
