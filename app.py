import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error, mean_absolute_percentage_error

# --- Page Config ---
st.set_page_config(page_title="WGS Reactor Digital Twin", layout="wide", page_icon="🏭")

# --- Custom CSS ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .stSelectbox label { font-size: 16px; font-weight: bold; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 WGS Reactor Digital Twin & Optimizer Version 4.2.0")
st.markdown("Train a surrogate model, validate physics (Histograms, Learning Curves), and **optimize**.")
st.markdown("Developed by VeltEng")
# --- Sidebar: Configuration ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(df)} rows")
    
    # --- Sidebar: Feature Selection ---
    st.sidebar.divider()
    st.sidebar.header("2. Variables")
    
    all_cols = df.columns.tolist()
    
    # Smart Detection (Anti-Leakage)
    target_keywords = ['out', 'produced', 'release', 'dp', 'conversion', 'x_co_out']
    guess_targets = [c for c in all_cols if any(k in c.lower() for k in target_keywords)]
    if not guess_targets: guess_targets = [all_cols[-1]]
    guess_features = [c for c in all_cols if c not in guess_targets]
    
    feature_cols = st.sidebar.multiselect("Inputs (X)", all_cols, default=guess_features)
    target_col = st.sidebar.selectbox("Target (y)", all_cols, index=all_cols.index(guess_targets[0]) if guess_targets else 0)
    split_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # --- Sidebar: Advanced Model Selection ---
    st.sidebar.divider()
    st.sidebar.header("3. Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Algorithm",
        ["Random Forest", "Neural Network (MLP)", "Polynomial Regression", "Linear Regression"]
    )
    
    params = {}
    
    # --- Dynamic Hyperparameter Tuners ---
    if model_type == "Random Forest":
        st.sidebar.caption("Best for: Robustness, handling outliers.")
        with st.sidebar.expander("⚙️ Hyperparameters", expanded=True):
            params['n_estimators'] = st.slider("Trees (n_estimators)", 50, 500, 100)
            params['min_samples_leaf'] = st.slider("Min Samples per Leaf", 1, 20, 5, help="Increase to 5+ to prevent overfitting.")
            params['max_depth'] = st.slider("Max Tree Depth", 5, 100, 0, help="0 = Unlimited.")
            if params['max_depth'] == 0: params['max_depth'] = None

    elif model_type == "Neural Network (MLP)":
        st.sidebar.caption("Best for: Smooth physics, large datasets.")
        with st.sidebar.expander("⚙️ Hyperparameters", expanded=True):
            layers_opt = st.selectbox("Hidden Layers", ["(100, 50)", "(64, 64)", "(100, 100, 50)", "(32,)"])
            params['hidden_layer_sizes'] = eval(layers_opt)
            params['activation'] = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])
            params['alpha'] = st.number_input("Regularization (Alpha)", 0.0, 1.0, 0.0001, format="%.4f")
            params['max_iter'] = st.slider("Max Iterations", 200, 2000, 1000)

    elif model_type == "Polynomial Regression":
        st.sidebar.caption("Best for: Simple, interpretable physics.")
        with st.sidebar.expander("⚙️ Hyperparameters", expanded=True):
            params['degree'] = st.slider("Polynomial Degree", 1, 4, 2)

    # --- Main Content ---
    if st.button("🚀 Train Model", type="primary"):
        if not feature_cols:
            st.error("Select at least one input feature.")
        else:
            # Prepare Data
            X = df[feature_cols]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
            
            # Initialize Model
            model = None
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Polynomial Regression":
                model = make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(degree=params['degree']),
                    LinearRegression()
                )
            elif model_type == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=params['n_estimators'], 
                    min_samples_leaf=params['min_samples_leaf'],
                    max_depth=params['max_depth'],
                    random_state=42
                )
            elif model_type == "Neural Network (MLP)":
                model = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=params['hidden_layer_sizes'], 
                        activation=params['activation'],
                        alpha=params['alpha'],
                        max_iter=params['max_iter'],
                        random_state=42
                    )
                )
            
            # Train
            with st.spinner("Training Digital Twin..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)
                
            # Store model
            st.session_state['model'] = model
            st.session_state['feature_cols'] = feature_cols
            st.session_state['X_train'] = X_train

            # --- Metrics Dashboard ---
            st.markdown("### 📊 Performance Dashboard")
            
            r2 = r2_score(y_test, y_pred)
            r2_train = r2_score(y_train, y_train_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            max_err = max_error(y_test, y_pred)
            gap = r2_train - r2
            
            # Row 1: Health
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy (R²)", f"{r2:.4f}")
            c2.metric("Train Accuracy", f"{r2_train:.4f}")
            
            if gap > 0.10: status, color = "High Overfitting", "inverse"
            elif gap > 0.02: status, color = "Slight Overfitting", "off"
            else: status, color = "Good Fit", "normal"
            c3.metric("Overfitting Gap", f"{gap:.4f}", delta=status, delta_color=color)
            c4.metric("RMSE", f"{rmse:.4f}")

            # Row 2: Precision
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("MAE", f"{mae:.4f}")
            d2.metric("MAPE (%)", f"{mape:.2f}%")
            d3.metric("Max Error", f"{max_err:.4f}")
            d4.metric("Test Points", f"{len(y_test)}")

            # --- 4-Pack Diagnostics (Restored!) ---
            st.divider()
            st.subheader("📈 Diagnostics")
            
            tab1, tab2, tab3, tab4 = st.tabs(["1. Parity Plot", "2. Residual Plot", "3. Error Histogram", "4. Learning Curve"])
            
            with tab1:
                st.caption("Actual vs Predicted. Ideal = Red Line.")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                st.pyplot(fig)
            
            with tab2:
                st.caption("Residuals (Errors). Look for 'U-shapes' (bad physics) or 'Funnels' (bad variance).")
                residuals = y_test - y_pred
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                sc = ax2.scatter(y_pred, residuals, c=np.abs(residuals), cmap='coolwarm', alpha=0.6, edgecolors='k')
                ax2.axhline(0, color='r', linestyle='--')
                plt.colorbar(sc, ax=ax2, label='Abs Error')
                st.pyplot(fig2)

            with tab3: # Restored!
                st.caption("Error Distribution. Should look like a Bell Curve centered at 0.")
                fig3, ax3 = plt.subplots(figsize=(6, 3))
                ax3.hist(residuals, bins=30, color='purple', edgecolor='k', alpha=0.7)
                ax3.axvline(0, color='r', linestyle='--')
                st.pyplot(fig3)

            with tab4: # Restored!
                st.caption("Does more data help? Green line should rise to meet Red line.")
                try:
                    train_sizes = np.linspace(0.1, 1.0, 5)
                    train_sizes_abs, train_scores, test_scores = learning_curve(
                        model, X, y, train_sizes=train_sizes, cv=3, n_jobs=-1, scoring='r2'
                    )
                    fig4, ax4 = plt.subplots(figsize=(6, 3))
                    ax4.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'r-o', label="Train")
                    ax4.plot(train_sizes_abs, np.mean(test_scores, axis=1), 'g-o', label="Test")
                    ax4.legend()
                    st.pyplot(fig4)
                except Exception as e:
                    st.warning("Curve generation failed.")

    # --- OPTIMIZER SECTION ---
    if 'model' in st.session_state:
        st.divider()
        st.header("🧪 Process Optimizer")
        
        col_opt1, col_opt2 = st.columns([1, 2])
        
        with col_opt1:
            opt_goal = st.radio("Optimization Goal", ["Maximize Target", "Minimize Target"])
            
            st.subheader("Constraints")
            bounds = []
            
            for feat in st.session_state['feature_cols']:
                d_min = float(st.session_state['X_train'][feat].min())
                d_max = float(st.session_state['X_train'][feat].max())
                d_mean = float(st.session_state['X_train'][feat].mean())
                
                with st.expander(f"{feat}", expanded=False):
                    mode = st.radio(f"Mode", ["Vary", "Fix"], key=f"mode_{feat}")
                    if mode == "Vary":
                        min_v = st.number_input(f"Min", value=d_min, key=f"min_{feat}")
                        max_v = st.number_input(f"Max", value=d_max, key=f"max_{feat}")
                        bounds.append((min_v, max_v))
                    else:
                        val = st.number_input(f"Value", value=d_mean, key=f"fix_{feat}")
                        bounds.append((val, val))

        with col_opt2:
            if st.button("✨ Run Optimizer", type="primary"):
                model = st.session_state['model']
                feats = st.session_state['feature_cols']
                
                def objective(x):
                    input_df = pd.DataFrame([x], columns=feats)
                    pred = model.predict(input_df)[0]
                    return -pred if opt_goal == "Maximize Target" else pred

                x0 = [np.mean(b) for b in bounds]
                with st.spinner("Solving..."):
                    res = minimize(objective, x0, bounds=bounds, method='SLSQP')
                
                if res.success:
                    st.success("✅ Optimization Converged!")
                    final_val = -res.fun if opt_goal == "Maximize Target" else res.fun
                    st.metric(f"Optimal {target_col}", f"{final_val:.4f}")
                    
                    res_df = pd.DataFrame([res.x], columns=feats)
                    st.dataframe(res_df.style.highlight_max(axis=0))
                    st.download_button("Download Optimal Set", res_df.to_csv(index=False).encode(), "optimal.csv")
                else:
                    st.error(f"Failed: {res.message}")

else:
    st.info("👈 Upload data to enable the Digital Twin.")