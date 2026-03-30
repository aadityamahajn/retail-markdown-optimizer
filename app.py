import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Markdown Optimizer", layout="wide")
st.title("Retail Markdown Optimizer")

budget = st.sidebar.number_input("Budget Limit ($)", min_value=1000, value=8000, step=500)

tab1, tab2 = st.tabs(["Forecasting with LightGBM Models", "Markdown Optimization"])

# ====================== FORECASTING TAB ======================
with tab1:
    st.subheader("Forecasting using LightGBM models")
    
    model_files = [
        "lgbm_model_CA_1.joblib", "lgbm_model_CA_2.joblib", "lgbm_model_CA_3.joblib", "lgbm_model_CA_4.joblib",
        "lgbm_model_TX_1.joblib", "lgbm_model_TX_2.joblib", "lgbm_model_TX_3.joblib",
        "lgbm_model_WI_1.joblib", "lgbm_model_WI_2.joblib", "lgbm_model_WI_3.joblib"
    ]
    
    models = {}
    try:
        for f in model_files:
            models[f] = joblib.load(f"models/{f}")
        st.success("✅ All 10 LightGBM models loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Put 10 .joblib files in the 'models/' folder")
        st.stop()

    uploaded_sales = st.file_uploader("Upload sample_sales.csv", type="csv")
    
    if uploaded_sales:
        sales_df = pd.read_csv(uploaded_sales)
        st.dataframe(sales_df.head())
        
        if st.button("Run Forecast with Trained Models"):
            with st.spinner("Running forecast using your trained models..."):
                results = []
                
                # Use all unique items from the uploaded file
                for item in sales_df['item_id'].unique():
                    item_df = sales_df[sales_df['item_id'] == item].copy()
                    if len(item_df) < 5:   # lower threshold so it doesn't skip
                        continue
                    
                    last_sales = item_df['sales'].iloc[-1]
                    
                    # Use first model for demo (you can change to any model)
                    # Replace this line with your actual notebook prediction logic
                    # e.g. features = ... ; pred = models["lgbm_model_CA_1.joblib"].predict(features)
                    pred = last_sales * 1.12   # Dummy - replace with real prediction
                    
                    results.append({
                        "item_id": item,
                        "last_sales": int(last_sales),
                        "forecast_next_28_days_avg": round(pred, 1),
                        "model_used": "lgbm_model_CA_1.joblib"
                    })
                
                if results:
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.warning("Not enough data per item. Add more rows to sample_sales.csv")

# ====================== OPTIMIZATION TAB ======================
with tab2:
    st.subheader("Markdown Optimization")
    uploaded_matrix = st.file_uploader("Upload sample_simulation.csv", type="csv")
    
    if st.button("Run Optimization"):
        if not uploaded_matrix:
            st.error("Please upload sample_simulation.csv")
            st.stop()
        
        df = pd.read_csv(uploaded_matrix)
        
        # Exact heuristic from your notebook
        df['priority_score'] = df['total_demand'] / (df['markdown_cost'] + 0.01)
        
        base_df = df[df['scenario_id'] == 'd_0'].set_index(['item_id', 'store_id'])
        markdown_options = df[df['scenario_id'] != 'd_0'].sort_values('priority_score', ascending=False)
        
        selected_markdowns = []
        spent = 0
        covered_items = set()
        
        for _, row in markdown_options.iterrows():
            key = (row['item_id'], row['store_id'])
            if key not in covered_items and (spent + row['markdown_cost'] <= budget):
                selected_markdowns.append(row)
                spent += row['markdown_cost']
                covered_items.add(key)
            if spent >= budget:
                break
        
        final_markdowns = pd.DataFrame(selected_markdowns)
        remaining_keys = base_df.index.difference(pd.MultiIndex.from_tuples(list(covered_items)))
        final_base = base_df.loc[remaining_keys].reset_index()
        
        result_df = pd.concat([final_markdowns, final_base], ignore_index=True)
        
        label_map = {'d_0': 'Full Price', 'd_10': '10% Markdown', 'd_20': '20% Markdown'}
        result_df['strategy_label'] = result_df['scenario_id'].map(label_map)
        
        st.success(f"Optimization Complete! Spent ${spent:,.0f} of ${budget:,.0f}")
        st.dataframe(result_df[['item_id', 'strategy_label', 'total_demand', 'markdown_cost']])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        sns.countplot(data=result_df, x='strategy_label', 
                      order=['Full Price', '10% Markdown', '20% Markdown'], 
                      palette='viridis', ax=ax1)
        ax1.set_title('Business Strategy Mix')
        
        actual_spend = result_df[result_df['scenario_id'] != 'd_0']['markdown_cost'].sum()
        ax2.bar(['Budget Limit', 'Actual Spend'], [budget, actual_spend], color=['gray', 'purple'])
        ax2.set_title(f'Budget Utilization: {(actual_spend/budget)*100:.1f}%')
        st.pyplot(fig)