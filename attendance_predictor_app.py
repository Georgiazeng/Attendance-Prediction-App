import streamlit as st
import pandas as pd
import numpy as np
import datetime
from joblib import load
from streamlit import column_config

# --- LOAD MODEL & PARAMS ---
model = load('sigmpoid_logdays_model_params_2025-07-14.joblib')
sigmoid_parameters = pd.read_csv('sigmoid_parameters.csv')

multipliers = pd.DataFrame({
    'range': [(0, 100), (100, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, np.inf)],
    'f1': [2.4, 1.6, 1.5, 1.3, 1.3, 1.1],
    'f2': [3.1, 2.0, 1.7, 1.4, 1.3, 1.2],
    'f3': [3.1, 2.1, 1.8, 1.5, 1.4, 1.2],
    'f4': [3.5, 2.2, 2.0, 1.5, 1.5, 1.3],
    'f5': [3.8, 2.5, 2.2, 1.6, 1.5, 1.3]
})


# --- FUNCTIONS ---
def sigmoid_search(x):
    L  = sigmoid_parameters['L'].iloc[0]
    x0 = sigmoid_parameters['x0'].iloc[0]
    k  = sigmoid_parameters['k'].iloc[0]
    return L / (1 + np.exp(-k * (x - x0)))

def hybrid_log_linear(x, k_high):
    k_low = np.percentile(x, 5)
    x_clipped = np.clip(x, 0, k_high)
    tail = np.where(x > k_high, ((x - k_high) / k_high), 0)
    return x_clipped + tail

def get_multiplier(volume, years):
    for idx, row in multipliers.iterrows():
        low, high = row['range']
        if low < volume <= high:
            return row[f'f{years}']
    return 1.0


# --- INIT SESSION ---
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = pd.DataFrame(columns=[
        "Date", "Exhibition", "Keywords", "Weights", "Volumes",
        "Gallery", "Days", "Adjusted Search Volume", "RA Factor",
        "Predicted Attendance", "Selected"
    ])
kw_data = pd.DataFrame([
        {"Keyword": "", "Volume": 0, "Weight": 0.0, "GrowthYears": 0}
        for _ in range(5)
    ])
if "keywords_table_data" not in st.session_state:
    st.session_state.keywords_table_data = kw_data.copy()
# --- FORM ---
st.title("🎨 Exhibition Attendance Predictor")

with st.form("input_form"):
    st.subheader("📥 Input Exhibition Info")

    exhibition = st.text_input("Exhibition Title", value="Exhibition Title")

    # Keyword input via data_editor
    st.write("🔑 Keyword Table (enter up to 5)")
    

    keyword_df = st.data_editor(
    st.session_state.keywords_table_data,
    key="keywords_table",
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Keyword": column_config.TextColumn(),
        "Volume": column_config.NumberColumn(format="%.0f"),
        "Weight": column_config.NumberColumn(format="%.2f", step=0.01, min_value=0.0, max_value=1.0),
        "GrowthYears": column_config.NumberColumn(format="%d", min_value=0, max_value=5)
    })

    gallery_type = st.selectbox("Gallery Type", ["Main", "Sackler", "GJW"])
    days = st.slider("Exhibition Duration (days)", 30, 180, 90)
    era = st.selectbox("Era", ["Post-COVID", "COVID", "Pre-COVID"])
    ra_drive = st.slider("RA Drive (%)", 0, 20, 0)

    submitted = st.form_submit_button("✅ Confirm and Add")



# --- ON SUBMIT ---
if submitted:
    # Clean keyword entries
    valid_keywords = keyword_df[keyword_df["Keyword"].str.strip() != ""]
    keywords = valid_keywords.to_dict("records")
    # Ensure correct dtypes
    valid_keywords["Volume"] = valid_keywords["Volume"].astype(float)
    valid_keywords["Weight"] = valid_keywords["Weight"].astype(float)
    valid_keywords["GrowthYears"] = valid_keywords["GrowthYears"].astype(int)
    adjusted_volume = sum(
    float(row["Volume"]) * get_multiplier(float(row["Volume"]), int(row["GrowthYears"])) * float(row["Weight"])
    if float(row["GrowthYears"]) > 0 else float(row["Volume"]) * float(row["Weight"])
    for _, row in valid_keywords.iterrows() 
)

    st.write(f"🔍 Adjusted Search Volume: {adjusted_volume:.2f}")
    
    gallery_code = 0 if "Main" in gallery_type else 1
    era_code = {"Post-COVID": 1, "COVID": 0, "Pre-COVID": 2}[era]
    log_days = np.log1p(days)
    log_days_saturated = hybrid_log_linear(np.array([log_days]), k_high=np.log1p(90))[0]
    search_bounded_val = sigmoid_search(adjusted_volume)

    log_attendance = model.predict(pd.DataFrame({
        'sigmoid_search_volume_with_growth': [search_bounded_val],
        'log_saturating_days': [log_days_saturated],
        'era_encoded': [era_code],
        'gallery_type_encoded': [gallery_code]
    }))

    attendance = np.exp(log_attendance)[0]
    attendance_final = attendance * (1 + ra_drive / 100)

    # Store
    new_row = {
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Exhibition": exhibition,
        "Keywords": ", ".join([kw["Keyword"] for kw in keywords]),
        "Volumes": ", ".join([str(kw["Volume"]) for kw in keywords]),
        "Weights": ", ".join([str(kw["Weight"]) for kw in keywords]),
        "Gallery": gallery_type,
        "Days": days,
        "Adjusted Search Volume": round(adjusted_volume, 2),
        "RA Factor": ra_drive,
        "Predicted Attendance": int(attendance_final),
        "Selected": False
    }

    st.session_state.prediction_data = pd.concat([
        st.session_state.prediction_data,
        pd.DataFrame([new_row])
    ], ignore_index=True)

    st.success(f"🎯 Final Predicted Attendance (with RA): {attendance_final:,.0f} visitors")
    st.toast("Inputs cleared — ready for new entry.", icon="🧹")


# --- DISPLAY TABLE ---
st.subheader("📊 Predictions Table")
edited_df = st.data_editor(
    st.session_state.prediction_data,
    column_config={"Selected": st.column_config.CheckboxColumn(required=False)},
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic"
)
st.session_state.prediction_data = edited_df

# --- CLEAR & DOWNLOAD ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear All"):
        st.session_state.prediction_data = pd.DataFrame(columns=st.session_state.prediction_data.columns)
        st.success("Cleared.")

with col2:
    selected_rows = st.session_state.prediction_data[st.session_state.prediction_data["Selected"]]
    if not selected_rows.empty:
        csv = selected_rows.drop(columns="Selected").to_csv(index=False)
        st.download_button(
            label="📥 Download Selected",
            data=csv,
            file_name="selected_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("Select at least one row to download.")
