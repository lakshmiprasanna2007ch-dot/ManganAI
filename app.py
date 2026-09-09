# ============================================================
# ManganAI - COMPLETE STREAMLIT FRONTEND
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ManganAI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 3. ATTRACTIVE FRONTEND STYLE
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 0;
}

.sub-title {
    font-size: 19px;
    opacity: 0.75;
    margin-bottom: 20px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 12px;
}

.kpi {
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
    min-height: 120px;
}

.kpi-title {
    font-size: 14px;
    opacity: 0.70;
}

.kpi-value {
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
}

.feature-card {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 175px;
}

.result-card {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. FIND MODEL FILES
# ============================================================

def find_model(filename):

    possible_paths = [
        MODEL_DIR / filename,
        BASE_DIR / filename
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


LOCATION_MODEL_PATH = find_model(
    "exploration_model.pkl"
)

PRODUCTION_MODEL_PATH = find_model(
    "production_model.pkl"
)


# ============================================================
# 5. LOAD MODELS
#
# IMPORTANT:
# This version DOES NOT hide the real error.
# ============================================================

@st.cache_resource
def load_location_model():

    if LOCATION_MODEL_PATH is None:
        return None, "exploration_model.pkl was not found."

    try:
        model = joblib.load(
            LOCATION_MODEL_PATH
        )

        return model, None

    except Exception as e:

        return None, (
            f"Could not load exploration_model.pkl: "
            f"{type(e).__name__}: {e}"
        )


@st.cache_resource
def load_production_model():

    if PRODUCTION_MODEL_PATH is None:
        return None, "production_model.pkl was not found."

    try:
        model = joblib.load(
            PRODUCTION_MODEL_PATH
        )

        return model, None

    except Exception as e:

        return None, (
            f"Could not load production_model.pkl: "
            f"{type(e).__name__}: {e}"
        )


location_model, location_error = load_location_model()

production_model, production_error = load_production_model()


# ============================================================
# 6. GET MODEL FEATURES
# ============================================================

def get_features(model):

    if model is None:
        return []

    if hasattr(model, "feature_names_in_"):

        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    # Some models contain an estimator inside a pipeline
    if hasattr(model, "named_steps"):

        for _, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):

                try:
                    return list(step.feature_names_in_)
                except Exception:
                    pass

    return []


location_features = get_features(
    location_model
)

production_features = get_features(
    production_model
)


# ============================================================
# 7. LOAD DASHBOARD DATA
# ============================================================

def load_dashboard_data():

    if not DATA_DIR.exists():
        return None

    try:

        csv_files = list(
            DATA_DIR.glob("*.csv")
        )

        if not csv_files:
            return None

        # Prefer largest CSV
        csv_files = sorted(
            csv_files,
            key=lambda x: x.stat().st_size,
            reverse=True
        )

        df = pd.read_csv(
            csv_files[0]
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        return df

    except Exception:
        return None


dashboard_df = load_dashboard_data()


# ============================================================
# 8. SAFE NUMERIC CONVERSION
# ============================================================

def numeric_average(df, column):

    if df is None:
        return None

    if column not in df.columns:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return None

    return float(values.mean())


# ============================================================
# 9. DEFAULT INPUT VALUES
# ============================================================

def default_value(feature):

    name = str(feature).lower().strip()

    if "latitude" in name:
        return 20.50

    if "longitude" in name:
        return 79.50

    if "elevation" in name:
        return 500.0

    if "magnetic" in name:
        return 50.0

    if "density" in name:
        return 2.70

    if "geological" in name and "score" in name:
        return 0.70

    if "ndvi" in name:
        return 0.40

    if "vegetation" in name:
        return 0.40

    if "rainfall" in name:
        return 130.0

    if "soil" in name and "moisture" in name:
        return 45.0

    if "temperature" in name:
        return 32.0

    if "band" in name:
        return 0.20

    if name == "ei":
        return 400.0

    if "target" in name and "production" in name:
        return 10000.0

    if "previous" in name and "production" in name:
        return 9000.0

    if "production" in name:
        return 9000.0

    if "downtime" in name:
        return 10.0

    if "availability" in name:
        return 85.0

    if "blasting" in name and "delay" in name:
        return 10.0

    if "ore" in name and "grade" in name:
        return 4.0

    return 0.0


# ============================================================
# 10. CREATE MODEL INPUTS
# ============================================================

def create_inputs(features, prefix):

    values = {}

    if not features:
        return values

    cols = st.columns(2)

    for i, feature in enumerate(features):

        with cols[i % 2]:

            values[feature] = st.number_input(
                str(feature),
                value=float(
                    default_value(feature)
                ),
                key=f"{prefix}_{feature}"
            )

    return values


# ============================================================
# 11. DASHBOARD
# ============================================================

def show_dashboard():

    st.markdown(
        '<div class="main-title">⛏️ ManganAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'AI-Assisted Manganese Exploration & Production System'
        '</div>',
        unsafe_allow_html=True
    )

    st.success(
        "🟢 ManganAI System Ready"
    )

    # --------------------------------------------------------
    # KPI DATA
    # --------------------------------------------------------

    production = numeric_average(
        dashboard_df,
        "Production"
    )

    target_production = numeric_average(
        dashboard_df,
        "Target Production"
    )

    rainfall = numeric_average(
        dashboard_df,
        "Rainfall"
    )

    soil_moisture = numeric_average(
        dashboard_df,
        "Soil Moisture"
    )

    equipment_availability = numeric_average(
        dashboard_df,
        "Equipment Availability"
    )

    equipment_downtime = numeric_average(
        dashboard_df,
        "Equipment Downtime"
    )

    blasting_delay = numeric_average(
        dashboard_df,
        "Blasting Delay"
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📊 Operational Overview'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI FUNCTION
    # --------------------------------------------------------

    def kpi(title, value, unit=""):

        if value is None:
            text = "N/A"
        else:
            text = f"{value:,.2f} {unit}"

        return f"""
        <div class="kpi">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{text}</div>
        </div>
        """

    # --------------------------------------------------------
    # KPI ROW 1
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            kpi(
                "Average Production",
                production,
                "tonnes"
            ),
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            kpi(
                "Target Production",
                target_production,
                "tonnes"
            ),
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            kpi(
                "Equipment Availability",
                equipment_availability,
                "%"
            ),
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            kpi(
                "Equipment Downtime",
                equipment_downtime,
                "hrs"
            ),
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # KPI ROW 2
    # --------------------------------------------------------

    if (
        production is not None
        and target_production is not None
        and target_production != 0
    ):
        achievement = (
            production / target_production
        ) * 100
    else:
        achievement = None

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.markdown(
            kpi(
                "Average Rainfall",
                rainfall
            ),
            unsafe_allow_html=True
        )

    with c6:
        st.markdown(
            kpi(
                "Soil Moisture",
                soil_moisture
            ),
            unsafe_allow_html=True
        )

    with c7:
        st.markdown(
            kpi(
                "Blasting Delay",
                blasting_delay,
                "hrs"
            ),
            unsafe_allow_html=True
        )

    with c8:
        st.markdown(
            kpi(
                "Production Achievement",
                achievement,
                "%"
            ),
            unsafe_allow_html=True
        )

    st.divider()

    # ========================================================
    # PRODUCTION GRAPH
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Production Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    if dashboard_df is not None:

        production_columns = []

        for col in [
            "Production",
            "Target Production",
            "Previous Production"
        ]:

            if col in dashboard_df.columns:
                production_columns.append(col)

        if production_columns:

            graph_df = dashboard_df[
                production_columns
            ].copy()

            for col in production_columns:

                graph_df[col] = pd.to_numeric(
                    graph_df[col],
                    errors="coerce"
                )

            graph_df = graph_df.dropna(
                how="all"
            )

            if not graph_df.empty:

                st.line_chart(
                    graph_df,
                    use_container_width=True
                )

            else:

                st.info(
                    "Production data is empty."
                )

        else:

            st.info(
                "Production columns were not found "
                "in the CSV file."
            )

    else:

        st.warning(
            "No CSV file was found inside the data folder."
        )

    # ========================================================
    # ENVIRONMENT GRAPH
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🌦️ Environmental Conditions'
        '</div>',
        unsafe_allow_html=True
    )

    if dashboard_df is not None:

        environmental_columns = []

        for col in [
            "Rainfall",
            "Soil Moisture",
            "Vegetation Index",
            "Land Temperature"
        ]:

            if col in dashboard_df.columns:
                environmental_columns.append(col)

        if environmental_columns:

            env_df = dashboard_df[
                environmental_columns
            ].copy()

            for col in environmental_columns:

                env_df[col] = pd.to_numeric(
                    env_df[col],
                    errors="coerce"
                )

            env_df = env_df.dropna(
                how="all"
            )

            if not env_df.empty:

                st.line_chart(
                    env_df,
                    use_container_width=True
                )

    # ========================================================
    # EQUIPMENT GRAPH
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '⚙️ Equipment Performance'
        '</div>',
        unsafe_allow_html=True
    )

    if dashboard_df is not None:

        equipment_columns = []

        for col in [
            "Equipment Availability",
            "Equipment Downtime",
            "Blasting Delay"
        ]:

            if col in dashboard_df.columns:
                equipment_columns.append(col)

        if equipment_columns:

            equipment_df = dashboard_df[
                equipment_columns
            ].copy()

            for col in equipment_columns:

                equipment_df[col] = pd.to_numeric(
                    equipment_df[col],
                    errors="coerce"
                )

            equipment_df = equipment_df.dropna(
                how="all"
            )

            if not equipment_df.empty:

                st.bar_chart(
                    equipment_df,
                    use_container_width=True
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section-title">'
            '🌱 Environmental Summary'
            '</div>',
            unsafe_allow_html=True
        )

        if rainfall is not None:
            st.write(
                f"🌧️ Average rainfall: "
                f"**{rainfall:.2f}**"
            )

        if soil_moisture is not None:
            st.write(
                f"💧 Average soil moisture: "
                f"**{soil_moisture:.2f}**"
            )

        temperature = numeric_average(
            dashboard_df,
            "Land Temperature"
        )

        if temperature is not None:
            st.write(
                f"🌡️ Average land temperature: "
                f"**{temperature:.2f}**"
            )

        vegetation = numeric_average(
            dashboard_df,
            "Vegetation Index"
        )

        if vegetation is not None:
            st.write(
                f"🌿 Vegetation index: "
                f"**{vegetation:.2f}**"
            )

    with right:

        st.markdown(
            '<div class="section-title">'
            '🟢 System Health'
            '</div>',
            unsafe_allow_html=True
        )

        if location_model is not None:
            st.success(
                "📍 Location Prediction — Ready"
            )
        else:
            st.error(
                "📍 Location Prediction — Model Error"
            )

        if production_model is not None:
            st.success(
                "🏭 Production Prediction — Ready"
            )
        else:
            st.error(
                "🏭 Production Prediction — Model Error"
            )

        if production_model is not None:
            st.success(
                "🔄 Production Simulator — Ready"
            )
        else:
            st.error(
                "🔄 Production Simulator — Model Error"
            )

    # ========================================================
    # MODULE CARDS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🚀 ManganAI Modules'
        '</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="feature-card">

            <h3>📍 Manganese Location Prediction</h3>

            Identify promising manganese locations
            using geological and exploration data.

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            """
            <div class="feature-card">

            <h3>🏭 Production Prediction</h3>

            Predict expected manganese production
            using mining, environmental and equipment
            conditions.

            </div>
            """,
            unsafe_allow_html=True
        )

    with c:

        st.markdown(
            """
            <div class="feature-card">

            <h3>🔄 Production Improvement Simulator</h3>

            Compare current conditions with improved
            operating conditions and estimate production
            change.

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.caption(
        "ManganAI | AI-assisted manganese exploration "
        "and production decision-support system"
    )


# ============================================================
# 12. SIDEBAR
# ============================================================

st.sidebar.title("⛏️ ManganAI")

st.sidebar.write(
    "AI-Based Manganese Mining Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Select Module",
    [
        "📊 Dashboard",
        "📍 Manganese Location Prediction",
        "🏭 Production Prediction",
        "🔄 Production Improvement Simulator"
    ]
)


# ============================================================
# SIDEBAR MODEL STATUS
# ============================================================

with st.sidebar.expander(
    "🔧 Model Status"
):

    if location_model is not None:

        st.success(
            "Location model loaded"
        )

    else:

        st.error(
            "Location model could not be loaded"
        )

        if location_error:
            st.caption(
                location_error
            )

    if production_model is not None:

        st.success(
            "Production model loaded"
        )

    else:

        st.error(
            "Production model could not be loaded"
        )

        if production_error:
            st.caption(
                production_error
            )


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "📊 Dashboard":

    show_dashboard()


# ============================================================
# PAGE 1
# LOCATION PREDICTION
# ============================================================

elif page == "📍 Manganese Location Prediction":

    st.markdown(
        '<div class="main-title">'
        '📍 Manganese Location Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use the trained exploration model to predict "
        "manganese potential."
    )

    if location_model is None:

        st.error(
            "❌ Location model could not be loaded."
        )

        if location_error:

            st.code(
                location_error
            )

        st.info(
            "Check the models folder and the error shown above."
        )

        st.stop()

    if not location_features:

        st.warning(
            "The model does not expose feature names."
        )

        st.write(
            "The model was loaded, but its training "
            "feature names are unavailable."
        )

        st.stop()

    st.success(
        "✅ Location model loaded successfully."
    )

    with st.expander(
        "View exact model features"
    ):

        st.write(
            location_features
        )

    st.subheader(
        "Enter Location / Geological Values"
    )

    location_values = create_inputs(
        location_features,
        "location"
    )

    if st.button(
        "🔍 Predict Manganese Location",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [location_values],
            columns=location_features
        )

        try:

            prediction = location_model.predict(
                input_df
            )[0]

            st.divider()

            st.subheader(
                "📊 Prediction Result"
            )

            st.success(
                f"Model Prediction: {prediction}"
            )

            st.subheader(
                "📋 Input Data"
            )

            st.dataframe(
                input_df,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Location prediction failed."
            )

            st.code(
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# PAGE 2
# PRODUCTION PREDICTION
# ============================================================

elif page == "🏭 Production Prediction":

    st.markdown(
        '<div class="main-title">'
        '🏭 Production Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Predict expected manganese production from "
        "current mining conditions."
    )

    if production_model is None:

        st.error(
            "❌ Production model could not be loaded."
        )

        if production_error:

            st.code(
                production_error
            )

        st.stop()

    if not production_features:

        st.warning(
            "The production model does not expose "
            "feature names."
        )

        st.stop()

    st.success(
        "✅ Production model loaded successfully."
    )

    with st.expander(
        "View exact model features"
    ):

        st.write(
            production_features
        )

    st.subheader(
        "Enter Current Mining Conditions"
    )

    production_values = create_inputs(
        production_features,
        "production"
    )

    if st.button(
        "🏭 Predict Production",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [production_values],
            columns=production_features
        )

        try:

            prediction = float(
                production_model.predict(
                    input_df
                )[0]
            )

            st.divider()

            st.subheader(
                "🎯 Production Prediction"
            )

            m1, m2 = st.columns(2)

            with m1:

                st.metric(
                    "Predicted Production",
                    f"{prediction:,.2f} tonnes"
                )

            target_column = None

            for feature in production_features:

                name = str(feature).lower()

                if (
                    "target" in name
                    and "production" in name
                ):

                    target_column = feature
                    break

            with m2:

                if target_column is not None:

                    target = float(
                        production_values[
                            target_column
                        ]
                    )

                    achievement = (
                        prediction / target * 100
                        if target != 0
                        else 0
                    )

                    st.metric(
                        "Target Achievement",
                        f"{achievement:.2f}%"
                    )

            if target_column is not None:

                target = float(
                    production_values[
                        target_column
                    ]
                )

                shortfall = (
                    target - prediction
                )

                if shortfall > 0:

                    st.warning(
                        f"⚠️ Possible production shortfall: "
                        f"{shortfall:,.2f} tonnes"
                    )

                else:

                    st.success(
                        "✅ Production meets or exceeds target."
                    )

            with st.expander(
                "View Values Used for Prediction"
            ):

                st.dataframe(
                    input_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Production prediction failed."
            )

            st.code(
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# PAGE 3
# PRODUCTION IMPROVEMENT SIMULATOR
# ============================================================

else:

    st.markdown(
        '<div class="main-title">'
        '🔄 Production Improvement Simulator'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare current mining conditions with an "
        "improved scenario."
    )

    if production_model is None:

        st.error(
            "❌ Production model could not be loaded."
        )

        if production_error:

            st.code(
                production_error
            )

        st.stop()

    if not production_features:

        st.warning(
            "The production model does not expose "
            "feature names."
        )

        st.stop()

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Current Mining Conditions"
    )

    current_values = create_inputs(
        production_features,
        "current"
    )

    st.divider()

    # --------------------------------------------------------
    # IMPROVED
    # --------------------------------------------------------

    st.subheader(
        "2️⃣ Improved Mining Conditions"
    )

    st.write(
        "Change the conditions you want to improve."
    )

    improved_values = {}

    cols = st.columns(2)

    for i, feature in enumerate(
        production_features
    ):

        with cols[i % 2]:

            improved_values[feature] = st.number_input(
                f"Improved {feature}",
                value=float(
                    current_values[feature]
                ),
                key=f"improved_{feature}"
            )

    st.divider()

    # --------------------------------------------------------
    # SIMULATION
    # --------------------------------------------------------

    if st.button(
        "🔮 Compare Current vs Improved Production",
        type="primary",
        use_container_width=True
    ):

        current_df = pd.DataFrame(
            [current_values],
            columns=production_features
        )

        improved_df = pd.DataFrame(
            [improved_values],
            columns=production_features
        )

        try:

            current_prediction = float(
                production_model.predict(
                    current_df
                )[0]
            )

            improved_prediction = float(
                production_model.predict(
                    improved_df
                )[0]
            )

            change = (
                improved_prediction
                -
                current_prediction
            )

            percentage_change = (
                (change / current_prediction) * 100
                if current_prediction != 0
                else 0
            )

            st.subheader(
                "📊 Simulation Result"
            )

            r1, r2, r3, r4 = st.columns(4)

            with r1:

                st.metric(
                    "Current Production",
                    f"{current_prediction:,.2f}"
                )

            with r2:

                st.metric(
                    "Improved Production",
                    f"{improved_prediction:,.2f}"
                )

            with r3:

                st.metric(
                    "Production Change",
                    f"{change:+,.2f}"
                )

            with r4:

                st.metric(
                    "Percentage Change",
                    f"{percentage_change:+.2f}%"
                )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if change > 0:

                st.success(
                    "🟢 Improved conditions increase "
                    "predicted production."
                )

            elif change < 0:

                st.warning(
                    "🟠 Improved scenario resulted in "
                    "lower predicted production."
                )

            else:

                st.info(
                    "Production prediction is unchanged."
                )

            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            st.subheader(
                "💡 Recommended Corrective Actions"
            )

            recommendations = []

            for feature in production_features:

                name = str(feature).lower()

                old = current_values[feature]
                new = improved_values[feature]

                if (
                    "downtime" in name
                    and new < old
                ):

                    recommendations.append(
                        f"🔧 Reduce {feature} "
                        f"from {old} to {new}."
                    )

                if (
                    "availability" in name
                    and new > old
                ):

                    recommendations.append(
                        f"🚜 Increase {feature} "
                        f"from {old} to {new}."
                    )

                if (
                    "blasting" in name
                    and "delay" in name
                    and new < old
                ):

                    recommendations.append(
                        f"💥 Reduce {feature} "
                        f"from {old} to {new}."
                    )

            if not recommendations:

                recommendations.append(
                    "ℹ️ No specific operational change "
                    "was detected. Try reducing downtime "
                    "or blasting delay, or increasing "
                    "equipment availability."
                )

            for item in recommendations:

                st.write(item)

            # ------------------------------------------------
            # COMPARISON
            # ------------------------------------------------

            st.subheader(
                "📋 Current vs Improved Inputs"
            )

            comparison = pd.DataFrame({

                "Parameter": [
                    str(x)
                    for x in production_features
                ],

                "Current": [
                    current_values[x]
                    for x in production_features
                ],

                "Improved": [
                    improved_values[x]
                    for x in production_features
                ]

            })

            st.dataframe(
                comparison,
                use_container_width=True
            )

            # ------------------------------------------------
            # PRODUCTION BAR CHART
            # ------------------------------------------------

            chart = pd.DataFrame({

                "Production": [
                    current_prediction,
                    improved_prediction
                ]

            }, index=[
                "Current",
                "Improved"
            ])

            st.subheader(
                "📈 Production Comparison"
            )

            st.bar_chart(
                chart,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Simulation failed."
            )

            st.code(
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.success(
    "ManganAI Ready"
)

st.sidebar.caption(
    "AI + Geological + Environmental + "
    "Equipment Intelligence"
)