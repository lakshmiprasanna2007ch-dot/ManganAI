# ============================================================
# MANGANAI - COMPLETE FRONTEND
# ============================================================
#
# Dashboard + 3 AI Modules
#
# 1. Dashboard
# 2. Manganese Location Prediction
# 3. Production Prediction
# 4. Production Improvement Simulator
#
# Dashboard automatically detects CSV column names.
# AI modules use the exact features exposed by trained models.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ManganAI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 4. GLOBAL FRONTEND STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main application spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    /* Main title */
    .mangan-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .mangan-subtitle {
        font-size: 19px;
        margin-top: 0;
        margin-bottom: 1rem;
        opacity: 0.75;
    }

    /* Section headings */
    .section-heading {
        font-size: 25px;
        font-weight: 750;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }

    /* Small dashboard information */
    .dashboard-note {
        font-size: 15px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. MODEL FILE FINDER
# ============================================================

def find_model(possible_names):

    for name in possible_names:

        path = MODEL_DIR / name

        if path.exists():
            return path

    return None


# ============================================================
# 6. FIND LOCATION MODEL
# ============================================================

LOCATION_MODEL_PATH = find_model([
    "location_model.pkl",
    "location_model.joblib",
    "exploration_model.pkl",
    "exploration_model.joblib",
    "manganese_model.pkl",
    "manganese_model.joblib",
    "manganese_location_model.pkl",
    "manganese_location_model.joblib"
])


# ============================================================
# 7. FIND PRODUCTION MODEL
# ============================================================

PRODUCTION_MODEL_PATH = find_model([
    "production_model.pkl",
    "production_model.joblib"
])


# ============================================================
# 8. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    location_model = None
    production_model = None

    # -------------------------
    # Location model
    # -------------------------

    if LOCATION_MODEL_PATH is not None:

        try:

            location_model = joblib.load(
                LOCATION_MODEL_PATH
            )

        except Exception:

            location_model = None


    # -------------------------
    # Production model
    # -------------------------

    if PRODUCTION_MODEL_PATH is not None:

        try:

            production_model = joblib.load(
                PRODUCTION_MODEL_PATH
            )

        except Exception:

            production_model = None


    return location_model, production_model


location_model, production_model = load_models()


# ============================================================
# 9. GET EXACT MODEL FEATURES
# ============================================================

def get_features(model):

    if model is None:
        return []

    # Most sklearn models and pipelines
    # expose feature_names_in_
    if hasattr(model, "feature_names_in_"):

        try:

            return list(
                model.feature_names_in_
            )

        except Exception:

            pass


    # Try pipeline steps
    if hasattr(model, "named_steps"):

        try:

            for _, step in reversed(
                list(model.named_steps.items())
            ):

                if hasattr(
                    step,
                    "feature_names_in_"
                ):

                    return list(
                        step.feature_names_in_
                    )

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
# 10. NORMALIZE COLUMN NAME
# ============================================================

def normalize_column_name(name):

    text = str(name).strip().lower()

    replacements = [
        "_",
        "-",
        "/",
        "\\",
        "(",
        ")",
        "[",
        "]",
        "%",
        ":",
        "."
    ]

    for char in replacements:
        text = text.replace(char, " ")

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# 11. AUTOMATIC COLUMN DETECTION
# ============================================================

def find_dashboard_column(
    dataframe,
    exact_names=None,
    required_words=None,
    excluded_words=None
):

    if dataframe is None:
        return None

    exact_names = exact_names or []
    required_words = required_words or []
    excluded_words = excluded_words or []

    normalized = {}

    for column in dataframe.columns:

        normalized[column] = (
            normalize_column_name(column)
        )


    # --------------------------------------------------------
    # First: exact normalized match
    # --------------------------------------------------------

    for wanted in exact_names:

        wanted_normalized = (
            normalize_column_name(wanted)
        )

        for original, clean in normalized.items():

            if clean == wanted_normalized:

                return original


    # --------------------------------------------------------
    # Second: required words
    # --------------------------------------------------------

    candidates = []

    for original, clean in normalized.items():

        words = set(
            clean.split()
        )

        valid = True

        for required in required_words:

            required_normalized = (
                normalize_column_name(required)
            )

            if required_normalized not in words:

                valid = False
                break


        if not valid:
            continue


        for excluded in excluded_words:

            excluded_normalized = (
                normalize_column_name(excluded)
            )

            if excluded_normalized in words:

                valid = False
                break


        if valid:

            candidates.append(original)


    if candidates:

        return candidates[0]


    return None


# ============================================================
# 12. AUTOMATIC CSV FINDER
# ============================================================

@st.cache_data
def load_dashboard_dataset():

    if not DATA_DIR.exists():

        return None, None


    csv_files = list(
        DATA_DIR.rglob("*.csv")
    )


    if not csv_files:

        return None, None


    # --------------------------------------------------------
    # Try to select the CSV that looks most like the
    # main ManganAI dataset.
    # --------------------------------------------------------

    scored_files = []


    dashboard_keywords = [
        "production",
        "rainfall",
        "soil",
        "moisture",
        "equipment",
        "downtime",
        "blasting",
        "vegetation",
        "temperature",
        "target"
    ]


    for file in csv_files:

        score = 0

        try:

            sample = pd.read_csv(
                file,
                nrows=5
            )

            columns_text = " ".join(
                normalize_column_name(c)
                for c in sample.columns
            )

            for keyword in dashboard_keywords:

                if keyword in columns_text:

                    score += 1


            # Slight preference for larger datasets
            score += min(
                file.stat().st_size / 1000000,
                5
            )


            scored_files.append(
                (
                    score,
                    file
                )
            )

        except Exception:

            continue


    if not scored_files:

        return None, None


    scored_files.sort(
        key=lambda x: x[0],
        reverse=True
    )


    selected_file = scored_files[0][1]


    try:

        dataframe = pd.read_csv(
            selected_file
        )

        dataframe.columns = [
            str(c).strip()
            for c in dataframe.columns
        ]

        return dataframe, selected_file


    except Exception:

        return None, None


dashboard_df, dashboard_file = (
    load_dashboard_dataset()
)


# ============================================================
# 13. CONVERT NUMERICAL COLUMNS
# ============================================================

if dashboard_df is not None:

    for column in dashboard_df.columns:

        try:

            converted = pd.to_numeric(
                dashboard_df[column],
                errors="coerce"
            )

            # Only replace original column when
            # at least some numeric values exist.
            if converted.notna().sum() > 0:

                dashboard_df[column] = converted

        except Exception:

            pass


# ============================================================
# 14. GET COLUMN BY TYPE
# ============================================================

def detect_column(column_type):

    if dashboard_df is None:
        return None


    # --------------------------------------------------------
    # Production
    # --------------------------------------------------------

    if column_type == "production":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Production",
                "Production Tonnes",
                "Production (tonnes)",
                "Production tonnes"
            ],
            required_words=[
                "production"
            ],
            excluded_words=[
                "target",
                "previous",
                "forecast",
                "predicted"
            ]
        )


    # --------------------------------------------------------
    # Target production
    # --------------------------------------------------------

    if column_type == "target_production":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Target Production",
                "Target Production Tonnes",
                "Production Target",
                "Target"
            ],
            required_words=[
                "target",
                "production"
            ]
        )


    # --------------------------------------------------------
    # Previous production
    # --------------------------------------------------------

    if column_type == "previous_production":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Previous Production",
                "Previous Production Tonnes"
            ],
            required_words=[
                "previous",
                "production"
            ]
        )


    # --------------------------------------------------------
    # Rainfall
    # --------------------------------------------------------

    if column_type == "rainfall":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Rainfall",
                "Rainfall mm",
                "Rainfall (mm)",
                "Annual Rainfall"
            ],
            required_words=[
                "rainfall"
            ]
        )


    # --------------------------------------------------------
    # Soil moisture
    # --------------------------------------------------------

    if column_type == "soil_moisture":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Soil Moisture",
                "Soil Moisture %",
                "Soil Moisture Percentage"
            ],
            required_words=[
                "soil",
                "moisture"
            ]
        )


    # --------------------------------------------------------
    # Equipment availability
    # --------------------------------------------------------

    if column_type == "equipment_availability":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Equipment Availability",
                "Equipment Availability %",
                "Equipment Availability Percentage"
            ],
            required_words=[
                "equipment",
                "availability"
            ]
        )


    # --------------------------------------------------------
    # Equipment downtime
    # --------------------------------------------------------

    if column_type == "equipment_downtime":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Equipment Downtime",
                "Equipment Downtime Hours",
                "Downtime",
                "Downtime Hours"
            ],
            required_words=[
                "downtime"
            ]
        )


    # --------------------------------------------------------
    # Blasting delay
    # --------------------------------------------------------

    if column_type == "blasting_delay":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Blasting Delay",
                "Blasting Delay Hours",
                "Blast Delay"
            ],
            required_words=[
                "blasting",
                "delay"
            ]
        )


    # --------------------------------------------------------
    # Vegetation index
    # --------------------------------------------------------

    if column_type == "vegetation":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Vegetation Index",
                "NDVI",
                "Vegetation"
            ],
            required_words=[
                "vegetation"
            ]
        )


    # --------------------------------------------------------
    # Land temperature
    # --------------------------------------------------------

    if column_type == "temperature":

        return find_dashboard_column(
            dashboard_df,
            exact_names=[
                "Land Temperature",
                "Land Temperature C",
                "Temperature",
                "Temperature C"
            ],
            required_words=[
                "temperature"
            ]
        )


    return None


# ============================================================
# 15. GET AVERAGE VALUE
# ============================================================

def get_average(column_name):

    if (
        dashboard_df is None
        or column_name is None
        or column_name not in dashboard_df.columns
    ):

        return None


    try:

        values = pd.to_numeric(
            dashboard_df[column_name],
            errors="coerce"
        ).dropna()


        if len(values) == 0:

            return None


        return float(
            values.mean()
        )

    except Exception:

        return None


# ============================================================
# 16. DASHBOARD
# ============================================================

def show_dashboard():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="mangan-title">⛏️ ManganAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="mangan-subtitle">'
        'AI-Based Manganese Mining Intelligence System'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore manganese potential, predict production, "
        "identify possible shortfalls and test corrective actions."
    )


    st.success(
        "🟢 ManganAI System Ready"
    )


    st.divider()


    # ========================================================
    # DETECT DASHBOARD COLUMNS AUTOMATICALLY
    # ========================================================

    production_column = detect_column(
        "production"
    )

    target_column = detect_column(
        "target_production"
    )

    previous_production_column = detect_column(
        "previous_production"
    )

    rainfall_column = detect_column(
        "rainfall"
    )

    soil_column = detect_column(
        "soil_moisture"
    )

    availability_column = detect_column(
        "equipment_availability"
    )

    downtime_column = detect_column(
        "equipment_downtime"
    )

    blasting_column = detect_column(
        "blasting_delay"
    )

    vegetation_column = detect_column(
        "vegetation"
    )

    temperature_column = detect_column(
        "temperature"
    )


    # ========================================================
    # VALUES
    # ========================================================

    production = get_average(
        production_column
    )

    target_production = get_average(
        target_column
    )

    rainfall = get_average(
        rainfall_column
    )

    soil_moisture = get_average(
        soil_column
    )

    equipment_availability = get_average(
        availability_column
    )

    equipment_downtime = get_average(
        downtime_column
    )

    blasting_delay = get_average(
        blasting_column
    )

    vegetation = get_average(
        vegetation_column
    )

    temperature = get_average(
        temperature_column
    )


    # ========================================================
    # OPERATIONAL OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '📊 Operational Overview'
        '</div>',
        unsafe_allow_html=True
    )


    k1, k2, k3, k4 = st.columns(4)


    with k1:

        if production is not None:

            st.metric(
                "Average Production",
                f"{production:,.2f}"
            )

        else:

            st.metric(
                "Average Production",
                "N/A"
            )


    with k2:

        if target_production is not None:

            st.metric(
                "Target Production",
                f"{target_production:,.2f}"
            )

        else:

            st.metric(
                "Target Production",
                "N/A"
            )


    with k3:

        if equipment_availability is not None:

            st.metric(
                "Equipment Availability",
                f"{equipment_availability:,.2f}%"
            )

        else:

            st.metric(
                "Equipment Availability",
                "N/A"
            )


    with k4:

        if equipment_downtime is not None:

            st.metric(
                "Equipment Downtime",
                f"{equipment_downtime:,.2f}"
            )

        else:

            st.metric(
                "Equipment Downtime",
                "N/A"
            )


    # ========================================================
    # SECOND KPI ROW
    # ========================================================

    k5, k6, k7, k8 = st.columns(4)


    with k5:

        if rainfall is not None:

            st.metric(
                "Average Rainfall",
                f"{rainfall:,.2f}"
            )

        else:

            st.metric(
                "Average Rainfall",
                "N/A"
            )


    with k6:

        if soil_moisture is not None:

            st.metric(
                "Soil Moisture",
                f"{soil_moisture:,.2f}"
            )

        else:

            st.metric(
                "Soil Moisture",
                "N/A"
            )


    with k7:

        if blasting_delay is not None:

            st.metric(
                "Blasting Delay",
                f"{blasting_delay:,.2f}"
            )

        else:

            st.metric(
                "Blasting Delay",
                "N/A"
            )


    # Production achievement

    if (
        production is not None
        and target_production is not None
        and target_production != 0
    ):

        achievement = (
            production
            /
            target_production
        ) * 100

    else:

        achievement = None


    with k8:

        if achievement is not None:

            st.metric(
                "Production Achievement",
                f"{achievement:,.2f}%"
            )

        else:

            st.metric(
                "Production Achievement",
                "N/A"
            )


    st.divider()


    # ========================================================
    # PRODUCTION ANALYTICS
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '📈 Production Analytics'
        '</div>',
        unsafe_allow_html=True
    )


    if dashboard_df is not None:

        production_columns = []


        if production_column is not None:
            production_columns.append(
                production_column
            )


        if target_column is not None:
            production_columns.append(
                target_column
            )


        if previous_production_column is not None:
            production_columns.append(
                previous_production_column
            )


        # Remove duplicates
        production_columns = list(
            dict.fromkeys(
                production_columns
            )
        )


        if production_columns:

            chart_df = dashboard_df[
                production_columns
            ].copy()


            for column in production_columns:

                chart_df[column] = pd.to_numeric(
                    chart_df[column],
                    errors="coerce"
                )


            chart_df = chart_df.dropna(
                how="all"
            )


            if len(chart_df) > 0:

                st.line_chart(
                    chart_df,
                    use_container_width=True
                )

            else:

                st.info(
                    "Production data is available "
                    "but does not contain usable numeric values."
                )

        else:

            st.info(
                "No production-related column was automatically "
                "detected in the dataset."
            )

    else:

        st.info(
            "No CSV dataset was found in the data folder."
        )


    # ========================================================
    # ENVIRONMENTAL CONDITIONS
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '🌦️ Environmental Conditions'
        '</div>',
        unsafe_allow_html=True
    )


    if dashboard_df is not None:

        environmental_columns = []


        if rainfall_column is not None:
            environmental_columns.append(
                rainfall_column
            )


        if soil_column is not None:
            environmental_columns.append(
                soil_column
            )


        if vegetation_column is not None:
            environmental_columns.append(
                vegetation_column
            )


        if temperature_column is not None:
            environmental_columns.append(
                temperature_column
            )


        environmental_columns = list(
            dict.fromkeys(
                environmental_columns
            )
        )


        if environmental_columns:

            env_df = dashboard_df[
                environmental_columns
            ].copy()


            for column in environmental_columns:

                env_df[column] = pd.to_numeric(
                    env_df[column],
                    errors="coerce"
                )


            env_df = env_df.dropna(
                how="all"
            )


            if len(env_df) > 0:

                st.line_chart(
                    env_df,
                    use_container_width=True
                )

        else:

            st.info(
                "No environmental columns were detected."
            )


    # ========================================================
    # EQUIPMENT PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '⚙️ Equipment Performance'
        '</div>',
        unsafe_allow_html=True
    )


    if dashboard_df is not None:

        equipment_columns = []


        if availability_column is not None:
            equipment_columns.append(
                availability_column
            )


        if downtime_column is not None:
            equipment_columns.append(
                downtime_column
            )


        if blasting_column is not None:
            equipment_columns.append(
                blasting_column
            )


        equipment_columns = list(
            dict.fromkeys(
                equipment_columns
            )
        )


        if equipment_columns:

            equipment_df = dashboard_df[
                equipment_columns
            ].copy()


            for column in equipment_columns:

                equipment_df[column] = pd.to_numeric(
                    equipment_df[column],
                    errors="coerce"
                )


            equipment_df = equipment_df.dropna(
                how="all"
            )


            if len(equipment_df) > 0:

                st.bar_chart(
                    equipment_df,
                    use_container_width=True
                )

        else:

            st.info(
                "No equipment-related columns were detected."
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.divider()


    left, right = st.columns(2)


    # --------------------------------------------------------
    # Environmental Summary
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-heading">'
            '🌱 Environmental Summary'
            '</div>',
            unsafe_allow_html=True
        )


        if rainfall is not None:

            st.write(
                f"🌧️ Average rainfall: "
                f"**{rainfall:.2f}**"
            )

        else:

            st.write(
                "🌧️ Rainfall: **Not available**"
            )


        if soil_moisture is not None:

            st.write(
                f"💧 Average soil moisture: "
                f"**{soil_moisture:.2f}**"
            )

        else:

            st.write(
                "💧 Soil moisture: **Not available**"
            )


        if temperature is not None:

            st.write(
                f"🌡️ Average land temperature: "
                f"**{temperature:.2f}**"
            )

        else:

            st.write(
                "🌡️ Land temperature: **Not available**"
            )


        if vegetation is not None:

            st.write(
                f"🌿 Average vegetation index: "
                f"**{vegetation:.2f}**"
            )

        else:

            st.write(
                "🌿 Vegetation index: **Not available**"
            )


    # --------------------------------------------------------
    # System Health
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-heading">'
            '🟢 System Health'
            '</div>',
            unsafe_allow_html=True
        )


        if location_model is not None:

            st.success(
                "Location Prediction — Ready"
            )

        else:

            st.error(
                "Location Prediction — Model not found"
            )


        if production_model is not None:

            st.success(
                "Production Prediction — Ready"
            )

        else:

            st.error(
                "Production Prediction — Model not found"
            )


        if production_model is not None:

            st.success(
                "Production Simulator — Ready"
            )

        else:

            st.error(
                "Production Simulator — Model not found"
            )


    # ========================================================
    # MANGANAI MODULES
    # ========================================================

    st.divider()


    st.markdown(
        '<div class="section-heading">'
        '🚀 ManganAI Modules'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.info(
            "📍 **Manganese Location Prediction**\n\n"
            "Identify promising manganese locations "
            "using geological and exploration data."
        )


    with c2:

        st.info(
            "🏭 **Production Prediction**\n\n"
            "Predict expected manganese production "
            "from operating and environmental conditions."
        )


    with c3:

        st.info(
            "🔄 **Production Improvement Simulator**\n\n"
            "Change operating conditions and compare "
            "the expected production result."
        )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.divider()


    if dashboard_file is not None:

        st.caption(
            "Dashboard data source: "
            + str(
                dashboard_file.relative_to(
                    BASE_DIR
                )
            )
        )

        st.caption(
            f"Automatically detected "
            f"{len(dashboard_df.columns)} dataset columns."
        )


    st.caption(
        "ManganAI | AI-assisted manganese exploration "
        "and production decision-support system"
    )


# ============================================================
# 17. DEFAULT INPUT VALUES
# ============================================================

def default_value(feature):

    name = normalize_column_name(
        feature
    )


    # Location / geological
    if "latitude" in name:
        return 20.50


    if "longitude" in name:
        return 79.50


    if "elevation" in name:
        return 500.0


    if "magnetic" in name:
        return 50.0


    if (
        "rock density" in name
        or
        "rock_density" in name
    ):
        return 2.70


    if (
        "geological score" in name
        or
        "geological_score" in name
    ):
        return 0.70


    if "ndvi" in name:
        return 0.40


    if "rainfall" in name:
        return 130.0


    if (
        "soil moisture" in name
        or
        "soil_moisture" in name
    ):
        return 45.0


    if "temperature" in name:
        return 32.0


    if "band" in name:
        return 0.20


    if name == "ei":
        return 400.0


    # Production
    if "target production" in name:
        return 10000.0


    if "target_production" in name:
        return 10000.0


    if "previous production" in name:
        return 9000.0


    if "previous_production" in name:
        return 9000.0


    if "equipment downtime" in name:
        return 10.0


    if "equipment_downtime" in name:
        return 10.0


    if "equipment availability" in name:
        return 85.0


    if "equipment_availability" in name:
        return 85.0


    if "blasting delay" in name:
        return 10.0


    if "blasting_delay" in name:
        return 10.0


    if "ore grade" in name:
        return 4.0


    if "ore_grade" in name:
        return 4.0


    return 0.0


# ============================================================
# 18. CREATE MODEL INPUTS
# ============================================================

def create_inputs(
    features,
    prefix
):

    values = {}


    if len(features) == 0:

        return values


    columns = st.columns(2)


    for i, feature in enumerate(features):

        with columns[i % 2]:

            values[feature] = st.number_input(
                str(feature),
                value=float(
                    default_value(feature)
                ),
                key=f"{prefix}_{feature}"
            )


    return values


# ============================================================
# 19. SIDEBAR
# ============================================================

st.sidebar.title(
    "⛏️ ManganAI"
)

st.sidebar.caption(
    "Manganese Mining Intelligence System"
)


st.sidebar.divider()


st.sidebar.subheader(
    "🧭 Navigation"
)


page = st.sidebar.radio(
    "Select a module",
    [
        "🏠 Dashboard",
        "📍 Manganese Location Prediction",
        "🏭 Production Prediction",
        "🔄 Production Improvement Simulator"
    ]
)


# ============================================================
# 20. MODEL STATUS
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
            "Location model not found"
        )


    if production_model is not None:

        st.success(
            "Production model loaded"
        )

    else:

        st.error(
            "Production model not found"
        )


# ============================================================
# 21. DASHBOARD PAGE
# ============================================================

if page == "🏠 Dashboard":

    show_dashboard()


# ============================================================
# 22. LOCATION PREDICTION PAGE
# ============================================================

elif page == "📍 Manganese Location Prediction":

    st.title(
        "📍 Manganese Location Prediction"
    )

    st.write(
        "This module uses the trained location/exploration "
        "model to predict manganese potential."
    )


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if location_model is None:

        st.error(
            "Location model could not be loaded."
        )

        st.write(
            "Please check the ManganAI/models folder."
        )

        st.stop()


    # --------------------------------------------------------
    # Check features
    # --------------------------------------------------------

    if len(location_features) == 0:

        st.error(
            "The location model does not expose "
            "feature names."
        )

        st.write(
            "The frontend cannot safely create the "
            "location inputs without knowing the "
            "training columns."
        )

        st.stop()


    st.success(
        "Using the exact columns from the trained "
        "location model."
    )


    with st.expander(
        "View model features"
    ):

        st.write(
            location_features
        )


    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    st.subheader(
        "Enter Location / Geological Values"
    )


    location_values = create_inputs(
        location_features,
        "location"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Manganese Location",
        type="primary",
        use_container_width=True
    ):

        location_input = pd.DataFrame(
            [location_values],
            columns=location_features
        )


        try:

            prediction = location_model.predict(
                location_input
            )[0]


            st.divider()


            st.subheader(
                "📊 Location Prediction Result"
            )


            st.success(
                f"Model Prediction: {prediction}"
            )


            with st.expander(
                "View Values Used for Prediction"
            ):

                st.dataframe(
                    location_input,
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "Location prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# 23. PRODUCTION PREDICTION PAGE
# ============================================================

elif page == "🏭 Production Prediction":

    st.title(
        "🏭 Production Prediction"
    )

    st.write(
        "Predict expected manganese production from "
        "production, environmental and equipment conditions."
    )


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if production_model is None:

        st.error(
            "Production model could not be loaded."
        )

        st.write(
            "Please check the ManganAI/models folder."
        )

        st.stop()


    # --------------------------------------------------------
    # Check features
    # --------------------------------------------------------

    if len(production_features) == 0:

        st.error(
            "The production model does not expose "
            "feature names."
        )

        st.stop()


    st.success(
        "Using the exact columns from the trained "
        "production model."
    )


    with st.expander(
        "View model features"
    ):

        st.write(
            production_features
        )


    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    st.subheader(
        "Enter Current Mining Conditions"
    )


    production_values = create_inputs(
        production_features,
        "production"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "🏭 Predict Production",
        type="primary",
        use_container_width=True
    ):

        production_input = pd.DataFrame(
            [production_values],
            columns=production_features
        )


        try:

            predicted_production = float(
                production_model.predict(
                    production_input
                )[0]
            )


            st.divider()


            st.subheader(
                "🎯 Production Prediction"
            )


            st.metric(
                "Predicted Production",
                f"{predicted_production:,.2f} tonnes"
            )


            # ------------------------------------------------
            # Find target feature
            # ------------------------------------------------

            target_feature = None


            for feature in production_features:

                name = normalize_column_name(
                    feature
                )


                if (
                    "target production"
                    in name
                ):

                    target_feature = feature
                    break


            # ------------------------------------------------
            # Shortfall
            # ------------------------------------------------

            if target_feature is not None:

                target = float(
                    production_values[
                        target_feature
                    ]
                )


                shortfall = (
                    target
                    -
                    predicted_production
                )


                if shortfall > 0:

                    st.warning(
                        f"⚠️ Possible Production Shortfall: "
                        f"{shortfall:,.2f} tonnes"
                    )

                else:

                    st.success(
                        "✅ Predicted production meets "
                        "or exceeds the target."
                    )


            with st.expander(
                "View Values Used for Prediction"
            ):

                st.dataframe(
                    production_input,
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "Production prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# 24. PRODUCTION IMPROVEMENT SIMULATOR
# ============================================================

else:

    st.title(
        "🔄 Production Improvement Simulator"
    )

    st.write(
        "Compare current mining conditions with an improved "
        "scenario and see how predicted production changes."
    )


    # --------------------------------------------------------
    # Check production model
    # --------------------------------------------------------

    if production_model is None:

        st.error(
            "Production model could not be loaded."
        )

        st.stop()


    if len(production_features) == 0:

        st.error(
            "The production model does not expose "
            "feature names."
        )

        st.stop()


    # ========================================================
    # CURRENT CONDITIONS
    # ========================================================

    st.subheader(
        "1️⃣ Current Mining Conditions"
    )


    current_values = create_inputs(
        production_features,
        "current"
    )


    st.divider()


    # ========================================================
    # IMPROVED CONDITIONS
    # ========================================================

    st.subheader(
        "2️⃣ Improved Mining Conditions"
    )


    st.write(
        "Change the values you want to improve. "
        "For example, reduce equipment downtime or "
        "blasting delay."
    )


    improved_values = {}


    columns = st.columns(2)


    for i, feature in enumerate(
        production_features
    ):

        with columns[i % 2]:

            improved_values[feature] = st.number_input(
                f"Improved {feature}",
                value=float(
                    current_values[feature]
                ),
                key=f"improved_{feature}"
            )


    st.divider()


    # ========================================================
    # COMPARE
    # ========================================================

    if st.button(
        "🔮 Compare Current vs Improved Production",
        type="primary",
        use_container_width=True
    ):

        try:

            current_input = pd.DataFrame(
                [current_values],
                columns=production_features
            )


            improved_input = pd.DataFrame(
                [improved_values],
                columns=production_features
            )


            # ------------------------------------------------
            # Current prediction
            # ------------------------------------------------

            current_prediction = float(
                production_model.predict(
                    current_input
                )[0]
            )


            # ------------------------------------------------
            # Improved prediction
            # ------------------------------------------------

            improved_prediction = float(
                production_model.predict(
                    improved_input
                )[0]
            )


            # ------------------------------------------------
            # Difference
            # ------------------------------------------------

            change = (
                improved_prediction
                -
                current_prediction
            )


            st.subheader(
                "📊 Simulation Result"
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Current Production",
                    f"{current_prediction:,.2f} tonnes"
                )


            with c2:

                st.metric(
                    "Improved Production",
                    f"{improved_prediction:,.2f} tonnes"
                )


            with c3:

                st.metric(
                    "Production Change",
                    f"{change:+,.2f} tonnes"
                )


            # ------------------------------------------------
            # Interpretation
            # ------------------------------------------------

            if change > 0:

                st.success(
                    "🟢 The improved scenario gives a "
                    "higher predicted production."
                )

            elif change < 0:

                st.warning(
                    "🟠 The improved scenario gives a "
                    "lower predicted production."
                )

            else:

                st.info(
                    "Production prediction is unchanged."
                )


            # =================================================
            # RECOMMENDATIONS
            # =================================================

            st.subheader(
                "💡 Recommended Corrective Actions"
            )


            recommendations = []


            for feature in production_features:

                name = normalize_column_name(
                    feature
                )


                old = current_values[
                    feature
                ]

                new = improved_values[
                    feature
                ]


                # Equipment downtime
                if "downtime" in name:

                    if new < old:

                        recommendations.append(
                            f"🔧 Equipment downtime reduced "
                            f"from {old} to {new}."
                        )


                # Equipment availability
                if "availability" in name:

                    if new > old:

                        recommendations.append(
                            f"🚜 Equipment availability improved "
                            f"from {old} to {new}."
                        )


                # Blasting delay
                if (
                    "blasting" in name
                    and
                    "delay" in name
                ):

                    if new < old:

                        recommendations.append(
                            f"💥 Blasting delay reduced "
                            f"from {old} to {new}."
                        )


            if len(recommendations) == 0:

                recommendations.append(
                    "ℹ️ No obvious operational improvement "
                    "was entered. Try reducing downtime or "
                    "blasting delay, or increasing equipment "
                    "availability."
                )


            for recommendation in recommendations:

                st.write(
                    recommendation
                )


            # =================================================
            # COMPARISON TABLE
            # =================================================

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


        except Exception as e:

            st.error(
                "Simulation failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# 25. SIDEBAR FOOTER
# ============================================================

st.sidebar.divider()


st.sidebar.success(
    "ManganAI Ready"
)


st.sidebar.caption(
    "AI + Geological + Environmental + "
    "Equipment Intelligence"
)