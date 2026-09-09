import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# ManganAI - COMPLETE STREAMLIT WEBSITE
# ============================================================

st.set_page_config(
    page_title="ManganAI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

EXPLORATION_CSV = DATA_DIR / "exploration.csv"
PRODUCTION_CSV = DATA_DIR / "production.csv"

EXPLORATION_MODEL = MODEL_DIR / "exploration_model.pkl"
PRODUCTION_MODEL = MODEL_DIR / "production_model.pkl"


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    font-size: 19px;
    opacity: 0.75;
    margin-bottom: 20px;
}

.card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 12px;
}

.card-title {
    font-size: 15px;
    opacity: 0.7;
}

.card-value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 7px;
}

.module-card {
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 180px;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_exploration_data():
    if EXPLORATION_CSV.exists():
        try:
            return pd.read_csv(EXPLORATION_CSV)
        except Exception:
            return None
    return None


@st.cache_data
def load_production_data():
    if PRODUCTION_CSV.exists():
        try:
            return pd.read_csv(PRODUCTION_CSV)
        except Exception:
            return None
    return None


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_exploration_model():
    if EXPLORATION_MODEL.exists():
        try:
            return joblib.load(EXPLORATION_MODEL)
        except Exception:
            return None
    return None


@st.cache_resource
def load_production_model():
    if PRODUCTION_MODEL.exists():
        try:
            return joblib.load(PRODUCTION_MODEL)
        except Exception:
            return None
    return None


exploration_df = load_exploration_data()
production_df = load_production_data()

exploration_model = load_exploration_model()
production_model = load_production_model()


# ============================================================
# FEATURE DETECTION
# ============================================================

def get_model_features(model):

    if model is None:
        return []

    # Most sklearn models
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    # Pipeline
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                try:
                    return list(step.feature_names_in_)
                except Exception:
                    pass

    return []


exploration_features = get_model_features(
    exploration_model
)

production_features = get_model_features(
    production_model
)


# ============================================================
# CSV FEATURE FALLBACK
# ============================================================

def get_csv_features(df, model_features):

    if len(model_features) > 0:
        return model_features

    if df is not None:
        return list(df.select_dtypes(
            include=np.number
        ).columns)

    return []


exploration_features = get_csv_features(
    exploration_df,
    exploration_features
)

production_features = get_csv_features(
    production_df,
    production_features
)


# ============================================================
# DEFAULT VALUE
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

    if "geological" in name:
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

    if "ore" in name and "grade" in name:
        return 4.0

    if "target" in name and "production" in name:
        return 10000.0

    if "previous" in name and "production" in name:
        return 9000.0

    if "production" in name:
        return 9000.0

    if "availability" in name:
        return 85.0

    if "downtime" in name:
        return 10.0

    if "blasting" in name and "delay" in name:
        return 10.0

    return 0.0


# ============================================================
# SAFE INPUT CREATOR
# ============================================================

def create_inputs(features, prefix):

    values = {}

    if not features:
        return values

    cols = st.columns(2)

    for i, feature in enumerate(features):

        with cols[i % 2]:

            key = f"{prefix}_{str(feature)}"

            values[feature] = st.number_input(
                str(feature),
                value=float(default_value(feature)),
                key=key
            )

    return values


# ============================================================
# PREDICTION HELPER
# ============================================================

def make_prediction(model, values, features):

    if model is None:
        raise ValueError("Model could not be loaded.")

    if not features:
        raise ValueError(
            "No model input features were detected."
        )

    input_df = pd.DataFrame(
        [values],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    return float(prediction), input_df


# ============================================================
# TARGET COLUMN FINDER
# ============================================================

def find_target_column(features):

    for feature in features:

        name = str(feature).lower()

        if (
            "target production" in name
            or
            "target_production" in name
        ):
            return feature

    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⛏️ ManganAI")

st.sidebar.caption(
    "AI-Assisted Manganese Mining Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📍 Manganese Location Prediction",
        "🏭 Production Prediction",
        "🔄 Production Improvement Simulator"
    ]
)


# ============================================================
# SIDEBAR MODEL STATUS
# ============================================================

with st.sidebar.expander("🔧 System Status"):

    if exploration_model is not None:
        st.success("Exploration Model ✓")
    else:
        st.error("Exploration Model ✗")

    if production_model is not None:
        st.success("Production Model ✓")
    else:
        st.error("Production Model ✗")

    if exploration_df is not None:
        st.success("Exploration Data ✓")
    else:
        st.error("Exploration Data ✗")

    if production_df is not None:
        st.success("Production Data ✓")
    else:
        st.error("Production Data ✗")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">⛏️ ManganAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Assisted Manganese Exploration & Production System'
        '</div>',
        unsafe_allow_html=True
    )

    st.success("🟢 ManganAI System Online")

    st.divider()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("📊 Mining Data Overview")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        if exploration_df is not None:
            st.metric(
                "Exploration Records",
                f"{len(exploration_df):,}"
            )
        else:
            st.metric("Exploration Records", "N/A")

    with d2:
        if production_df is not None:
            st.metric(
                "Production Records",
                f"{len(production_df):,}"
            )
        else:
            st.metric("Production Records", "N/A")

    with d3:
        st.metric(
            "Exploration Features",
            len(exploration_features)
        )

    with d4:
        st.metric(
            "Production Features",
            len(production_features)
        )

    st.divider()

    # --------------------------------------------------------
    # PRODUCTION KPI
    # --------------------------------------------------------

    st.subheader("🏭 Production Overview")

    if production_df is not None:

        production_columns = [
            c for c in production_df.columns
            if "production" in str(c).lower()
        ]

        if production_columns:

            numeric_production = []

            for c in production_columns:

                series = pd.to_numeric(
                    production_df[c],
                    errors="coerce"
                )

                if series.notna().any():
                    numeric_production.append(
                        (c, series.mean())
                    )

            k1, k2, k3, k4 = st.columns(4)

            if numeric_production:

                first_name, first_value = numeric_production[0]

                with k1:
                    st.metric(
                        "Average Production",
                        f"{first_value:,.2f}"
                    )

            target_value = None

            for name, value in numeric_production:

                if "target" in str(name).lower():

                    target_value = value
                    break

            with k2:
                if target_value is not None:
                    st.metric(
                        "Average Target",
                        f"{target_value:,.2f}"
                    )
                else:
                    st.metric("Average Target", "N/A")

            with k3:
                st.metric(
                    "Production Rows",
                    f"{len(production_df):,}"
                )

            with k4:
                st.metric(
                    "Production Columns",
                    f"{len(production_df.columns):,}"
                )

        else:
            st.info(
                "No production column detected in production.csv."
            )

    else:

        st.error(
            "production.csv could not be loaded."
        )

    # --------------------------------------------------------
    # PRODUCTION GRAPH
    # --------------------------------------------------------

    st.subheader("📈 Production Analytics")

    if production_df is not None:

        graph_columns = []

        for col in production_df.columns:

            if (
                "production" in str(col).lower()
                and
                pd.to_numeric(
                    production_df[col],
                    errors="coerce"
                ).notna().any()
            ):
                graph_columns.append(col)

        if graph_columns:

            chart = production_df[
                graph_columns
            ].copy()

            for col in graph_columns:
                chart[col] = pd.to_numeric(
                    chart[col],
                    errors="coerce"
                )

            st.line_chart(chart)

        else:
            st.info(
                "No numeric production columns available "
                "for graph."
            )

    # --------------------------------------------------------
    # ENVIRONMENT GRAPH
    # --------------------------------------------------------

    st.subheader("🌦️ Environmental Conditions")

    if production_df is not None:

        env_columns = []

        keywords = [
            "rain",
            "moisture",
            "temperature",
            "vegetation",
            "ndvi"
        ]

        for col in production_df.columns:

            name = str(col).lower()

            if any(k in name for k in keywords):

                if pd.to_numeric(
                    production_df[col],
                    errors="coerce"
                ).notna().any():

                    env_columns.append(col)

        if env_columns:

            env = production_df[
                env_columns
            ].copy()

            for col in env_columns:
                env[col] = pd.to_numeric(
                    env[col],
                    errors="coerce"
                )

            st.line_chart(env)

        else:
            st.info(
                "No environmental columns detected."
            )

    # --------------------------------------------------------
    # THREE MODULES
    # --------------------------------------------------------

    st.divider()

    st.subheader("🚀 ManganAI Modules")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="module-card">
        <h3>📍 Location Prediction</h3>
        <p>
        Identify potential manganese locations using
        exploration and geological information.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="module-card">
        <h3>🏭 Production Prediction</h3>
        <p>
        Predict expected manganese production from
        current mining conditions.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="module-card">
        <h3>🔄 Improvement Simulator</h3>
        <p>
        Compare current and improved operating
        conditions before making decisions.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.caption(
        "ManganAI | Geological + Environmental + "
        "Production + Equipment Intelligence"
    )


# ============================================================
# LOCATION PAGE
# ============================================================

elif page == "📍 Manganese Location Prediction":

    st.title("📍 Manganese Location Prediction")

    st.write(
        "Use the trained exploration model to estimate "
        "manganese potential."
    )

    if exploration_model is None:

        st.error(
            "Exploration model could not be loaded."
        )

        st.info(
            "Expected file: "
            "models/exploration_model.pkl"
        )

        st.stop()

    if not exploration_features:

        st.error(
            "The exploration model does not expose "
            "feature names."
        )

        st.stop()

    st.success(
        f"Exploration model loaded successfully. "
        f"{len(exploration_features)} input features detected."
    )

    with st.expander("🔎 Model Features"):

        st.write(
            exploration_features
        )

    st.subheader(
        "Enter Exploration / Geological Values"
    )

    location_values = create_inputs(
        exploration_features,
        "exploration"
    )

    if st.button(
        "🔍 Predict Manganese Location",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction, input_df = make_prediction(
                exploration_model,
                location_values,
                exploration_features
            )

            st.divider()

            st.subheader("🎯 Prediction Result")

            st.metric(
                "Model Prediction",
                str(prediction)
            )

            st.success(
                "Exploration prediction completed successfully."
            )

            with st.expander(
                "View Input Values"
            ):

                st.dataframe(
                    input_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Location prediction failed."
            )

            st.code(str(e))


# ============================================================
# PRODUCTION PAGE
# ============================================================

elif page == "🏭 Production Prediction":

    st.title("🏭 Production Prediction")

    st.write(
        "Predict expected manganese production using "
        "current mining conditions."
    )

    if production_model is None:

        st.error(
            "Production model could not be loaded."
        )

        st.info(
            "Expected file: "
            "models/production_model.pkl"
        )

        st.stop()

    if not production_features:

        st.error(
            "The production model does not expose "
            "feature names."
        )

        st.stop()

    st.success(
        f"Production model loaded successfully. "
        f"{len(production_features)} input features detected."
    )

    with st.expander("🔎 Model Features"):

        st.write(
            production_features
        )

    st.subheader(
        "Enter Current Mining Conditions"
    )

    production_values = create_inputs(
        production_features,
        "production_prediction"
    )

    if st.button(
        "🏭 Predict Production",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction, input_df = make_prediction(
                production_model,
                production_values,
                production_features
            )

            st.divider()

            st.subheader("🎯 Production Result")

            st.metric(
                "Predicted Production",
                f"{prediction:,.2f} tonnes"
            )

            target_column = find_target_column(
                production_features
            )

            if target_column is not None:

                target = float(
                    production_values[target_column]
                )

                difference = target - prediction

                if difference > 0:

                    st.warning(
                        f"⚠️ Estimated shortfall: "
                        f"{difference:,.2f} tonnes"
                    )

                else:

                    st.success(
                        "✅ Predicted production meets "
                        "or exceeds the target."
                    )

            with st.expander(
                "📋 View Values Used"
            ):

                st.dataframe(
                    input_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Production prediction failed."
            )

            st.code(str(e))


# ============================================================
# SIMULATOR PAGE
# ============================================================

else:

    st.title(
        "🔄 Production Improvement Simulator"
    )

    st.write(
        "Compare current mining conditions with an "
        "improved scenario."
    )

    if production_model is None:

        st.error(
            "Production model could not be loaded."
        )

        st.info(
            "Expected file: "
            "models/production_model.pkl"
        )

        st.stop()

    if not production_features:

        st.error(
            "Production model features could not be detected."
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
        "sim_current"
    )

    st.divider()

    # --------------------------------------------------------
    # IMPROVED
    # --------------------------------------------------------

    st.subheader(
        "2️⃣ Improved Mining Conditions"
    )

    st.write(
        "Enter the values you want to test in the "
        "improved scenario."
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
                key=f"sim_improved_{feature}"
            )

    st.divider()

    if st.button(
        "🔮 Compare Current vs Improved Production",
        type="primary",
        use_container_width=True
    ):

        try:

            current_prediction, current_df = make_prediction(
                production_model,
                current_values,
                production_features
            )

            improved_prediction, improved_df = make_prediction(
                production_model,
                improved_values,
                production_features
            )

            change = (
                improved_prediction
                -
                current_prediction
            )

            percentage = 0.0

            if current_prediction != 0:

                percentage = (
                    change /
                    abs(current_prediction)
                ) * 100

            st.divider()

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
                    "Change",
                    f"{change:+,.2f} tonnes",
                    f"{percentage:+.2f}%"
                )

            if change > 0:

                st.success(
                    "🟢 The improved scenario increases "
                    "predicted production."
                )

            elif change < 0:

                st.warning(
                    "🟠 The improved scenario decreases "
                    "predicted production."
                )

            else:

                st.info(
                    "Production prediction is unchanged."
                )

            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            st.subheader(
                "💡 Operational Changes"
            )

            recommendations = []

            for feature in production_features:

                old = current_values[feature]
                new = improved_values[feature]

                name = str(feature).lower()

                if "downtime" in name and new < old:

                    recommendations.append(
                        f"🔧 Reduced {feature}: "
                        f"{old} → {new}"
                    )

                elif (
                    "availability" in name
                    and
                    new > old
                ):

                    recommendations.append(
                        f"🚜 Increased {feature}: "
                        f"{old} → {new}"
                    )

                elif (
                    "blasting" in name
                    and
                    "delay" in name
                    and
                    new < old
                ):

                    recommendations.append(
                        f"💥 Reduced {feature}: "
                        f"{old} → {new}"
                    )

            if recommendations:

                for item in recommendations:
                    st.write(item)

            else:

                st.info(
                    "No standard operational improvement "
                    "was detected in the changed inputs."
                )

            # ------------------------------------------------
            # COMPARISON TABLE
            # ------------------------------------------------

            st.subheader(
                "📋 Current vs Improved Conditions"
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
            # PRODUCTION COMPARISON GRAPH
            # ------------------------------------------------

            graph = pd.DataFrame({

                "Scenario": [
                    "Current",
                    "Improved"
                ],

                "Production": [
                    current_prediction,
                    improved_prediction
                ]

            })

            st.subheader(
                "📈 Production Comparison"
            )

            st.bar_chart(
                graph.set_index("Scenario")
            )

        except Exception as e:

            st.error(
                "Simulation failed."
            )

            st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "⛏️ ManganAI"
)

st.sidebar.caption(
    "AI + Geological + Environmental + "
    "Production Intelligence"
)