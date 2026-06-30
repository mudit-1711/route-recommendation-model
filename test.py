import streamlit as st
from datetime import datetime
from recommend import RouteRecommender

st.set_page_config(page_title="AI Travel Route Optimizer", page_icon="plane", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e3d59;
    text-align: center;
    margin-bottom: 4px;
}
.header-sub {
    text-align: center;
    color: #6c757d;
    font-size: 0.95rem;
    margin-bottom: 28px;
}
.metric-card {
    background: linear-gradient(135deg, #1e3d59, #17607f);
    padding: 22px 16px;
    border-radius: 14px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    opacity: 0.75;
    margin-top: 4px;
}
.route-box {
    background: #f0f4f8;
    border-left: 4px solid #1e3d59;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e3d59;
    margin-top: 10px;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 4px;
}
.badge-flight  { background: #d0ebff; color: #1864ab; }
.badge-train   { background: #d3f9d8; color: #1b6b3a; }
.badge-bus     { background: #fff3bf; color: #6b4e00; }
.badge-cab     { background: #ffe8cc; color: #8a4500; }
.divider { border: none; border-top: 1px solid #dee2e6; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-title'>AI Travel Route Optimizer</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>End-to-End Deep Learning based Travel Recommendation Engine</div>", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    return RouteRecommender()

try:
    recommender = load_recommender()
    loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure preprocess.py and train.py have been run.")
    loaded = False
    st.stop()

CITIES = ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Agra",
          "Pune", "Hyderabad", "Goa", "Jaipur", "Lucknow", "Varanasi"]

PREF_ICONS = {
    "Cheapest":          "Budget (Min Cost)",
    "Fastest":           "Speed (Min Time)",
    "Economical":        "Economical (Best Value)",
    "Balanced":          "Balanced (All Factors)",
    "Luxury":            "Luxury (Premium Travel)",
    "Comfort optimized": "Comfort (Smoothest Ride)",
}

col1, col2 = st.columns(2)
with col1:
    source_city = st.selectbox("Source City", CITIES, index=0)
with col2:
    dest_city = st.selectbox("Destination City", CITIES, index=1)

pickup_area = ""
drop_area   = ""

col3, col4 = st.columns(2)
with col3:
    if source_city in ["Delhi", "Bangalore"]:
        pickup_area = st.text_input(f"Pickup Area in {source_city}", placeholder="e.g. Rohini, Whitefield")
with col4:
    if dest_city in ["Delhi", "Bangalore"]:
        drop_area = st.text_input(f"Drop Area in {dest_city}", placeholder="e.g. Dwarka, Electronic City")

col5, col6 = st.columns(2)
with col5:
    date_val = st.date_input("Date of Travel", min_value=datetime.today())
with col6:
    preference_label = st.selectbox("Travel Preference", list(PREF_ICONS.values()))
    # Map display label back to key
    pref_key_map = {v: k for k, v in PREF_ICONS.items()}
    preference = pref_key_map[preference_label]

st.markdown("")

if st.button("Find Best Route", use_container_width=True, type="primary"):
    if source_city == dest_city:
        st.warning("Source and destination cities are the same. Showing local cab options.")

    with st.spinner("Analysing routes using Deep Learning..."):
        try:
            result = recommender.get_best_route(
                source_city=source_city,
                dest_city=dest_city,
                pickup_area=pickup_area,
                drop_area=drop_area,
                date_str=date_val.strftime('%Y-%m-%d'),
                preference=preference,
            )

            st.success("Best Route Found!")

            # ── Metrics row ────────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Estimated Cost</div>
                    <div class='metric-value'>{result['Total Cost']}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Estimated Time</div>
                    <div class='metric-value'>{result['Estimated Time']}</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>AI Confidence</div>
                    <div class='metric-value'>{result['Confidence Score']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # ── Route sequence ─────────────────────────────────────────────
            st.markdown("#### Transport Sequence")
            badge_colors = {
                'Flight': 'badge-flight', 'Train': 'badge-train',
                'Bus': 'badge-bus',       'Cab': 'badge-cab',
            }
            badges_html = ""
            steps_html  = ""
            for i, leg in enumerate(result['Route Details']):
                bc   = badge_colors.get(leg['Transport_Type'], 'badge-cab')
                badges_html += f"<span class='badge {bc}'>{leg['Transport_Type']}</span>"
                arrow = " &rarr; " if i < len(result['Route Details']) - 1 else ""
                steps_html += f"<b>{leg['Mode']}</b>: {leg['Source']} &rarr; {leg['Destination']}{arrow}"

            st.markdown(f"<div class='route-box'>{badges_html}<hr class='divider'>{steps_html}</div>",
                        unsafe_allow_html=True)

            # ── AI Reasoning ───────────────────────────────────────────────
            st.markdown("#### Why This Route?")
            st.info(result['Reasoning'])

            # ── Leg breakdown ──────────────────────────────────────────────
            with st.expander("Show detailed leg breakdown"):
                for i, leg in enumerate(result['Route Details']):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Mode",     f"{leg['Mode']} ({leg['Transport_Type']})")
                    col_b.metric("Price",    f"Rs.{leg['Price']:.0f}")
                    col_c.metric("Duration", f"{leg['Duration']:.0f} min")
                    col_d.metric("Comfort",  f"{leg['Comfort']:.2f}/1.0")
                    if i < len(result['Route Details']) - 1:
                        st.divider()

        except Exception as e:
            st.error(f"Prediction error: {e}")
