import streamlit as st
from home import show_home
from prediction import show_prediction
from visualization import show_visualization  # if you have it

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

# -------------------- SESSION STATE NAVIGATION --------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def navigate(page):
    st.session_state.page = page

# -------------------- TOP NAVIGATION BAR --------------------
col1, col2, col3,col4 = st.columns([1, 1, 1,1])
with col1:
    if st.button("🏠 Home", key="home_nav", use_container_width=True):
        navigate("home")
with col2:
    if st.button("🎯 Prediction", key="pred_nav", use_container_width=True):
        navigate("prediction")
with col3:
    if st.button("📊 Visualization", key="viz_nav", use_container_width=True):
        navigate("visualization")
with col4:
    st.markdown(
        "<a href='http://127.0.0.1:5000/chatbot' target='_blank'>"
        "<button class='chatbot-btn'>🤖 Chatbot</button>"
        "</a>",
        unsafe_allow_html=True
    )

# -------------------- RENDER SELECTED PAGE --------------------
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "prediction":
    show_prediction()
elif st.session_state.page == "visualization":
    show_visualization()
