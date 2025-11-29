import streamlit as st

def load_theme():
    st.markdown("""
        <style>
            /* 🌈 Background gradient and base layout */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(90deg, rgba(42, 123, 155, 1) 0%, rgba(87, 199, 133, 1) 50%, rgba(237, 221, 83, 1) 100%);
                color: #002B36;
                font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            }

            /* 🧭 Sidebar styling */
            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.85);
                backdrop-filter: blur(12px);
                border-right: 1px solid rgba(255,255,255,0.2);
            }

            /* 🏷️ Headers */
            h1, h2, h3, h4, h5, h6 {
                color: #0D3B66;
                font-weight: 700;
                letter-spacing: 0.5px;
            }

            h1 {
                text-align: center;
                color: #0D3B66;
                font-size: 2.5rem;
                margin-bottom: 0.8em;
                text-shadow: 1px 1px 2px rgba(255,255,255,0.4);
            }

            /* 🧊 Main content container */
            .block-container {
                padding-top: 3rem !important;
                padding-bottom: 3rem !important;
                background: rgba(255, 255, 255, 0.85);
                border-radius: 20px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.1);
                margin: 2rem auto;
                max-width: 95%;
            }

            /* ✨ Buttons */
            .stButton>button {
                background: linear-gradient(90deg, rgba(42,123,155,1) 0%, rgba(87,199,133,1) 100%);
                color: white !important;
                border-radius: 10px;
                padding: 0.6em 1.2em;
                border: none;
                font-weight: 600;
                transition: all 0.3s ease-in-out;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            }

            .stButton>button:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 16px rgba(0,0,0,0.2);
                background: linear-gradient(90deg, rgba(87,199,133,1) 0%, rgba(237,221,83,1) 100%);
            }

            /* 📦 Input boxes and widgets */
            .stTextInput>div>div>input,
            .stSelectbox>div>div>select,
            .stSlider>div>div>div>input {
                background-color: white !important;
                border-radius: 10px !important;
                border: 1px solid rgba(42, 123, 155, 0.4) !important;
                color: #002B36 !important;
                padding: 0.4em 0.8em;
                transition: all 0.2s ease-in-out;
            }

            .stTextInput>div>div>input:focus,
            .stSelectbox>div>div>select:focus {
                border-color: #57C785 !important;
                box-shadow: 0 0 0 3px rgba(87,199,133,0.3);
            }

            /* 📊 DataFrame styling */
            .stDataFrame {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }

            /* 🔗 Links */
            a {
                color: #0D3B66 !important;
                text-decoration: none;
                font-weight: 600;
            }

            a:hover {
                color: #57C785 !important;
                text-decoration: underline;
            }

            /* 🧩 Tabs */
            div[data-baseweb="tab-list"] {
                gap: 10px;
            }

            button[data-baseweb="tab"] {
                background: rgba(255,255,255,0.8);
                border-radius: 12px;
                padding: 10px 20px;
                font-weight: 600;
                border: 1px solid rgba(42,123,155,0.2);
                transition: all 0.25s ease-in-out;
            }

            button[data-baseweb="tab"]:hover {
                background: rgba(87,199,133,0.1);
                border-color: rgba(87,199,133,0.4);
            }

            button[data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(90deg, rgba(42,123,155,1) 0%, rgba(87,199,133,1) 100%);
                color: white !important;
                border: none;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }

            /* 🩵 Small tweaks for charts and spacing */
            .plotly {
                border-radius: 15px !important;
                background: white !important;
                padding: 1rem;
            }
              /* 🤖 Chatbot Button Styling */
.chatbot-btn {
    width: 100%;
    padding: 10px 18px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(90deg, rgba(42,123,155,1) 0%, rgba(87,199,133,1) 100%);
    color: white !important;
    font-weight: 600;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease-in-out;
}

.chatbot-btn:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, rgba(87,199,133,1) 0%, rgba(237,221,83,1) 100%);
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

        </style>
    """, unsafe_allow_html=True)
