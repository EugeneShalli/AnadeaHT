import streamlit as st
from streamlit_option_menu import option_menu
from pages_custom import inference_page, home_page, report_page


if __name__ == "__main__":

    st.set_page_config(
        page_title="Anadea: LIVECell",
        page_icon="🦠",
        layout="wide",
    )

    with st.sidebar:
        selected = option_menu(None, ["Main Page", "Inference", "Report"],
                                icons=['house', "play", "file-earmark-text"],
                                menu_icon="cast", default_index=0)

    if selected == "Main Page":
        home_page.main()
    if selected == "Inference":
        inference_page.main()
    if selected == "Report":
        report_page.main()
