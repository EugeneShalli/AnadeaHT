import streamlit as st


def app():
    link = '[W&B log](https://wandb.ai/eugeneshally/livecell?workspace=user-eugeneshally)'
    st.markdown(link, unsafe_allow_html=True)


def main():
    st.markdown("# 📄 Experiments report")
    st.sidebar.markdown("# 📄 Experiments report")
    app()


if __name__ == "__main__":
    main()
