import streamlit as st

def main():
    st.set_page_config(page_title="piano_to_note", page_icon="🎹", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Accueil", "À propos"])

    if page == "Accueil":
        # Importation dynamique du module home
        import home as home
        home.page_home()
    elif page == "À propos":
        # Importation dynamique du module A_propos
        import A_propos as A_propos
        A_propos.page_about()

if __name__ == "__main__":
    main()
