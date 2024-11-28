import streamlit as st
from ml_app import run_ml_app

def main():
    menu = ['Home','Machine Learning']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':
        st.subheader('Welcome to Homepage')
    elif choice == 'Machine Learning':
        # st.subheader('Machine Learning')
        run_ml_app()

if __name__ == '__main__':
    main()