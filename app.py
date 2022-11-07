import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_shap import st_shap

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import joblib
import shap
from lightgbm import LGBMClassifier
import requests

#Téléchargement du jeu de données échantillon
df = pd.read_csv('data_sample.csv')
df_ml = pd.read_csv('data_ml_sample.csv')

#Utilisation de la largeur de la page
st.set_page_config(layout="wide")

st.sidebar.image('pretadepenser.png')

#Création d'un menu de navigation entre les pages
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Fichier clients', 'Client individuel', 'Comparaison'],
        icons=['folder2-open', 'file-person', 'graph-up'],
        menu_icon='filter',
        default_index=0
        )

st.sidebar.markdown('''
---
Créée par Raphaël Lagarde
''')


#Page 1 : Fichier clients
if selected == 'Fichier clients':
    st.title(f'Onglet {selected}')
    st.write("""### Veuillez trouvez ci-dessous le fichier clients avec filtres
    Informations filtres :
        Target = 1 : le client a des difficultés de paiement
        Filtre Montant Income : choisir un montant maximal
    """)
    
    #Création de colonnes pour les filtre
    col1, col2, col3, col4, col5, col6 = st.columns(6, gap='small')
    #Création des filtres
    target_filter = col1.multiselect(
        'Filtre Target', 
        options=df['TARGET'].unique().tolist(),
        default=df['TARGET'].unique().tolist())
    gender_filter = col2.multiselect(
        'Filtre Genre',
        options=df['CODE_GENDER'].unique().tolist(),
        default=df['CODE_GENDER'].unique().tolist())
    car_filter = col3.multiselect(
        'Filtre Voiture', 
        options=df['FLAG_OWN_CAR'].unique().tolist(),
        default=df['FLAG_OWN_CAR'].unique().tolist())
    contract_filter = col4.multiselect(
        'Filtre Contrat', 
        options=df['NAME_CONTRACT_TYPE'].unique().tolist(),
        default=df['NAME_CONTRACT_TYPE'].unique().tolist())
    income_filter = col5.multiselect(
        'Filtre Income', 
        options=df['NAME_INCOME_TYPE'].unique().tolist(),
        default=df['NAME_INCOME_TYPE'].unique().tolist())
    amt_income_filter = col6.selectbox(
        'Filtre Montant Income',
        options=list(range(0, 450000, 50000)),
        index=8)
    st.write('---')
    #Jeu de données filtré
    df_selection = df.query("TARGET==@target_filter & CODE_GENDER==@gender_filter & FLAG_OWN_CAR==@car_filter & NAME_CONTRACT_TYPE==@contract_filter & NAME_INCOME_TYPE==@income_filter & AMT_INCOME_TOTAL<=@amt_income_filter")
    st.dataframe(df_selection)



#Page 2 : Client Individuel
if selected == 'Client individuel':
    st.title(f'Onglet {selected}')
    st.write("""### Veuillez trouver ci-dessous la prédiction faite pour le client choisi
    Informations jauge :
        Le seuil de décision est de 0.55, ce dernier nous permet de faire de meilleures prédictions
    """)

    client_selection = st.selectbox('Choisir un identifiant client', df['SK_ID_CURR'])
    with st.expander('Informations clients'):
        st.dataframe(df[df['SK_ID_CURR']==client_selection])
    st.write('---')
    
    idx_client = df_ml.loc[df_ml['SK_ID_CURR']==client_selection].index[0]
    client_data = df_ml.drop(columns='SK_ID_CURR').loc[idx_client]
    client_dict = dict(client_data)

    r = requests.post(url='https://fastapip7.herokuapp.com/predict/', json=client_dict)
    proba = round(float(r.json().replace("[", "").replace("]", "")), 2)
    
    if proba < 0.55:
        st.error('Le prêt est refusé')
    else:
        st.success('Le prêt est accordé')
    #Radar plot
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = proba,
        mode = 'gauge+number+delta',
        title = {'text': 'Probabilité'},
        gauge = {'axis': {'range': [0, 1]},
                'bar': {'color': 'gainsboro'},
                'steps' : [
                    {'range': [0, 0.55], 'color': 'firebrick'},
                    {'range': [0.55, 1], 'color': 'mediumseagreen'}]}))

    st.plotly_chart(fig, use_container_width=True)

    #Téléchargement du modèle
    model = joblib.load('p7_pipeline.joblib')

    X_train = df_ml.loc[df_ml['SK_ID_CURR']!=client_selection].drop(columns='SK_ID_CURR')
    X = df_ml.loc[df_ml['SK_ID_CURR']==client_selection].drop(columns='SK_ID_CURR')
    #Shap plot
    X_train_scaled = model[0].transform(X_train)
    X_scaled = model[0].transform(X)

    explainer = shap.KernelExplainer(model[1].predict_proba, X_train_scaled)
    shap_values = explainer.shap_values(X_scaled)

    st.write("""###
    Voici le détail de l'impact des informations sur la probabilité
    """)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], X_scaled, feature_names=X_train.columns, plot_cmap=['#3CB371', '#B22222']))
    
    st.write("""###
    Ci-dessous un graphique plus intuitif pour comprendre l'impact négatif (bleu) et positif (rose) sur la probabilité
    """)
    st_shap(shap.bar_plot(shap_values[1][0], feature_names=X_train.columns, max_display=15), width=1000)

#Page3 : Comparaison    
if selected == 'Comparaison':
    st.title(f'Onglet {selected}')
    st.write("""### Veuillez trouver ci-dessous les comparaisons de certaines informations du client choisi
    La ligne verte nous indique où se trouve le client dans la distribution
    Rappel TARGET :
        1 = le client a des difficultés de paiement
    """)

    client_selection = st.selectbox('Choisir un identifiant client', df['SK_ID_CURR'])
    st.write('---')

    c1, c2 = st.columns(2)

    fig1 = plt.figure()
    sns.kdeplot(data=df, x='AMT_INCOME_TOTAL', hue='TARGET')
    plt.axvline(df.loc[df['SK_ID_CURR']==client_selection, 'AMT_INCOME_TOTAL'].values[0], color='green')
    plt.ylabel(None)
    plt.yticks([], [])
    plt.xlabel(None)
    plt.gca().spines['left'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.title('AMT Income Total', fontsize=12)
    c1.pyplot(fig1)

    fig2 = plt.figure()
    sns.kdeplot(data=df, x='DAYS_BIRTH', hue='TARGET')
    plt.axvline(df.loc[df['SK_ID_CURR']==client_selection, 'DAYS_BIRTH'].values[0], color='green')
    plt.ylabel(None)
    plt.yticks([], [])
    plt.xlabel(None)
    plt.gca().spines['left'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.title('Age', fontsize=12)
    c2.pyplot(fig2)

    c1, c2 = st.columns(2)

    fig3 = plt.figure()
    sns.kdeplot(data=df, x='DAYS_EMPLOYED_PERC', hue='TARGET')
    plt.axvline(df.loc[df['SK_ID_CURR']==client_selection, 'DAYS_EMPLOYED_PERC'].values[0], color='green')
    plt.ylabel(None)
    plt.yticks([], [])
    plt.xlabel(None)
    plt.gca().spines['left'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.title('% Days employed', fontsize=12)

    c1.pyplot(fig3)
    fig4 = plt.figure()
    sns.kdeplot(data=df, x='PAYMENT_RATE', hue='TARGET')
    plt.axvline(df.loc[df['SK_ID_CURR']==client_selection, 'PAYMENT_RATE'].values[0], color='green')
    plt.ylabel(None)
    plt.yticks([], [])
    plt.xlabel(None)
    plt.gca().spines['left'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.title('Payment rate', fontsize=12)
    c2.pyplot(fig4)