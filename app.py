import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador de Músicas", layout="wide")

st.title("🎵 AI Music Recommender")
st.markdown("Descubra músicas novas baseadas na ciência de dados (Billboard, Spotify, Rádio).")

@st.cache_data
def carregar_dados():
    with open('dados_musica.pkl', 'rb') as f:
        df = pickle.load(f)
    return df

try:
    df = carregar_dados()
except FileNotFoundError:
    st.error("Arquivo 'dados_musica.pkl' não encontrado. Rode o notebook primeiro!")
    st.stop()

@st.cache_resource
def preparar_modelo(df_input):
    features = [
        'Hot100_Score', 'Radio_Score', 'Streaming_Score', 'Digital_Score', 
        'Weeks in Charts', 'Radio_Weeks', 'Streaming_Weeks', 'Digital_Weeks',
        'Album_Counts', 'Year'
    ]
    
    df_modelo = df_input[features].fillna(0)
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(df_modelo)
    
    matriz = cosine_similarity(dados_norm)
    return matriz

with st.spinner('Ligando os motores da IA...'):
    sim_matrix = preparar_modelo(df)

def recomendar(termo, df, matriz):
    termo = termo.lower()
    songs_lower = df['Song'].str.lower()
    artists_lower = df['Artist'].str.lower()
    
    idx_alvo = None
    msg = ""

    matches_song = df[songs_lower == termo]
    if not matches_song.empty:
        idx_alvo = matches_song.index[0]
        msg = f"Baseado na música **{df.loc[idx_alvo, 'Song']}**:"
    else:
        matches_artist = df[artists_lower == termo]
        if not matches_artist.empty:
            top_track = matches_artist.sort_values(by='Hot100_Score', ascending=False).iloc[0]
            idx_alvo = top_track.name
            msg = f"Artista encontrado! Usando o hit **{top_track['Song']}** como referência:"
        else:
            return None, "Não encontrado."

    scores = list(enumerate(matriz[idx_alvo]))
    scores_ordenados = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores_ordenados[1:11]]
    
    return df.iloc[top_indices], msg

input_usuario = st.text_input("Digite uma música ou artista que você ama:", placeholder="Ex: Adele, Toxic, Queen...")

if st.button("Recomendar"):
    if input_usuario:
        resultados, mensagem = recomendar(input_usuario, df, sim_matrix)
        
        if resultados is not None:
            st.success(mensagem)
            st.dataframe(
                resultados[['Song', 'Artist', 'Year', 'Hot100_Score']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("Ops! Não encontramos essa música ou artista na base de dados.")
    else:
        st.warning("Por favor, digite algo antes de buscar.")