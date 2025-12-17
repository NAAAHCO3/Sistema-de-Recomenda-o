import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors # Mudança aqui!

# 1. Configuração da Página
st.set_page_config(page_title="Recomendador de Músicas", layout="wide")
st.title("🎵 AI Music Recommender")
st.markdown("Descubra músicas novas baseadas na ciência de dados (Billboard, Spotify, Rádio).")

# 2. Carregar os Dados
@st.cache_data
def carregar_dados():
    try:
        with open('dados_musica.pkl', 'rb') as f:
            df = pickle.load(f)
        return df
    except FileNotFoundError:
        return None

df = carregar_dados()

if df is None:
    st.error("Erro: Arquivo 'dados_musica.pkl' não encontrado no repositório.")
    st.stop()

# 3. Engenharia Leve (Sem Matriz Gigante)
@st.cache_resource
def treinar_modelo_leve(df_input):
    features = [
        'Hot100_Score', 'Radio_Score', 'Streaming_Score', 'Digital_Score', 
        'Weeks in Charts', 'Radio_Weeks', 'Streaming_Weeks', 'Digital_Weeks',
        'Album_Counts', 'Year'
    ]
    
    # Preparar dados numéricos
    df_modelo = df_input[features].fillna(0)
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(df_modelo)
    
    # EM VEZ DE CALCULAR TUDO, APENAS TREINAMOS O BUSCADOR
    # Isso gasta muito menos memória
    modelo_nn = NearestNeighbors(n_neighbors=11, metric='cosine')
    modelo_nn.fit(dados_norm)
    
    return modelo_nn, dados_norm

# Treina o modelo leve ao iniciar
modelo_nn, dados_norm = treinar_modelo_leve(df)

# 4. Função de Recomendação Otimizada
def recomendar(termo, df, modelo, matriz_dados):
    termo = termo.lower()
    songs_lower = df['Song'].str.lower()
    artists_lower = df['Artist'].str.lower()
    
    idx_alvo = None
    msg = ""

    # Busca Música
    matches_song = df[songs_lower == termo]
    if not matches_song.empty:
        idx_alvo = matches_song.index[0]
        msg = f"Baseado na música **{df.loc[idx_alvo, 'Song']}**:"
    else:
        # Busca Artista
        matches_artist = df[artists_lower == termo]
        if not matches_artist.empty:
            top_track = matches_artist.sort_values(by='Hot100_Score', ascending=False).iloc[0]
            idx_alvo = top_track.name
            msg = f"Artista encontrado! Usando o hit **{top_track['Song']}** como referência:"
        else:
            return None, "Não encontrado."

    # A MÁGICA: O modelo calcula os vizinhos SÓ AGORA, e só para esse item
    # Retorna distâncias e índices
    distances, indices = modelo.kneighbors([matriz_dados[idx_alvo]])
    
    # O indices[0] é uma lista com os 11 índices mais próximos
    # indices[0][0] é a própria música, então pegamos do 1 ao 11
    top_indices = indices[0][1:]
    
    return df.iloc[top_indices], msg

# 5. Interface
input_usuario = st.text_input("Digite uma música ou artista:", placeholder="Ex: Adele, Queen, Toxic...")

if st.button("Recomendar"):
    if input_usuario:
        resultados, mensagem = recomendar(input_usuario, df, modelo_nn, dados_norm)
        
        if resultados is not None:
            st.success(mensagem)
            st.dataframe(
                resultados[['Song', 'Artist', 'Year', 'Hot100_Score', 'Album_Counts']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("Ops! Não encontramos na base de dados.")