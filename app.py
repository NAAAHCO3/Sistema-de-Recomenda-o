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

# 4. Função de Recomendação com "Artist Boost"
def recomendar(termo, df, modelo, matriz_dados):
    termo = termo.lower().strip()
    songs_lower = df['Song'].str.lower()
    artists_lower = df['Artist'].str.lower()
    
    idx_alvo = None
    msg = ""
    artista_alvo = ""

    # --- ETAPA 1: LOCALIZAR O ALVO ---
    matches_song = df[songs_lower.str.contains(termo, na=False)]
    if not matches_song.empty:
        idx_alvo = matches_song.sort_values(by='Hot100_Score', ascending=False).index[0]
        artista_alvo = df.loc[idx_alvo, 'Artist'] # Guardamos o nome do artista
        msg = f"Baseado na música **{df.loc[idx_alvo, 'Song']}**:"
    
    else:
        matches_artist = df[artists_lower.str.contains(termo, na=False)]
        if not matches_artist.empty:
            top_track = matches_artist.sort_values(by=['Hot100_Score', 'Weeks in Charts'], ascending=[False, False]).iloc[0]
            idx_alvo = top_track.name
            artista_alvo = top_track['Artist']
            msg = f"Artista encontrado! Usando o megahit **{top_track['Song']}** como referência:"
        else:
            return None, "Não encontrado."

    # --- ETAPA 2: BUSCAR VIZINHOS (MATH) ---
    # Pedimos 50 vizinhos agora (para ter margem de escolha)
    distances, indices = modelo.kneighbors([matriz_dados[df.index.get_loc(idx_alvo)]], n_neighbors=50)
    
    # Indices dos vizinhos (ignorando o primeiro que é a própria âncora)
    vizinhos_indices = indices[0][1:]
    
    # --- ETAPA 3: FILTRAGEM INTELIGENTE (BUSINESS LOGIC) ---
    # Pegamos as linhas do dataframe correspondentes
    vizinhos_df = df.iloc[vizinhos_indices].copy()
    
    # Separamos em dois grupos
    do_mesmo_artista = vizinhos_df[vizinhos_df['Artist'] == artista_alvo]
    de_outros = vizinhos_df[vizinhos_df['Artist'] != artista_alvo]
    
    # Estratégia de Mix: Garantir até 3 do mesmo artista, completar com outros
    recomendacoes_finais = pd.concat([
        do_mesmo_artista.head(3),  # Pega até 3 do mesmo artista
        de_outros.head(7)          # Completa com 7 de outros
    ])
    
    # Se tiver menos de 10 no total, completa com o que tiver
    if len(recomendacoes_finais) < 10:
        recomendacoes_finais = vizinhos_df.head(10)
    
    return recomendacoes_finais, msg

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