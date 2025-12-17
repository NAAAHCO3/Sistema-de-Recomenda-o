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

# 4. Função de Recomendação Blindada (Garantia de Artista)
def recomendar(termo, df, modelo, matriz_dados):
    termo = termo.lower().strip()
    songs_lower = df['Song'].str.lower()
    artists_lower = df['Artist'].str.lower()
    
    idx_alvo = None
    msg = ""
    artista_alvo = ""
    nome_musica_alvo = ""

    # --- ETAPA 1: LOCALIZAR O ALVO (Música ou Artista) ---
    matches_song = df[songs_lower.str.contains(termo, na=False)]
    
    if not matches_song.empty:
        # Prioriza match exato se houver, senão pega o primeiro parcial
        match_exato = matches_song[songs_lower == termo]
        if not match_exato.empty:
            matches_song = match_exato
            
        idx_alvo = matches_song.sort_values(by='Hot100_Score', ascending=False).index[0]
        artista_alvo = df.loc[idx_alvo, 'Artist']
        nome_musica_alvo = df.loc[idx_alvo, 'Song']
        msg = f"Baseado na música **{nome_musica_alvo}**:"
    
    else:
        matches_artist = df[artists_lower.str.contains(termo, na=False)]
        if not matches_artist.empty:
            # Pega o match exato de artista se possível
            match_exato = matches_artist[artists_lower == termo]
            if not match_exato.empty:
                matches_artist = match_exato
                
            top_track = matches_artist.sort_values(by=['Hot100_Score', 'Weeks in Charts'], ascending=[False, False]).iloc[0]
            idx_alvo = top_track.name
            artista_alvo = top_track['Artist']
            nome_musica_alvo = top_track['Song']
            msg = f"Artista encontrado! Usando o megahit **{nome_musica_alvo}** como referência:"
        else:
            return None, "Não encontrado."

    # --- ETAPA 2: BALDE A - DO MESMO ARTISTA (Busca Direta) ---
    # Aqui está a correção: buscamos no DF inteiro, não só nos vizinhos
    # Filtramos pelo artista e removemos a música que usamos de âncora
    df_artista = df[
        (df['Artist'] == artista_alvo) & 
        (df['Song'] != nome_musica_alvo)
    ].sort_values(by='Hot100_Score', ascending=False)
    
    # Pegamos até 3 músicas dele
    recomendacoes_artista = df_artista.head(3)

    # --- ETAPA 3: BALDE B - DESCOBERTA (KNN) ---
    # Buscamos vizinhos matemáticos
    distances, indices = modelo.kneighbors([matriz_dados[df.index.get_loc(idx_alvo)]], n_neighbors=20)
    vizinhos_indices = indices[0][1:] # Ignora a própria âncora
    
    vizinhos_df = df.iloc[vizinhos_indices].copy()
    
    # Removemos o próprio artista dessa lista para dar espaço aos outros
    recomendacoes_outros = vizinhos_df[vizinhos_df['Artist'] != artista_alvo].head(7)
    
    # --- ETAPA 4: MIX FINAL ---
    recomendacoes_finais = pd.concat([recomendacoes_artista, recomendacoes_outros])
    
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