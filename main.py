import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

class GeradorTCL:
    @staticmethod
    def gerar_medias(opt, tam_amostra, m_amostras, params):
        #Gera amostras de tamanho para cada um das distribuicoes
        if opt == "Binomial":
            prob = params
            amostra = np.random.binomial(n=tam_amostra, p=prob, size=(m_amostras, tam_amostra))
            media_dist = tam_amostra * prob
            sigma_dist = np.sqrt(tam_amostra * prob * (1 - prob))

        elif opt == "Bernoulli":
            prob = params
            amostra = np.random.binomial(n=1, p=prob, size=(m_amostras, tam_amostra))
            media_dist = prob
            sigma_dist = np.sqrt(prob * (1 - prob))

        elif opt == "Uniforme":
            a, b = params
            amostra = np.random.uniform(low=a, high=b, size=(m_amostras, tam_amostra))
            media_dist = (a + b) / 2
            sigma_dist = np.sqrt(((b - a) ** 2) / 12)

        elif opt == "Exponencial":
            lambda_ = params
            amostra = np.random.exponential(scale=1/lambda_, size=(m_amostras, tam_amostra))
            media_dist = 1 / lambda_
            sigma_dist = 1 / lambda_

        elif opt == "Poisson":
            lambda_ = params
            amostra = np.random.poisson(lam=lambda_, size=(m_amostras, tam_amostra))
            media_dist = lambda_
            sigma_dist = np.sqrt(lambda_)
        else:
            return None, 0, 0

        # Calcula a média de cada uma das 'm' amostras
        medias_amostrais = amostra.mean(axis=1)
        #Retorna as médias amostrais, média e sigima teórcios da distribuiçcap
        return medias_amostrais, media_dist, sigma_dist

def graficos(media_u, sigma_media, quantd_amostras,medias_amostrais,medias_padronizadas, key = "bt"):
    #Faz os graficos de histograma da normal
    coli1, coli2 = st.columns(2)

    with coli1:
        #Histograma dos dados nao padronizados
        st.subheader("Escala Original")
        fig1 = plt.figure(figsize=(5, 4))
        ax1 = fig1.add_subplot(111)
        posc_i = quantd_amostras - 1

        contagens, intervalos, _ = ax1.hist(
            medias_amostrais[:posc_i], bins=30, density=True, alpha=0.6, color='#2b5c8f', label='Médias Amostrais'
        )

        x1 = np.linspace(intervalos.min(), intervalos.max(), 100)
        y1 = norm.pdf(x1, loc=media_u, scale=sigma_media)
        ax1.plot(x1, y1, 'r-', lw=2, label='Normal Teórica')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.1)

        st.pyplot(fig1)
        plt.close(fig1)

    with coli2:
        #Histograma dos dados padronizados
        st.subheader("Escala Padronizada (Z)")
        espaco_grafico_z = st.empty()

        def desenhar_grafico_z(tamanho_atual):
            fig2 = plt.figure(figsize=(5, 4))
            ax2 = fig2.add_subplot(111)

            contagens2, intervalos2, _ = ax2.hist(
                medias_padronizadas[:tamanho_atual], bins=30, density=True, alpha=0.6, color='#2ecc71'
            )

            x2 = np.linspace(intervalos2[0], intervalos2[-1], 100)
            y2 = norm.pdf(x2, loc=0, scale=1)
            ax2.plot(x2, y2, 'r-', lw=2, label='N(0,1)')

            ax2.set_xlim([min(-3.5, intervalos2[0]), max(3.5, intervalos2[-1])])
            ax2.grid(True, alpha=0.1)
            ax2.legend(fontsize=8)

            espaco_grafico_z.pyplot(fig2)
            plt.close(fig2)

        animacao = st.button("Ver Animação Histograma", key = "bt" + key)
        posc_slider = st.slider("Quantidade de amostras (Z):", 10, quantd_amostras, quantd_amostras, key= "sl" + key)

        if animacao:
            passos = np.linspace(10, quantd_amostras, 20, dtype=int)
            for tam in passos:
                desenhar_grafico_z(tam)
                time.sleep(0.05)
        else:
            desenhar_grafico_z(posc_slider)

def SimuladorTCL():
    st.title("Simulador TCL")
    #Simula o TCL para as distribuições
    col_1, col_2 = st.columns(2)
    opcoes_dist = ["Binomial", "Bernoulli", "Uniforme", "Poisson", "Exponencial"]
    
    with col_1:
        #Define a distribuição
        st.text("Distribuição:")
        opt = st.selectbox("Selecione a Distribuição", opcoes_dist)
        
        #Define parametros
        if opt in ["Uniforme"]:
            col1, col2 = st.columns(2)
            with col1: a = st.number_input("Valor mínimo (a)", value=1.0)
            with col2: b = st.number_input("Valor máximo (b)", value=10.0)
            if a >= b: a, b = b, a
            parametros = (a, b)

        elif opt in ["Exponencial", "Poisson"]:
            parametros = st.number_input("Taxa Lambda", value=0.5, min_value=0.001, step=0.5)
            
        elif opt in ["Binomial", "Bernoulli"]:
            prob = st.number_input("Probabilidade de Sucesso (p)", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
            parametros = prob
        
    with col_2: 
        st.text("Amostra:")
        #Define tamanho e quantidade das amostras
        n_amostra = st.number_input("Tamanho da Amostra (n)", value=30, min_value=1)
        quantd_amostras = st.number_input("Quantidade de Amostras (m)", value=1000, min_value=20)

    if "dados_gerados" not in st.session_state:
        st.session_state.dados_gerados = None

    simulacao = st.button("Simular")

    if simulacao:
        #Define as medias amostrais
        medias_amostrais, media_u, sigma = GeradorTCL.gerar_medias(opt, n_amostra, quantd_amostras, parametros)
        #Padroniza as medias
        sigma_da_media = sigma / np.sqrt(n_amostra)
        medias_padronizadas = (medias_amostrais - media_u) / sigma_da_media
        
        #Cache Streamlit
        st.session_state.dados_gerados = {
            "medias_amostrais": medias_amostrais,
            "medias_padronizadas": medias_padronizadas,
            "media_u": media_u,
            "sigma_media": sigma_da_media,
            "quantd_amostras": quantd_amostras
        }

    #Produz os graficos
    if st.session_state.dados_gerados is not None:
        dados = st.session_state.dados_gerados
        graficos(dados["media_u"], dados["sigma_media"], dados["quantd_amostras"], dados["medias_amostrais"], dados["medias_padronizadas"], "tcl")
        


def Teste():
    quantd_amostras = 600
    st.title(f"Casos de Teste para {quantd_amostras} amostras")
    #Define o o tamanho da amostra
    n_amostra = st.number_input("Tamanho da Amostra (n)", value=30, min_value=1)

    dist = {
        ("Binomial", "Probabilidade sucesso", 0.05),
        ("Binomial", "Probabilidade sucesso",  0.1),
        ("Binomial", "Probabilidade sucesso",  0.5),
        ("Exponencial", "Lambda", 10),
        ("Uniforme", "a, b",  (10.6, 40.3))
    }
    
    abas = st.tabs([f"{d[0]} ({d[2]})" for d in dist])
    i = 0
    for d in dist:
        #Para cada teste simula o TCL
        opt, tipo_p, param = d

        with abas[i]:
            st.subheader(f"Teste: {opt} | {tipo_p} = {param}")
            medias_amostrais, media_u, sigma = GeradorTCL.gerar_medias(opt, n_amostra, quantd_amostras, param)
            sigma_media = sigma / np.sqrt(n_amostra)
            medias_padronizadas = (medias_amostrais - media_u) / sigma_media
            graficos(media_u, sigma_media, quantd_amostras, medias_amostrais, medias_padronizadas, (opt + str(i)))

        i += 1

# --- NAVEGAÇÃO ENTRE PÁGINAS POR SIDEBAR ---
pagina = st.sidebar.radio("Ir para:", ["Simulador TCL", "Testes (i, ii, iii)"])

if pagina == "Simulador TCL":
    SimuladorTCL()
else:
    Teste()
