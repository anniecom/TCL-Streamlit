import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import io
from PIL import Image

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

def graficos(
    media_u,
    sigma_media,
    quantd_amostras,
    medias_amostrais,
    medias_padronizadas,
    key="bt",
):
    coli1, coli2 = st.columns(2)

    with coli1:
        st.subheader("Escala Original")
        fig1 = plt.figure(figsize=(5, 4))
        ax1 = fig1.add_subplot(111)
        ax1.hist(
            medias_amostrais[:quantd_amostras],
            bins=30,
            density=True,
            alpha=0.6,
            color='#2b5c8f',
        )
        x1 = np.linspace(min(medias_amostrais), max(medias_amostrais), 100)
        ax1.plot(x1, norm.pdf(x1, loc=media_u, scale=sigma_media), 'r-', lw=2)
        ax1.grid(True, alpha=0.1)
        st.pyplot(fig1)
        plt.close(fig1)

    with coli2:
        st.subheader("Escala Padronizada (Z)")

        # Estado para guardar o GIF gerado na sessão
        gif_key = f"gif_{key}"
        if gif_key not in st.session_state:
            st.session_state[gif_key] = None

        animacao = st.button("Ver Animação Histograma", key="bt" + key)
        
        if animacao:
            with st.spinner("Gerando animação..."):
                frames = []
                # Divide em 30 passos para ficar visualmente contínuo e rápido
                passos = np.linspace(10, quantd_amostras, 30, dtype=int)

                fig2 = plt.figure(figsize=(5, 4))
                ax2 = fig2.add_subplot(111)
                x2 = np.linspace(-3.5, 3.5, 100)
                y2 = norm.pdf(x2, loc=0, scale=1)

                for tam in passos:
                    ax2.clear()
                    # Plota apenas as barras subindo
                    ax2.hist(
                        medias_padronizadas[:tam],
                        bins=30,
                        density=True,
                        alpha=0.6,
                        color='#2ecc71',
                    )
                    # Mantém os eixos e a linha estáticos no mesmo lugar do frame
                    ax2.plot(x2, y2, 'r-', lw=2)
                    ax2.set_xlim([-3.5, 3.5])
                    ax2.set_ylim([0, 0.5])
                    ax2.grid(True, alpha=0.1)
                    ax2.set_title(
                        f"Amostras acumuladas: {tam}", fontsize=9
                    )

                    # Salva o frame na memória RAM como objeto de imagem pura
                    buf = io.BytesIO()
                    fig2.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    frames.append(Image.open(buf))
                    # Não fechamos o buffer aqui dentro para o PIL não perder a referência da imagem

                plt.close(fig2)

                # TRUQUE DA PIL: Junta todas as imagens e compila em um GIF na memória
                gif_buffer = io.BytesIO()
                frames[0].save(
                    gif_buffer,
                    format='GIF',
                    save_all=True,
                    append_images=frames[1:],
                    duration=500,  # Tempo de cada frame em milissegundos (100ms = fluido)
                    loop=0,  # 0 significa repetição infinita do GIF
                )
                st.session_state[gif_key] = gif_buffer.getvalue()
                gif_buffer.close()
        
        # Exibição limpa na tela
        if st.session_state[gif_key] is not None:
            # Mostra o GIF. O navegador roda ele liso sem piscar os eixos
            st.image(st.session_state[gif_key], use_container_width=True)
            if st.button("Parar"):
                st.session_state[gif_key] = None
                st.rerun()
        else:
            # Caso padrão: Mostra o gráfico estático final normal
            fig_fixo = plt.figure(figsize=(5, 4))
            ax_fixo = fig_fixo.add_subplot(111)
            ax_fixo.hist(
                medias_padronizadas[:quantd_amostras],
                bins=30,
                density=True,
                alpha=0.6,
                color='#2ecc71',
            )
            ax_fixo.plot(
                np.linspace(-3.5, 3.5, 100),
                norm.pdf(np.linspace(-3.5, 3.5, 100), 0, 1),
                'r-',
                lw=2,
            )
            ax_fixo.set_xlim([-3.5, 3.5])
            ax_fixo.set_ylim([0, 0.5])
            ax_fixo.grid(True, alpha=0.1)
            st.pyplot(fig_fixo)
            plt.close(fig_fixo)

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
    quantd_amostras = 1000
    st.title(f"Casos de Teste para {quantd_amostras} amostras")
    #Define o o tamanho da amostra
    n_amostra = st.number_input("Tamanho da Amostra (n)", value=30, min_value=1)

    dist = {
        ("Binomial", "Probabilidade sucesso",  0.09),
        ("Binomial", "Probabilidade sucesso",  0.5),
        ("Exponencial", "Lambda", 10),
        ("Uniforme", "a, b",  (10.6, 40.3))
    }

    comment=[
        ("A convergência é lenta. Como a probabilidade de sucesso é muito baixa, a distribuição é assimétrica. Necessitando um N maior para que a média amostral consiga convergir para uma Normal."),
        ("A convergência é rápida. Como a distribuição original já é simétrica, o histograma das médias amostrais já vai parecer uma curva Normal."),
        ("A convergência é lenta. Como a distribuição é assimétrica, logo precisa de um número robusto de amostras para que o consiga convergir para uma Normal."),
        ("A convergência é rápida.Como a distribuição é simétrica, a média amostral converge para uma Normal rapidamente.")
    ]
    
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
            st.text(comment[i])

        i += 1

# --- NAVEGAÇÃO ENTRE PÁGINAS POR SIDEBAR ---
pagina = st.sidebar.radio("Ir para:", ["Simulador TCL", "Testes"])

if pagina != "Simulador TCL":
    Teste()
else:
    SimuladorTCL()
