# ====================================================================
# SMARTLEAN IA - PLATAFORMA EDUCACIONAL DE INDÚSTRIA 4.0 & LEAN
# Aperfeiçoamento Profissional em Inteligência Artificial Generativa
# SENAI-SP | Licença MIT | Foco: Chão de Fábrica & Manufatura Avançada
# ====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# 1. Machine Learning Clássico (Scikit-Learn)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPRegressor

# 2. Processamento de Linguagem Natural (NLTK)
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
except ImportError:
    SentimentIntensityAnalyzer = None

# 3. Visão Computacional (OpenCV)
try:
    import cv2
except ImportError:
    cv2 = None

# 4. IA Generativa na Nuvem
from openai import OpenAI

# ====================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ====================================================================
st.set_page_config(
    page_title="SmartLean IA - Indústria 4.0 & Chão de Fábrica",
    page_icon="⚙️",
    layout="wide"
)

# ====================================================================
# FUNÇÃO DE RODAPÉ EDUCACIONAL & AVISO LEGAL
# ====================================================================
def exibir_rodape_educacional():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85em;'>
        <p><b>🎓 Projeto Educacional de Código Aberto (Open Source) — Licença MIT</b><br>
        Desenvolvido para o curso de <i>Aperfeiçoamento Profissional em Programação de Inteligência Artificial</i> — <b>SENAI-SP</b>.</p>
        <p>⚠️ <b>Aviso Legal / Disclaimer:</b> Este software tem finalidade estritamente didática e acadêmica para o ensino de Lean Manufacturing e IA. 
        As simulações de OEE, controle de qualidade e manutenção preditiva <u>não substituem</u> calibrações de instrumentos metrológicos (Inmetro/ISO) ou laudos de engenharia de produção.</p>
    </div>
    """, unsafe_allow_html=True)

# ====================================================================
# MENU LATERAL (SIDEBAR)
# ====================================================================
st.sidebar.title("⚙️ SmartLean IA")
st.sidebar.caption("Inteligência Artificial Aplicada ao Chão de Fábrica")

menu = st.sidebar.radio(
    "Módulos da Manufatura Enxuta:",
    [
        "🏠 Início & Ementa Lean", 
        "📈 1. Regressão (Previsão de OEE & Lead Time)", 
        "🚦 2. Classificação (Qualidade Poka-Yoke)", 
        "📊 3. Clusterização (Gargalos de Produção)", 
        "🧠 4. Deep Learning (Manutenção TPM CNC)", 
        "📝 5. PLN & 5S (Auditoria & Kaizen)", 
        "👁️ 6. Visão Computacional (Defeitos em Peças)", 
        "💬 7. IA Generativa (Assistente POP & NR-12)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("📜 Código Aberto sob Licença MIT")
st.sidebar.caption("SENAI-SP — Formação Inicial e Continuada")

# ====================================================================
# MÓDULO 0: INÍCIO E EMENTA
# ====================================================================
if menu == "🏠 Início & Ementa Lean":
    st.title("Bem-vindo ao SmartLean IA 🚀🏭")
    st.subheader("Laboratório Prático de Inteligência Artificial e Indústria 4.0")
    
    st.info("Este ambiente integra os conceitos do Sistema Toyota de Produção (Lean Manufacturing) com algoritmos modernos de Inteligência Artificial para eliminar os 7 desperdícios da manufatura.")
    
    st.markdown("""
    ### 📚 Competências e Ferramentas Industriais Desenvolvidas:
    * **Previsão de OEE & Lead Time (Regressão):** Modelagem de perdas de velocidade, paradas de setup (SMED) e taxa de qualidade.
    * **Inspeção de Qualidade Poka-Yoke (Classificação):** Tomada de decisão em tempo real com tolerâncias micrométricas (Aprovado / Retrabalho / Refugo).
    * **Mapeamento de Gargalos (Clusterização):** Identificação de células de montagem críticas e balanceamento de linha via K-Means e Teoria das Restrições.
    * **Manutenção Preditiva TPM (Deep Learning):** Rede Neural MLP para monitoramento de vibração, temperatura e corrente de fusos em tornos CNC.
    * **Gestão Visual Andon & Kaizen (PLN):** Mineração de relatos de chão de fábrica e auditorias de 5S com dicionário léxico industrial em português.
    * **Inspeção Óptica de Usinagem (Visão Computacional):** Detecção de trincas, rebarbas e furos fora de posição em peças metálicas com OpenCV.
    * **Assistente de Chão de Fábrica & NR-12 (IA Generativa):** Chatbot conectado a Procedimentos Operacionais Padrão (POP) na nuvem Azure.
    """)
    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 1: REGRESSÃO (OEE & LEAD TIME)
# ====================================================================
elif menu == "📈 1. Regressão (Previsão de OEE & Lead Time)":
    st.title("📈 Previsão de OEE (Eficiência Global do Equipamento) & Lead Time")
    st.caption("Modelagem não-linear via Random Forest Regressor integrando Disponibilidade, Performance e Qualidade")
    
    aba_dados, aba_simulador, aba_metricas = st.tabs(["📂 Base de Turnos de Produção", "⚙️ Simulador de Linha", "📊 Validação do Modelo"])
    
    # Base sintética calibrada de 350 turnos fabris
    @st.cache_data
    def gerar_dados_oee():
        np.random.seed(42)
        n = 350
        disp = np.random.uniform(70.0, 98.0, n)       # % Disponibilidade (Horas rodando / planejadas)
        perf = np.random.uniform(65.0, 99.0, n)       # % Desempenho (Velocidade da linha vs padrão)
        qual = np.random.uniform(85.0, 99.9, n)       # % Taxa de Qualidade (Peças boas / total)
        temp_proc = np.random.uniform(20.0, 85.0, n)  # Temperatura do processo (°C)
        
        # Equação Lean de OEE = Disp * Perf * Qual + Ruído de processo
        oee_real = (disp * perf * qual) / 10000.0 + np.random.normal(0, 1.2, n)
        oee_real = np.clip(oee_real, 0.0, 100.0)
        
        # Lead Time em minutos
        lead_time = (5000.0 / (perf + 1.0)) * (100.0 / (disp + 1.0)) * 1.5 + np.random.normal(0, 3, n)
        
        return pd.DataFrame({
            'Disponibilidade_%': disp,
            'Performance_%': perf,
            'Qualidade_%': qual,
            'Temperatura_Proc_C': temp_proc,
            'OEE_Real_%': oee_real,
            'Lead_Time_Minutos': lead_time
        })

    df_oee = gerar_dados_oee()
    
    features = ['Disponibilidade_%', 'Performance_%', 'Qualidade_%', 'Temperatura_Proc_C']
    X = df_oee[features]
    y = df_oee['OEE_Real_%']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    modelo_oee = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    modelo_oee.fit(X_train, y_train)
    
    with aba_dados:
        st.dataframe(df_oee, use_container_width=True)
        
    with aba_simulador:
        st.markdown("##### 🎛️ Parâmetros Operacionais do Turno:")
        c1, c2 = st.columns(2)
        with c1:
            in_disp = st.slider("Taxa de Disponibilidade (%)", 60.0, 100.0, 88.0, step=0.5, help="Considera paradas de setup SMED e manutenções")
            in_perf = st.slider("Taxa de Desempenho / Takt Time (%)", 60.0, 100.0, 92.0, step=0.5, help="Velocidade real vs nominal da esteira")
        with c2:
            in_qual = st.slider("Taxa de Qualidade (%)", 80.0, 100.0, 98.5, step=0.1, help="Porcentagem de peças aprovadas de primeira")
            in_temp = st.slider("Temperatura Operacional (°C)", 15.0, 90.0, 45.0, step=1.0)
            
        if st.button("Calcular OEE Previsto da Fábrica", type="primary"):
            amostra_oee = pd.DataFrame({
                'Disponibilidade_%': [in_disp],
                'Performance_%': [in_perf],
                'Qualidade_%': [in_qual],
                'Temperatura_Proc_C': [in_temp]
            })
            oee_pred = modelo_oee.predict(amostra_oee)[0]
            
            st.markdown("---")
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("OEE Projetado para o Turno", f"{oee_pred:.1f}%")
            
            if oee_pred >= 85.0:
                col_r2.success("🟢 **CLASSE MUNDIAL (World Class OEE):** Planta em alta produtividade e fluxo enxuto.")
            elif oee_pred >= 65.0:
                col_r2.warning("🟡 **OPERAÇÃO TÍPICA / ATENÇÃO:** Oportunidades de Kaizen em redução de paradas de setup (SMED).")
            else:
                col_r2.error("🔴 **BAIXA EFICIÊNCIA:** Linha com alto índice de ociosidade, micro-paradas ou refugo excessivo.")

    with aba_metricas:
        y_pred = modelo_oee.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        cm1, cm2 = st.columns(2)
        cm1.metric("Acurácia R² em Teste", f"{r2 * 100:.1f}%")
        cm2.metric("Margem Média de Erro (MAE)", f"± {mae:.2f}% de OEE")
        
        fig = px.scatter(
            x=y_test, y=y_pred,
            labels={'x': 'OEE Real de Chão de Fábrica (%)', 'y': 'Previsão da IA (%)'},
            title="Aferição do Modelo de Regressão: OEE Real vs Previsto",
            color_discrete_sequence=['#0055A5']
        )
        fig.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
        st.plotly_chart(fig, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 2: CLASSIFICAÇÃO (QUALIDADE POKA-YOKE)
# ====================================================================
elif menu == "🚦 2. Classificação (Qualidade Poka-Yoke)":
    st.title("🚦 Controle de Qualidade Poka-Yoke & Six Sigma")
    st.caption("Classificação Automática de Peças Usinadas via Árvore de Decisão Calibrada")
    
    aba_dados_q, aba_teste_q, aba_diag_q = st.tabs(["📂 Histórico de Medições", "🚨 Teste Metrológico em Tempo Real", "📊 Diagnóstico da Árvore"])
    
    @st.cache_data
    def gerar_base_qualidade():
        np.random.seed(42)
        n = 350
        desvio_um = np.random.uniform(0.0, 50.0, n)       # Desvio dimensional em micrometros (µm)
        rugosidade = np.random.uniform(0.1, 6.0, n)       # Rugosidade Ra (µm)
        temp_peca = np.random.uniform(18.0, 65.0, n)      # Temperatura da peça pós-usinagem (°C)
        
        status = []
        for d, r, t in zip(desvio_um, rugosidade, temp_peca):
            if d > 28.0 or r > 3.8 or (d > 20.0 and r > 2.8) or t > 55.0:
                status.append(2) # 🔴 Refugo / Sucata
            elif d > 12.0 or r > 1.8:
                status.append(1) # 🟡 Retrabalho
            else:
                status.append(0) # 🟢 Aprovado
                
        return pd.DataFrame({
            'Desvio_Dimensional_um': desvio_um,
            'Rugosidade_Ra_um': rugosidade,
            'Temperatura_Peca_C': temp_peca,
            'Status_Qualidade': status
        })

    df_qual = gerar_base_qualidade()
    X_q = df_qual[['Desvio_Dimensional_um', 'Rugosidade_Ra_um', 'Temperatura_Peca_C']]
    y_q = df_qual['Status_Qualidade']
    
    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_q, y_q, test_size=0.25, random_state=42, stratify=y_q)
    modelo_poka = DecisionTreeClassifier(max_depth=5, min_samples_split=3, class_weight='balanced', random_state=42)
    modelo_poka.fit(X_train_q, y_train_q)
    
    with aba_dados_q:
        df_view_q = df_qual.copy()
        mapa_q = {0: "🟢 Aprovado (Fluxo Contínuo)", 1: "🟡 Retrabalho (Re-usinagem)", 2: "🔴 Refugo (Sucata)"}
        df_view_q['Diagnóstico'] = df_view_q['Status_Qualidade'].map(mapa_q)
        st.dataframe(df_view_q, use_container_width=True)
        
    with aba_teste_q:
        st.markdown("##### 💡 Carregar Amostras Rápidas da Linha:")
        cq1, cq2, cq3 = st.columns(3)
        v_desv, v_rug, v_temp = 5.0, 0.8, 24.0
        
        if cq1.button("🟢 Amostra 1: Peça Perfeita"):
            v_desv, v_rug, v_temp = 4.0, 0.6, 22.0
        if cq2.button("🟡 Amostra 2: Alerta de Rugosidade"):
            v_desv, v_rug, v_temp = 16.0, 2.4, 35.0
        if cq3.button("🔴 Amostra 3: Peça Fora de Tolerância"):
            v_desv, v_rug, v_temp = 38.0, 4.5, 58.0

        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            in_desv = st.slider("Desvio Dimensional Tolerância (µm)", 0.0, 50.0, float(v_desv), step=0.5, help="Tolerância micrométrica da cota crítica")
        with col_q2:
            in_rug = st.slider("Rugosidade Superficial Ra (µm)", 0.1, 6.0, float(v_rug), step=0.1)
        with col_q3:
            in_temp_p = st.slider("Temperatura da Peça pós-corte (°C)", 15.0, 70.0, float(v_temp), step=1.0)
            
        amostra_teste = pd.DataFrame({
            'Desvio_Dimensional_um': [in_desv],
            'Rugosidade_Ra_um': [in_rug],
            'Temperatura_Peca_C': [in_temp_p]
        })
        res_poka = modelo_poka.predict(amostra_teste)[0]
        prob_poka = modelo_poka.predict_proba(amostra_teste)[0]
        
        st.markdown("---")
        col_rq, col_pq = st.columns([1.2, 1])
        
        with col_rq:
            if res_poka == 0:
                st.success("### 🟢 **STATUS: PEÇA APROVADA**\nDimensões dentro do limite Six Sigma. Direto para a linha de montagem.")
            elif res_poka == 1:
                st.warning("### 🟡 **STATUS: RETRABALHO RECOMENDADO**\nCota recuperável. Encaminhar para passe fino de retífica / ajuste de ferramenta.")
            else:
                st.error("### 🔴 **STATUS: REFUGO / SUCATA**\nPeça irrecuperável. Descartar e acionar Andon na célula de usinagem.")
                
        with col_pq:
            st.markdown("##### Probabilidade do Diagnóstico:")
            df_prob_q = pd.DataFrame({
                'Decisão': ['🟢 Aprovado', '🟡 Retrabalho', '🔴 Refugo'],
                'Certeza': prob_poka * 100
            })
            fig_prob_q = px.bar(
                df_prob_q, x='Certeza', y='Decisão', orientation='h',
                text=df_prob_q['Certeza'].apply(lambda x: f"{x:.1f}%"),
                color='Decisão',
                color_discrete_map={'🟢 Aprovado': '#2ca02c', '🟡 Retrabalho': '#ff7f0e', '🔴 Refugo': '#d62728'}
            )
            fig_prob_q.update_layout(xaxis_range=[0, 100], showlegend=False, height=180, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_prob_q, use_container_width=True)

    with aba_diag_q:
        y_pred_q = modelo_poka.predict(X_test_q)
        acc_q = accuracy_score(y_test_q, y_pred_q)
        st.metric("Acurácia Metrológica do Modelo", f"{acc_q * 100:.1f}%")
        
        matriz_q = confusion_matrix(y_test_q, y_pred_q)
        cats_q = ["Aprovado", "Retrabalho", "Refugo"]
        fig_mq = px.imshow(
            matriz_q, x=cats_q, y=cats_q,
            labels=dict(x="Classificação da IA", y="Realidade Metrológica", color="Peças"),
            text_auto=True, color_continuous_scale="Teal"
        )
        st.plotly_chart(fig_mq, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 3: CLUSTERIZAÇÃO (GARGALOS DE LINHA)
# ====================================================================
elif menu == "📊 3. Clusterização (Gargalos de Produção)":
    st.title("📊 Mapeamento de Gargalos de Linha & Balanceamento Lean")
    st.caption("Agrupamento Não Supervisionado de Células de Manufatura via K-Means + StandardScaler")
    
    @st.cache_data
    def gerar_dados_celulas():
        np.random.seed(42)
        n_cel = 60
        setup_min = np.random.uniform(5.0, 90.0, n_cel)          # Tempo de Setup (minutos)
        refugo_pct = np.random.uniform(0.5, 12.0, n_cel)         # % Taxa de Refugo
        paradas_h = np.random.uniform(1.0, 35.0, n_cel)          # Horas de paradas não planejadas / mês
        
        df_c = pd.DataFrame({
            'ID_Celula': [f'Célula {i+1:02d}' for i in range(n_cel)],
            'Tempo_Setup_Min': setup_min,
            'Taxa_Refugo_%': refugo_pct,
            'Paradas_Nao_Planejadas_Horas': paradas_h
        })
        
        scaler_c = StandardScaler()
        X_scaled_c = scaler_c.fit_transform(df_c[['Tempo_Setup_Min', 'Taxa_Refugo_%', 'Paradas_Nao_Planejadas_Horas']])
        
        kmeans_c = KMeans(n_clusters=3, n_init=10, random_state=42)
        df_c['Cluster'] = kmeans_c.fit_predict(X_scaled_c)
        
        mapa_c = {
            0: "🟢 Fluxo Enxuto / Alta Performance",
            1: "🟡 Célula com Desvio Operacional",
            2: "🔴 Gargalo Crítico da Planta (Gargalo TOC)"
        }
        df_c['Diagnóstico_Lean'] = df_c['Cluster'].map(mapa_c)
        return df_c

    df_cel = gerar_dados_celulas()
    
    aba_tbl_c, aba_mapa_c = st.tabs(["📂 Tabela de Indicadores das Células", "🗺️ Mapa 3D de Eficiência Fabril"])
    
    with aba_tbl_c:
        st.dataframe(df_cel[['ID_Celula', 'Tempo_Setup_Min', 'Taxa_Refugo_%', 'Paradas_Nao_Planejadas_Horas', 'Diagnóstico_Lean']], use_container_width=True)
        
    with aba_mapa_c:
        fig_3d_c = px.scatter_3d(
            df_cel,
            x='Tempo_Setup_Min',
            y='Taxa_Refugo_%',
            z='Paradas_Nao_Planejadas_Horas',
            color='Diagnóstico_Lean',
            hover_name='ID_Celula',
            title="Distribuição Multidimensional das Células de Usinagem e Montagem",
            color_discrete_map={
                "🟢 Fluxo Enxuto / Alta Performance": "#2ca02c",
                "🟡 Célula com Desvio Operacional": "#ff7f0e",
                "🔴 Gargalo Crítico da Planta (Gargalo TOC)": "#d62728"
            }
        )
        st.plotly_chart(fig_3d_c, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 4: DEEP LEARNING (MANUTENÇÃO TPM CNC)
# ====================================================================
elif menu == "🧠 4. Deep Learning (Manutenção TPM CNC)":
    st.title("🧠 Manutenção Preditiva TPM em Tornos e Centros CNC")
    st.caption("Rede Neural Perceptron Multicamadas (MLP) com Inferência Reativa em Tempo Real")
    
    @st.cache_resource
    def treinar_rede_preditiva_cnc(epocas=120, lr=0.01, neuronios=32, ativacao='relu'):
        np.random.seed(42)
        n = 450
        vibracao = np.random.uniform(0.5, 12.0, n)       # Vibração RMS do fuso (mm/s)
        temp_mancal = np.random.uniform(25.0, 110.0, n)   # Temperatura do mancal (°C)
        corrente = np.random.uniform(5.0, 45.0, n)        # Corrente do servo-motor (A)
        pressao_hidr = np.random.uniform(20.0, 180.0, n)  # Pressão óleo hidráulico (bar)
        
        # Equação de degradação mecânica não-linear
        prob_falha = (
            ((vibracao / 4.0) ** 2.2) * 5.0 + 
            ((temp_mancal / 60.0) ** 2.5) * 4.0 + 
            (corrente * 0.4) + 
            (pressao_hidr * 0.1) + 
            np.random.normal(0, 2.0, n)
        )
        prob_falha = np.clip(prob_falha, 0.0, 100.0)
        
        X_df = pd.DataFrame({
            'Vibracao_RMS_mm_s': vibracao,
            'Temp_Mancal_C': temp_mancal,
            'Corrente_Motor_A': corrente,
            'Pressao_Hidraulica_bar': pressao_hidr
        })
        y_ser = prob_falha
        
        sc_cnc = StandardScaler()
        X_sc_cnc = sc_cnc.fit_transform(X_df)
        
        mlp_cnc = MLPRegressor(
            hidden_layer_sizes=(neuronios, neuronios // 2),
            activation=ativacao,
            solver='adam',
            learning_rate_init=lr,
            max_iter=epocas,
            random_state=42
        )
        mlp_cnc.fit(X_sc_cnc, y_ser)
        return mlp_cnc, sc_cnc, mlp_cnc.loss_curve_

    modelo_cnc, scaler_cnc, loss_cnc = treinar_rede_preditiva_cnc()
    
    aba_sim_cnc, aba_treino_cnc = st.tabs(["🧪 Telemetria & Simulação em Tempo Real", "⚙️ Arquitetura da Rede Neural"])
    
    with aba_sim_cnc:
        st.markdown("### 🎛️ Sensores de Telemetria do Eixo-Árvore (Spindle CNC):")
        c1, c2 = st.columns(2)
        with c1:
            in_vib = st.slider("Vibração RMS no Fuso (mm/s)", 0.5, 12.0, 2.5, step=0.1, help="Norma ISO 10816 considera > 4.5 mm/s em alerta")
            in_temp_m = st.slider("Temperatura do Mancal / Rolamento (°C)", 25.0, 110.0, 50.0, step=1.0)
        with c2:
            in_corr = st.slider("Corrente do Servo-Motor (A)", 5.0, 45.0, 15.0, step=0.5)
            in_press = st.slider("Pressão do Sistema Hidráulico (bar)", 20.0, 180.0, 80.0, step=5.0)
            
        dados_cnc_in = pd.DataFrame({
            'Vibracao_RMS_mm_s': [in_vib],
            'Temp_Mancal_C': [in_temp_m],
            'Corrente_Motor_A': [in_corr],
            'Pressao_Hidraulica_bar': [in_press]
        })
        dados_cnc_norm = scaler_cnc.transform(dados_cnc_in)
        falha_pred = modelo_cnc.predict(dados_cnc_norm)[0]
        falha_pred = max(0.0, min(100.0, float(falha_pred)))
        
        st.markdown("---")
        col_m1, col_m2 = st.columns([1, 1.5])
        with col_m1:
            st.metric(
                label="Probabilidade de Quebra / Falha Crítica",
                value=f"{falha_pred:.1f}%",
                delta=f"{'+' if falha_pred > 40 else '-'}{abs(falha_pred - 40):.1f}% do limiar",
                delta_color="inverse"
            )
            
        with col_m2:
            if falha_pred < 35.0:
                st.success("🟢 **MÁQUINA EM OPERAÇÃO SEGURA (TPM ZERO QUEBRA)**\nVibração e temperatura dentro das tolerâncias dinâmicas.")
            elif falha_pred < 65.0:
                st.warning("🟡 **ALERTA DE DESGASTE: AGENDAR LUBRIFICAÇÃO / BALANCEAMENTO**\nSinais de folga ou sobreaquecimento incipiente.")
            else:
                st.error("🔴 **PARADA DE EMERGÊNCIA RECOMENDADA (RISCO DE QUEBRA DE FUSO)**\nAcionar equipe de manutenção mecânica imediatamente!")
                
        st.progress(int(falha_pred))

    with aba_treino_cnc:
        st.markdown("### 🧠 Otimização dos Pesos Sinápticos da Rede")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            ep_cnc = st.slider("Épocas de Treinamento", 30, 300, 120, step=10)
            lr_cnc = st.select_slider("Taxa de Aprendizado", [0.001, 0.005, 0.01, 0.05], value=0.01)
        with col_t2:
            neur_cnc = st.slider("Neurônios na Camada Oculta", 8, 64, 32, step=8)
            act_cnc = st.selectbox("Função de Ativação", ["relu", "tanh", "logistic"])
            
        if st.button("Re-treinar Rede Neural TPM", type="primary"):
            st.cache_resource.clear()
            with st.spinner("Atualizando pesos da rede com novos hiperparâmetros..."):
                modelo_cnc, scaler_cnc, loss_cnc = treinar_rede_preditiva_cnc(ep_cnc, lr_cnc, neur_cnc, act_cnc)
            st.success("Rede Neural atualizada!")
            
        df_loss_cnc = pd.DataFrame({'Época': range(1, len(loss_cnc) + 1), 'Loss (MSE)': loss_cnc})
        fig_l_cnc = px.line(df_loss_cnc, x='Época', y='Loss (MSE)', title="Decaimento do Erro no Diagnóstico de Vibração")
        st.plotly_chart(fig_l_cnc, use_container_width=True)

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 5: PLN & 5S (AUDITORIA & KAIZEN)
# ====================================================================
elif menu == "📝 5. PLN & 5S (Auditoria & Kaizen)":
    st.title("📝 Processamento de Linguagem Natural: Gestão Visual & 5S")
    st.caption("Mineração Textual e Análise Semântica de Apontamentos de Chão de Fábrica e Kaizens")
    
    TERMOS_KAIZEN_5S = [
        'kaizen', '5s', 'melhoria', 'organizado', 'limpo', 'padronizado', 'sucesso', 'eficiente',
        'otimizado', 'smeg', 'poka-yoke', 'seguro', 'resolvido', 'conformidade', 'aprovado', 
        'treinado', 'manutenção concluída', 'redução de tempo', 'kanban'
    ]
    
    TERMOS_PARADA_DEFEITO = [
        'parada', 'quebra', 'defeito', 'refugo', 'rebarba', 'trinca', 'vazamento', 'travamento',
        'superaquecimento', 'falha', 'atraso', 'gargalo', 'falta de peça', 'sucata', 'sem epi',
        'acidente', 'desalinhado', 'alarme', 'erro', 'ruído estranho', 'emergência'
    ]
    
    def analisar_texto_fabril(texto):
        txt_l = texto.lower().replace(',', ' ').replace('.', ' ').split()
        pos_c = sum(1 for p in txt_l if any(tk in p for tk in TERMOS_KAIZEN_5S))
        neg_c = sum(1 for p in txt_l if any(tp in p for tp in TERMOS_PARADA_DEFEITO))
        tot = len(txt_l) if len(txt_l) > 0 else 1
        score = (pos_c - neg_c) / max(1, (pos_c + neg_c))
        return {
            'pos': pos_c, 'neg': neg_c, 'score': score,
            'pos_termos': [p for p in txt_l if any(tk in p for tk in TERMOS_KAIZEN_5S)],
            'neg_termos': [p for p in txt_l if any(tp in p for tp in TERMOS_PARADA_DEFEITO)]
        }

    st.markdown("##### 💡 Exemplos de Apontamentos de Turno:")
    c_b1, c_b2, c_b3 = st.columns(3)
    
    txt_fabrica = "Implementado evento Kaizen na bancada 4. Ferramentas organizadas com sombra conforme 5S e tempo de setup reduzido com sucesso."
    if c_b1.button("🟢 Exemplo: Kaizen / 5S (Melhoria)"):
        txt_fabrica = "Kaizen concluído na prensa hidráulica. Área limpa, padronizada e dispositivos Poka-Yoke instalados com sucesso."
    if c_b2.button("🔴 Exemplo: Alarme / Quebra de Linha"):
        txt_fabrica = "Linha de usinagem parada por quebra de ferramenta, vazamento de óleo de corte e alarme de sobreaquecimento no cabeçote."
    if c_b3.button("🟡 Exemplo: Apontamento Neutro"):
        txt_fabrica = "Início do terceiro turno com troca de operadores e recebimento de 200 tarugos de aço 1045."

    relato_in = st.text_area("Apontamento do Operador / Inspetor no Terminal da Linha:", value=txt_fabrica, height=110)
    
    if st.button("Classificar Relato via PLN", type="primary"):
        res_pln = analisar_texto_fabril(relato_in)
        
        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Termos Kaizen / 5S", f"{res_pln['pos']}")
        cp2.metric("Termos de Parada / Defeito", f"{res_pln['neg']}")
        cp3.metric("Polaridade da Linha", f"{res_pln['score']:+.2f}")
        
        st.markdown("---")
        if res_pln['score'] > 0.15:
            st.success(f"🟢 **KAIZEN & CONFORMIDADE OPERACIONAL DETECTADOS**\n\nPalavras-chave de melhoria: `{', '.join(set(res_pln['pos_termos']))}`")
        elif res_pln['score'] < -0.15:
            st.error(f"🔴 **NOTIFICAÇÃO DE ALERTA DE CHÃO DE FÁBRICA**\n\nTermos de parada/falha: `{', '.join(set(res_pln['neg_termos']))}`. Acionar Líder de Produção e Manutenção!")
        else:
            st.warning("🟡 **APONTAMENTO OPERACIONAL DE ROTINA / INFORMATIVO**")

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 6: VISÃO COMPUTACIONAL (DEFEITOS EM PEÇAS)
# ====================================================================
elif menu == "👁️ 6. Visão Computacional (Defeitos em Peças)":
    st.title("👁️ Visão Computacional: Inspeção Óptica em Linha de Usinagem")
    st.caption("Processamento Digital de Imagens (Filtros de Gradiente e Detecção de Rebarbas/Riscos)")
    
    if cv2 is None:
        st.error("⚠️ OpenCV não encontrado. Instale com `pip install opencv-python-headless`.")
    else:
        upload_peca = st.file_uploader("Envie a foto de uma peça usinada, engrenagem ou chapa estampada (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        if upload_peca is not None:
            img_p_pil = Image.open(upload_peca)
            img_p_np = np.array(img_p_pil)
            
            c_im1, c_im2 = st.columns(2)
            with c_im1:
                st.image(img_p_pil, caption="Peça Capturada na Câmera da Esteira", use_container_width=True)
            with c_im2:
                cinza = cv2.cvtColor(img_p_np, cv2.COLOR_RGB2GRAY)
                suav = cv2.GaussianBlur(cinza, (5, 5), 0)
                bordas_peca = cv2.Canny(suav, 60, 160)
                st.image(bordas_peca, caption="Segmentação de Contornos, Rebarbas e Defeitos Superficiais (Canny)", use_container_width=True)
                st.info("💡 Este mapa de bordas permite a modelos de Deep Learning (como YOLO) medir rebarbas em tempo real.")

    exibir_rodape_educacional()

# ====================================================================
# MÓDULO 7: IA GENERATIVA (ASSISTENTE POP & NR-12)
# ====================================================================
elif menu == "💬 7. IA Generativa (Assistente POP & NR-12)":
    st.title("💬 Assistente Virtual Especialista em Chão de Fábrica & POP")
    st.caption("Consulta Automatizada a Procedimentos Operacionais Padrão e Manuais de Máquinas Industriais")
    
    with st.expander("⚙️ Parâmetros de Integração Azure OpenAI"):
        c1, c2 = st.columns(2)
        with c1:
            api_k = st.text_input("Chave da API", type="password")
        with c2:
            endpoint_k = st.text_input("Endpoint", value="https://marcelomaffeis-05082026-resource.services.ai.azure.com/openai/v1")
        dep_model = st.text_input("Nome do Deployment", value="gpt-4.1-mini")

    manual_fabril_padrao = """
    MANUAL DE OPERAÇÃO E SEGURANÇA INDUSTRIAL (NR-12 / POP-042):
    1. ALARME 1024 (Sobrecarga no Servo-Motor do Eixo Z):
       - Causa: Bloqueio mecânico por acúmulo de cavacos ou falha no sistema de lubrificação centralizada.
       - Ação do Operador: Desligar avanço rápido, verificar guias lineares, acionar rearme térmico no painel 2.
    2. PROCEDIMENTO DE SETUP RÁPIDO (SMED):
       - Troca de castanhas e ferramentas deve ser feita com trava mecânica acionada e botão de emergência bloqueado (Lockout/Tagout).
    3. CONTROLE DE TEMPERATURA:
       - Óleo de corte sintético deve operar entre 20°C e 45°C. Se passar de 60°C, pausar usinagem para evitar deformação térmica da peça.
    """

    if "mensagens_lean" not in st.session_state:
        st.session_state.mensagens_lean = [
            {
                "role": "system",
                "content": f"""Você é o Engenheiro Especialista em Manufatura Enxuta e Segurança NR-12 do Chão de Fábrica.
                Responda dúvidas de operadores, técnicos de processos e manutenção com base estrita no procedimento padrão:
                
                --- PROCEDIMENTO OPERACIONAL PADRÃO (POP) ---
                {manual_fabril_padrao}
                """
            }
        ]

    for msg in st.session_state.mensagens_lean:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    duvida_fabril = st.chat_input("Digite sua dúvida (Ex: O torno CNC deu alarme 1024 no eixo Z, o que devo fazer?)")
    if duvida_fabril:
        if not api_k:
            st.warning("⚠️ Insira a chave da API nas configurações acima para liberar o chat.")
        else:
            st.session_state.mensagens_lean.append({"role": "user", "content": duvida_fabril})
            with st.chat_message("user"):
                st.markdown(duvida_fabril)
                
            with st.chat_message("assistant"):
                with st.spinner("Consultando POP e Manuais Técnicos de Fábrica..."):
                    try:
                        client_lean = OpenAI(base_url=endpoint_k, api_key=api_k)
                        resp_lean = client_lean.chat.completions.create(
                            model=dep_model,
                            messages=st.session_state.mensagens_lean,
                            temperature=0.2
                        )
                        txt_resposta = resp_lean.choices[0].message.content
                        st.markdown(txt_resposta)
                        st.session_state.mensagens_lean.append({"role": "assistant", "content": txt_resposta})
                    except Exception as err:
                        st.error(f"Erro na comunicação com a IA: {err}")

    exibir_rodape_educacional()