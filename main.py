import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata

# Configuração da página do Streamlit
st.set_page_config(
    page_title="StructStat - Princípio de Saint-Venant",
    page_icon="🏗️",
    layout="wide"
)

def plot_chapa(L, H, raio, tensao):
    """Gera o esquema visual do problema de contorno."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    # Desenho da chapa
    chapa = patches.Rectangle((0, 0), L, H, lw=2, ec='black', fc='#f0f0f0', zorder=1)
    ax.add_patch(chapa)
    
    # Desenho do Furo
    furo = patches.Circle((L/2, H/2), raio, fc='white', ec='black', lw=1.5, zorder=2)
    ax.add_patch(furo)
    
    # Hachuras do engaste
    num_hachuras = 15
    for i in np.linspace(0, H, num_hachuras):
        ax.plot([-L*0.04, 0], [i, i-H*0.04], color='black', lw=1)
    ax.plot([0, 0], [0, H], color='black', lw=3)
    ax.text(-L*0.06, H/2, 'Engaste Fixo\n(u=0, v=0)', ha='right', va='center', fontweight='bold')
    
    # Setas de Tração
    y_setas = np.linspace(H*0.1, H*0.9, 7)
    for y_p in y_setas:
        ax.annotate('', xy=(L + L*0.1, y_p), xytext=(L, y_p),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(L + L*0.02, H + H*0.02, f'Tração Axial $\sigma = {tensao}$ MPa', color='red', fontweight='bold')
    
    # Linhas de simetria
    ax.axhline(H/2, color='gray', ls='-.', lw=0.8, zorder=0)
    ax.axvline(L/2, color='gray', ls='-.', lw=0.8, zorder=0)
    
    # Cotas de Dimensões
    ax.annotate('', xy=(0, -H*0.08), xytext=(L, -H*0.08), arrowprops=dict(arrowstyle='<->'))
    ax.text(L/2, -H*0.15, f'L = {L}mm', ha='center')
    ax.annotate('', xy=(L*0.1, 0), xytext=(L*0.1, H), arrowprops=dict(arrowstyle='<->'))
    ax.text(L*0.06, H/2, f'H = {H}mm', rotation=90, va='center')

    ax.set_title('Problema Simulado (Condições de Contorno)', fontsize=14, pad=20)
    ax.set_xlim(-L*0.3, L*1.3)
    ax.set_ylim(-H*0.2, H*1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

@st.cache_data
def calcular_campo_tensoes(L, H, raio_furo, tensao_axial, nx=600, ny=300):
    """Calcula o campo de tensões usando as equações de Kirsch."""
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)

    Xc, Yc = L/2, H/2
    R = np.sqrt((X - Xc)**2 + (Y - Yc)**2)
    Theta = np.arctan2(Y - Yc, X - Xc)

    # Equações de Kirsch (Sigma_xx)
    a = raio_furo
    term1 = 1 - (a**2 / R**2) * (1.5 * np.cos(2*Theta) + np.cos(4*Theta))
    term2 = 1.5 * (a**4 / R**4) * np.cos(4*Theta)
    s_x = tensao_axial * (term1 + term2)

    # Remove tensões dentro do furo
    s_x[R < a] = np.nan
    
    return X, Y, s_x, Xc, Yc

def plot_simulacao_saint_venant(X, Y, s_x, Xc, Yc, L, H, raio_furo, tensao_axial, comparativo, x_coords):
    """Renderiza os gráficos de contorno e perfis de tensão."""
    vmin_plot = -100.0 if tensao_axial > 0 else tensao_axial * 3.5
    vmax_plot = tensao_axial * 3.5 if tensao_axial > 0 else -100.0
    
    # Ajuste para evitar bugs visuais se tensão for 0
    if tensao_axial == 0:
        vmin_plot, vmax_plot = -10, 10
        
    n_sec = len(x_coords)
    if n_sec == 0:
        x_coords = [Xc]
        n_sec = 1
        
    if comparativo:
        fig = plt.figure(figsize=(12, 11))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
        axes_sections = [fig.add_subplot(gs[1, 0])]
    else:
        # Aumenta a largura da imagem dinamicamente para caber todas as seções
        fig = plt.figure(figsize=(max(12, 4.5 * n_sec), 11))
        gs = fig.add_gridspec(2, n_sec, height_ratios=[1.3, 1])
        axes_sections = [fig.add_subplot(gs[1, i]) for i in range(n_sec)]

    # Gráfico de Contorno (Heatmap)
    ax_map = fig.add_subplot(gs[0, :])
    cp = ax_map.contourf(X, Y, s_x, levels=120, cmap='jet', vmin=vmin_plot, vmax=vmax_plot)
    cbar = fig.colorbar(cp, ax=ax_map)
    cbar.set_label('Tensão Longitudinal $\sigma_{xx}$ (MPa)', fontsize=13)
    
    furo = plt.Circle((Xc, Yc), raio_furo, color='white', zorder=12, ec='black', lw=1.2)
    ax_map.add_patch(furo)
    
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_sec)))
    for idx, x_val in enumerate(x_coords):
        cor = cmap_colors[idx % 10]
        ax_map.axvline(x_val, color=cor, linestyle='--', linewidth=2, alpha=0.9)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=cor, lw=1.5, alpha=0.9)
        ax_map.text(x_val + L*0.005, H*0.05, f'x={x_val:g}', color='black', fontsize=10, fontweight='bold', bbox=bbox_props)

    ax_map.set_title('Campo de Tensões $\sigma_{xx}$ (Equações de Kirsch)', fontsize=15, fontweight='bold')
    ax_map.set_xlabel('Comprimento (mm)', fontsize=12)
    ax_map.set_ylabel('Altura (mm)', fontsize=12)
    ax_map.set_aspect('equal')
    ax_map.tick_params(labelsize=11)

    y_fine = np.linspace(0, H, 800)
    
    for i, (sx, color) in enumerate(zip(x_coords, cmap_colors)):
        # Evita extrair fatias fora da chapa
        if sx > L or sx < 0:
            continue
            
        lab = f'Seção x = {sx:g} mm'
        
        stress_slice = griddata((X.flatten(), Y.flatten()), s_x.flatten(), 
                                (np.full_like(y_fine, sx), y_fine), method='linear')
        
        target_ax = axes_sections[0] if comparativo else axes_sections[i]
        target_ax.plot(stress_slice, y_fine, color=color, lw=3, label=lab)
        
        if not comparativo:
            target_ax.set_title(lab, fontsize=13, color=color, fontweight='bold')
            target_ax.set_xlim(vmin_plot * 1.1, vmax_plot * 1.1)
            target_ax.axvline(0, color='black', lw=1, ls='-')
        
        target_ax.axvline(tensao_axial, color='gray', ls='--', alpha=0.7, lw=1.5, label='Tensão Nominal ($S$)' if i==0 else "")
        target_ax.grid(True, linestyle=':', alpha=0.6)
        target_ax.set_xlabel('$\sigma_{xx}$ (MPa)', fontsize=12)
        target_ax.tick_params(labelsize=11)

    axes_sections[0].set_ylabel('Altura da Chapa $y$ (mm)', fontsize=12)
    if comparativo: 
        axes_sections[0].legend(fontsize=11, loc='best')

    plt.tight_layout()
    return fig

def main():
    st.title("🏗️ StructStat: Princípio de Saint-Venant e Concentração de Tensões")
    st.markdown("""
    Este módulo simula a distribuição de tensões longitudinais ($\sigma_{xx}$) em uma chapa tracionada contendo um furo circular central. 
    As equações fechadas de Kirsch são utilizadas para demonstrar o **Princípio de Saint-Venant**, que afirma que a perturbação no campo de tensões causada por uma descontinuidade se dissipa rapidamente à medida que nos afastamos do defeito.
    """)
    
    # Barra lateral para inputs
    with st.sidebar:
        st.header("Parâmetros do Modelo")
        
        L = st.number_input("Comprimento da Chapa (L) [mm]:", min_value=50.0, value=200.0, step=10.0)
        H = st.number_input("Altura da Chapa (H) [mm]:", min_value=20.0, value=100.0, step=10.0)
        diametro = st.number_input("Diâmetro do Furo (D) [mm]:", min_value=2.0, value=20.0, step=2.0)
        tensao = st.number_input("Tensão de Tração (MPa):", value=100.0, step=10.0)
        
        st.divider()
        
        st.header("Análise de Seções de Corte")
        st.write("Indique as posições onde as fatias de tensão serão extraídas.")
        
        val_padrao = f"{L/2}, {L/2 + diametro}, {L - 10}"
        secoes_str = st.text_input("Cortes em X (separados por vírgula) [mm]:", value=val_padrao)
        
        # Parsing seguro das coordenadas X
        try:
            x_coords_raw = [float(x.strip()) for x in secoes_str.split(',')]
            x_coords = sorted([x for x in x_coords_raw if 0 <= x <= L])
            if not x_coords:
                st.warning("Nenhuma seção válida dentro da chapa informada.")
                x_coords = [L/2]
        except Exception:
            st.error("Formato inválido. Use vírgulas. Ex: 100, 120, 150")
            x_coords = [L/2]
            
        st.divider()
        st.header("Opções Visuais")
        graficos_juntos = st.toggle("Sobrepor Gráficos de Seção", value=False, 
                                   help="Ative para plotar todas as curvas de tensão no mesmo eixo.")
        
        # Validação geométrica
        raio = diametro / 2.0
        geometria_valida = True
        if diametro >= H:
            st.error("Erro: O diâmetro do furo deve ser menor que a altura da chapa.")
            geometria_valida = False
        if diametro >= L:
            st.error("Erro: O diâmetro do furo deve ser menor que o comprimento da chapa.")
            geometria_valida = False

    if geometria_valida:
        col_esq, col_dir = st.columns([1, 2])
        
        with col_esq:
            st.subheader("Configuração Físico-Mecânica")
            fig_esquema = plot_chapa(L, H, raio, tensao)
            st.pyplot(fig_esquema)
            
            with st.expander("📚 Fundamentação Teórica", expanded=False):
                st.markdown("""
                **Fator de Concentração de Tensões Teórico ($K_t$)**
                Para uma placa de largura infinita tracionada axialmente com um furo circular, o valor máximo de tensão ocorre nas bordas do furo e vale exatamente $3 \times S$ (Tensão Nominal). 
                
                Neste cenário, se a tensão aplicada na borda for $100 \text{ MPa}$, a tensão limite no furo será de **$300 \text{ MPa}$**.
                """)
        
        with col_dir:
            st.subheader("Resultados Analíticos (Equações de Kirsch)")
            with st.spinner("Computando malha e interpolando campo de tensões..."):
                # O processamento matemático é protegido por cache do streamlit
                X, Y, s_x, Xc, Yc = calcular_campo_tensoes(L, H, raio, tensao)
                
                # Renderiza figura
                fig_mef = plot_simulacao_saint_venant(X, Y, s_x, Xc, Yc, L, H, raio, tensao, graficos_juntos, x_coords)
                st.pyplot(fig_mef)

if __name__ == "__main__":
    main()
