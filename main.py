import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata

def aplicar_tema(fig, axes, fg_color, bg_color='white', is_mapa=False, usar_transparencia=True):
    """Aplica estética Tufte e gerencia o fundo transparente ou sólido."""
    if usar_transparencia:
        fig.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor(bg_color)
        fig.patch.set_alpha(1.0)
    
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
        
    for ax in axes:
        if not is_mapa:
            if usar_transparencia:
                ax.patch.set_alpha(0.0)
            else:
                ax.set_facecolor(bg_color)
                ax.patch.set_alpha(1.0)
            
        ax.tick_params(axis='both', colors=fg_color, width=0.8, length=4, labelsize=10)
        ax.xaxis.label.set_color(fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.title.set_color(fg_color)
        ax.title.set_fontsize(13)
        
        # Maximizando Data-Ink Ratio (Tufte)
        if not is_mapa:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(fg_color)
            ax.spines['bottom'].set_color(fg_color)
            ax.spines['left'].set_alpha(0.3)
            ax.spines['bottom'].set_alpha(0.3)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor(fg_color)
                spine.set_alpha(0.3)

def plot_chapa(L, H, raio, tensao, fg_color='black', chapa_color='#F0F0F0', bg_color='white', usar_transparencia=True):
    """Gera o esquema visual do problema de contorno."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    
    # Desenho da chapa com cor adaptativa
    chapa = patches.Rectangle((0, 0), L, H, lw=1.5, ec=fg_color, fc=chapa_color, zorder=1, alpha=0.8)
    ax.add_patch(chapa)
    
    # Desenho do Furo (branco no modo claro, transparente no modo escuro)
    cor_furo = 'none' if usar_transparencia else bg_color
    furo = patches.Circle((L/2, H/2), raio, fc=cor_furo, ec=fg_color, lw=1.5, zorder=2)
    ax.add_patch(furo)
    
    # Hachuras do engaste sutis
    num_hachuras = 15
    for i in np.linspace(0, H, num_hachuras):
        ax.plot([-L*0.04, 0], [i, i-H*0.04], color=fg_color, lw=0.8, alpha=0.6)
    ax.plot([0, 0], [0, H], color=fg_color, lw=2.5, alpha=0.8)
    ax.text(-L*0.06, H/2, 'Engaste Fixo\n(u=0, v=0)', ha='right', va='center', fontweight='bold', color=fg_color, fontsize=10)
    
    # Setas de Tração (mais elegantes)
    y_setas = np.linspace(H*0.1, H*0.9, 7)
    for y_p in y_setas:
        ax.annotate('', xy=(L + L*0.12, y_p), xytext=(L, y_p),
                    arrowprops=dict(arrowstyle='->,head_width=0.15,head_length=0.3', color='#FF4B4B', lw=1.2))
    ax.text(L + L*0.04, H + H*0.02, f'Tração Axial\n$\sigma = {tensao}$ MPa', color='#FF4B4B', fontweight='bold', fontsize=10)
    
    # Linhas de simetria com alpha
    ax.axhline(H/2, color=fg_color, ls='-.', lw=0.7, zorder=0, alpha=0.4)
    ax.axvline(L/2, color=fg_color, ls='-.', lw=0.7, zorder=0, alpha=0.4)
    
    # Cotas de Dimensões
    ax.annotate('', xy=(0, -H*0.1), xytext=(L, -H*0.1), arrowprops=dict(arrowstyle='<->', color=fg_color, alpha=0.6))
    ax.text(L/2, -H*0.18, f'L = {L}mm', ha='center', color=fg_color, fontsize=10)
    ax.annotate('', xy=(L*0.1, 0), xytext=(L*0.1, H), arrowprops=dict(arrowstyle='<->', color=fg_color, alpha=0.6))
    ax.text(L*0.06, H/2, f'H = {H}mm', rotation=90, va='center', color=fg_color, fontsize=10)

    ax.set_title('Problema Simulado (Condições de Contorno)', pad=25, fontweight='600')
    ax.set_xlim(-L*0.35, L*1.35)
    ax.set_ylim(-H*0.3, H*1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    aplicar_tema(fig, ax, fg_color, bg_color, usar_transparencia=usar_transparencia)
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

def plot_mapa_calor(X, Y, s_x, Xc, Yc, L, H, raio_furo, tensao_axial, x_coords, fg_color, bg_color='white', usar_transparencia=True):
    """Renderiza isoladamente o mapa de calor das tensões (Heatmap)."""
    vmin_plot = -100.0 if tensao_axial > 0 else tensao_axial * 3.5
    vmax_plot = tensao_axial * 3.5 if tensao_axial > 0 else -100.0
    if tensao_axial == 0: vmin_plot, vmax_plot = -10, 10
        
    fig, ax_map = plt.subplots(figsize=(9.2, 4.5), dpi=120)
    
    # Substituição do 'jet' por 'turbo' (perceptualmente uniforme)
    cp = ax_map.contourf(X, Y, s_x, levels=120, cmap='turbo', vmin=vmin_plot, vmax=vmax_plot)
    cbar = fig.colorbar(cp, ax=ax_map, pad=0.04)
    cbar.set_label('Tensão Longitudinal $\sigma_{xx}$ (MPa)', fontsize=11, color=fg_color, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color, labelsize=9)
    cbar.outline.set_edgecolor(fg_color)
    cbar.outline.set_alpha(0.3)
    
    cor_furo = 'none' if usar_transparencia else bg_color
    furo = plt.Circle((Xc, Yc), raio_furo, color=cor_furo, zorder=12, ec=fg_color, lw=1.5)
    ax_map.add_patch(furo)
    
    n_sec = max(1, len(x_coords))
    cmap_colors = plt.cm.Dark2(np.linspace(0, 1, max(8, n_sec))) # Paleta profissional e acessível
    for idx, x_val in enumerate(x_coords):
        cor = cmap_colors[idx % 8]
        ax_map.axvline(x_val, color=cor, linestyle='--', linewidth=1.8, alpha=0.85)
        # Bbox sutil e transparente
        bbox_props = dict(boxstyle="round,pad=0.3", fc=bg_color, ec=cor, lw=1.2, alpha=0.8)
        ax_map.text(x_val + L*0.015, H*0.05, f'x={x_val:g}', color=fg_color, fontsize=9, fontweight='600', bbox=bbox_props)

    ax_map.set_title('Campo de Tensões $\sigma_{xx}$ (Equações de Kirsch)', fontweight='600', pad=15)
    ax_map.set_xlabel('Comprimento (mm)', fontsize=11, labelpad=10)
    ax_map.set_ylabel('Altura (mm)', fontsize=11, labelpad=10)
    ax_map.set_aspect('equal')
    
    aplicar_tema(fig, ax_map, fg_color, bg_color, is_mapa=True, usar_transparencia=usar_transparencia)
    plt.tight_layout()
    return fig

def plot_graficos_secao(X, Y, s_x, Xc, L, H, tensao_axial, comparativo, x_coords, fg_color, bg_color='white', usar_transparencia=True):
    """Renderiza a linha inferior contendo os cortes transversais da tensão."""
    vmin_plot = -100.0 if tensao_axial > 0 else tensao_axial * 3.5
    vmax_plot = tensao_axial * 3.5 if tensao_axial > 0 else -100.0
    if tensao_axial == 0: vmin_plot, vmax_plot = -10, 10
        
    n_sec = len(x_coords)
    if n_sec == 0:
        x_coords = [Xc]
        n_sec = 1
        
    if comparativo:
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
        axes_sections = [ax]
    else:
        fig, axes_sections = plt.subplots(1, n_sec, figsize=(max(12, 3.5 * n_sec), 4.5), dpi=120)
        if n_sec == 1:
            axes_sections = [axes_sections]

    y_fine = np.linspace(0, H, 800)
    cmap_colors = plt.cm.Dark2(np.linspace(0, 1, max(8, n_sec)))
    
    for i, (sx, color) in enumerate(zip(x_coords, cmap_colors)):
        if sx > L or sx < 0: continue
            
        lab = f'Seção x = {sx:g} mm'
        stress_slice = griddata((X.flatten(), Y.flatten()), s_x.flatten(), 
                                (np.full_like(y_fine, sx), y_fine), method='linear')
        
        target_ax = axes_sections[0] if comparativo else axes_sections[i]
        
        # Sombra sutil sob a linha para destaque profissional (Z-order trick)
        target_ax.plot(stress_slice, y_fine, color='black', lw=3.5, alpha=0.15, zorder=2)
        target_ax.plot(stress_slice, y_fine, color=color, lw=2.5, label=lab, zorder=3)
        
        if not comparativo:
            target_ax.set_title(lab, fontsize=11, color=color, fontweight='600')
            target_ax.set_xlim(vmin_plot * 1.1, vmax_plot * 1.1)
            target_ax.axvline(0, color=fg_color, lw=0.8, ls='-', alpha=0.5)
        
        target_ax.axvline(tensao_axial, color='#FF4B4B', ls=':', alpha=0.7, lw=1.5, label='Tensão Nominal' if i==0 else "")
        target_ax.grid(True, linestyle='--', alpha=0.15, color=fg_color)
        target_ax.set_xlabel('$\sigma_{xx}$ (MPa)', fontsize=10)

    axes_sections[0].set_ylabel('Altura $y$ (mm)', fontsize=10)
    if comparativo: 
        legend = axes_sections[0].legend(fontsize=10, loc='best', frameon=False)
        plt.setp(legend.get_texts(), color=fg_color)

    aplicar_tema(fig, axes_sections, fg_color, bg_color, usar_transparencia=usar_transparencia)
    fig.subplots_adjust(wspace=0.3)
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
        st.header("Configurações do Gráfico")
        tema_selecionado = st.selectbox("Tema Visual:", ["Modo Escuro (Dashboard)", "Modo Claro (Artigos/Word)"])
        
        # Controle de transparência e cores dependendo do tema escolhido
        if "Claro" in tema_selecionado:
            fg_color, chapa_color, bg_color = "#333333", "#F5F5F5", "white"
            usar_transparencia = False
        else:
            fg_color, chapa_color, bg_color = "#FAFAFA", "#1E1E1E", "#0E1117"
            usar_transparencia = True

        st.divider()
        
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
            
        # Adição do Footer solicitado na barra lateral
        st.markdown(
            """
            <br><br><br>
            <div style='text-align: center; color: gray; font-size: 0.85em;'>
                <b>Princípio de Saint-Venant</b><br>
                Desenvolvido por: Pedro Jardim<br>
                <i>v1.0 - Fevereiro/2026</i>
            </div>
            """, 
            unsafe_allow_html=True
        )

    if geometria_valida:
        # PRIMEIRA LINHA: Esquema Físico (Esquerda) e Mapa de Calor (Direita)
        col_esq, col_dir = st.columns([8, 9.2])
        
        with col_esq:
            st.subheader("Configuração Físico-Mecânica")
            fig_esquema = plot_chapa(L, H, raio, tensao, fg_color, chapa_color, bg_color, usar_transparencia)
            st.pyplot(fig_esquema, transparent=usar_transparencia)
            
            with st.expander("📚 Fundamentação Teórica", expanded=False):
                st.markdown("""
                **Fator de Concentração de Tensões Teórico ($K_t$)**
                Para uma placa de largura infinita tracionada axialmente com um furo circular, o valor máximo de tensão ocorre nas bordas do furo e vale exatamente $3 \times S$ (Tensão Nominal). 
                
                Neste cenário, se a tensão aplicada na borda for $100 \text{ MPa}$, a tensão limite no furo será de **$300 \text{ MPa}$**.
                """)
        
        with col_dir:
            st.subheader("Campo de Tensões (Equações de Kirsch)")
            with st.spinner("Computando malha..."):
                X, Y, s_x, Xc, Yc = calcular_campo_tensoes(L, H, raio, tensao)
                fig_mapa = plot_mapa_calor(X, Y, s_x, Xc, Yc, L, H, raio, tensao, x_coords, fg_color, bg_color, usar_transparencia)
                st.pyplot(fig_mapa, transparent=usar_transparencia)

        # SEGUNDA LINHA: Gráficos de Seção Isolados
        st.divider()
        st.subheader("Cortes Transversais do Perfil de Tensão")
        fig_secoes = plot_graficos_secao(X, Y, s_x, Xc, L, H, tensao, graficos_juntos, x_coords, fg_color, bg_color, usar_transparencia)
        st.pyplot(fig_secoes, transparent=usar_transparencia)

if __name__ == "__main__":
    main()
