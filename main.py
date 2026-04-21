import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata

def plot_chapa(L, H, raio, tensao):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=90)
    
    chapa = patches.Rectangle((0, 0), L, H, lw=2, ec='black', fc='#f0f0f0', zorder=1)
    ax.add_patch(chapa)
    
    furo = patches.Circle((L/2, H/2), raio, fc='white', ec='black', lw=1.5, zorder=2)
    ax.add_patch(furo)
    
    num_hachuras = 15
    for i in np.linspace(0, H, num_hachuras):
        ax.plot([-L*0.04, 0], [i, i-H*0.04], color='black', lw=1)
    ax.plot([0, 0], [0, H], color='black', lw=3)
    ax.text(-L*0.06, H/2, 'Engaste Fixo\n(u=0, v=0)', ha='right', va='center', fontweight='bold')
    
    y_setas = np.linspace(H*0.1, H*0.9, 7)
    for y_p in y_setas:
        ax.annotate('', xy=(L + L*0.1, y_p), xytext=(L, y_p),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(L + L*0.02, H + H*0.02, f'Tração Axial $\sigma = {tensao}$ MPa', color='red', fontweight='bold')
    
    ax.axhline(H/2, color='gray', ls='-.', lw=0.8, zorder=0)
    ax.axvline(L/2, color='gray', ls='-.', lw=0.8, zorder=0)
    
    # Dimensões
    ax.annotate('', xy=(0, -H*0.08), xytext=(L, -H*0.08), arrowprops=dict(arrowstyle='<->'))
    ax.text(L/2, -H*0.15, f'L = {L}mm', ha='center')
    ax.annotate('', xy=(L*0.1, 0), xytext=(L*0.1, H), arrowprops=dict(arrowstyle='<->'))
    ax.text(L*0.06, H/2, f'H = {H}mm', rotation=90, va='center')

    ax.set_title('Problema simulado:', fontsize=14, pad=20)
    ax.set_xlim(-L*0.3, L*1.3)
    ax.set_ylim(-H*0.2, H*1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def simulacao_saint_venant_mef(L = 200.0, H = 100.0, raio_furo = 10.0, tensao_axial = 100.0, comparativo=False):
    # --- Parâmetros de Entrada ---
    # Geometria e Carga
    L, H = 200.0, 100.0  # mm
    raio_furo = 10.0         # mm
    tensao_axial = 100.0  # MPa (Tensão nominal de tração)
    
    # Malha
    nx, ny = 600, 300 
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)

    # Centro do furo e coordenadas polares
    Xc, Yc = L/2, H/2
    R = np.sqrt((X - Xc)**2 + (Y - Yc)**2)
    # Theta é o ângulo em radianos, de -pi a pi
    Theta = np.arctan2(Y - Yc, X - Xc)

    # --- Kirsch (Sigma_xx) ---
    a = raio_furo
    term1 = 1 - (a**2 / R**2) * (1.5 * np.cos(2*Theta) + np.cos(4*Theta))
    term2 = 1.5 * (a**4 / R**4) * np.cos(4*Theta)
    s_x = tensao_axial * (term1 + term2)

    s_x[R < a] = np.nan

    vmin_plot = -100.0 
    vmax_plot = tensao_axial * 3.5 
    
    if comparativo:
        fig = plt.figure(figsize=(12, 11))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
        axes_sections = [fig.add_subplot(gs[1, 0])]
    else:
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
        axes_sections = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    ax_map = fig.add_subplot(gs[0, :])

    cp = ax_map.contourf(X, Y, s_x, levels=120, cmap='jet', vmin=vmin_plot, vmax=vmax_plot)
    cbar = fig.colorbar(cp, ax=ax_map)
    cbar.set_label('Tensão Longitudinal $\sigma_{xx}$ (MPa)', fontsize=13)
    
    furo = plt.Circle((Xc, Yc), raio_furo, color='white', zorder=12, ec='black', lw=1.2)
    ax_map.add_patch(furo)
    
    ax_map.set_title('Campo de Tensões $\sigma_{xx}$', fontsize=15, fontweight='bold')
    ax_map.set_xlabel('Comprimento (mm)', fontsize=12)
    ax_map.set_ylabel('Altura (mm)', fontsize=12)
    ax_map.set_aspect('equal')
    ax_map.tick_params(labelsize=11)

    x_coords = [Xc, Xc + 2*raio_furo, Xc + 10*raio_furo]
    colors = ['#d62728', '#2ca02c', '#1f77b4'] 
    labels = ['Seção no Furo ($x = L/2$)', 'Seção Intermediária ($x = L/2 + 2R$)', 'Região de Saint-Venant ($x = L/2 + 10R$)']
    
    y_fine = np.linspace(0, H, 800)
    
    for i, (sx, color, lab) in enumerate(zip(x_coords, colors, labels)):
        stress_slice = griddata((X.flatten(), Y.flatten()), s_x.flatten(), 
                                (np.full_like(y_fine, sx), y_fine), method='linear')
        
        target_ax = axes_sections[0] if comparativo else axes_sections[i]
        target_ax.plot(stress_slice, y_fine, color=color, lw=3, label=lab)
        
        if not comparativo:
            target_ax.set_title(lab, fontsize=13)
            target_ax.set_xlim(vmin_plot * 1.1, vmax_plot * 1.1)
            target_ax.axvline(0, color='black', lw=1, ls='-')
        
        target_ax.axvline(tensao_axial, color='gray', ls='--', alpha=0.7, lw=1.5, label='Tensão Nominal ($S$)')
        target_ax.grid(True, linestyle=':', alpha=0.6)
        target_ax.set_xlabel('$\sigma_{xx}$ (MPa)', fontsize=12)
        target_ax.tick_params(labelsize=11)

    axes_sections[0].set_ylabel('Altura da Chapa $y$ (mm)', fontsize=12)
    if comparativo: axes_sections[0].legend(fontsize=11)

    plt.tight_layout()
    plt.show()

def executar_simulacao(largura = 200.0, altura = 100.0, diametro_furo = 20.0, tensao_tracao = 100.0, graficos_juntos=False):
    plot_chapa(L=largura, H=altura, raio=diametro_furo/2, tensao=tensao_tracao)
    print('\nAguarde o processamento finalizar\n')
    simulacao_saint_venant_mef(L = largura, H = altura, raio_furo = diametro_furo/2, tensao_axial = tensao_tracao, comparativo=graficos_juntos)


if __name__ == "__main__":
    # 1. Gera a figura técnica isolada
    plot_chapa(L=200.0, H=100.0, raio=10.0, tensao=100.0)
    
    # 2. Gera os resultados
    simulacao_saint_venant_mef(comparativo=False)
