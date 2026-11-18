import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CONSTANTES FÍSICAS Y DE NORMALIZACIÓN (SOL - JÚPITER)
# ==========================================
mu = 0.00095372                 # Masa relativa Sol-Júpiter
mu1 = 1.0 - mu                  # Masa normalizada del Sol
mu2 = mu                        # Masa normalizada de Júpiter

AU_KM = 149597870.7             # 1 AU en kilómetros
R_JUP_AU = 5.2044               # Semieje mayor de Júpiter en AU
R_JUP_KM = R_JUP_AU * AU_KM     # Unidad de distancia canónica (d = 1) en km

T_JUP_SEG = 11.862615 * 365.25 * 86400.0
V_JUP_KMS = (2.0 * np.pi * R_JUP_KM) / T_JUP_SEG  # ~ 13.0697 km/s

# Coordenadas fijas en el sistema rotativo normalizado (PUNTO L5: y < 0)
x_sol, y_sol, z_sol = -mu, 0.0, 0.0
x_jup, y_jup, z_jup = 1.0 - mu, 0.0, 0.0
x_L5,  y_L5,  z_L5  = 0.5 - mu, -np.sqrt(3.0) / 2.0, 0.0

CJ_UMBRAL_L5 = 3.0 - mu * (1.0 - mu)  # 2.99904719 (mismo umbral que L4 por simetría)

# ==========================================
# 2. DATOS REALES DE HORIZONS (3240 LAOCOON - SIN RUNGE-KUTTA)
# ==========================================
A_AU_LAOCOON   = 5.230757045096841
EC_LAOCOON     = 0.127866346606249
IN_DEG_LAOCOON = 2.335941547331426
OM_DEG_LAOCOON = 296.3071576313041
W_DEG_LAOCOON  = 15.46993865731654
MA_DEG_LAOCOON = 254.4667554224485

# Primer vector cartesiano baricéntrico en KM y KM/S (2026-Nov-16 00:00:00 TDB)
estado_horizons_0 = np.array([
    2.704355750186293E+08, 7.693429236697876E+08, 2.382187033120334E+07,
    -1.115805027056187E+01, 5.495941873563467E+00, -3.097435923830230E-01
])

# ==========================================
# 3. NORMALIZACIÓN, TRANSFORMACIONES Y JACOBI
# ==========================================
def inercial_km_a_rotativo_norm(estado_km_kms, theta_jup_rad):
    X_in = estado_km_kms[0] / R_JUP_KM
    Y_in = estado_km_kms[1] / R_JUP_KM
    Z_in = estado_km_kms[2] / R_JUP_KM
    
    VX_in = estado_km_kms[3] / V_JUP_KMS
    VY_in = estado_km_kms[4] / V_JUP_KMS
    VZ_in = estado_km_kms[5] / V_JUP_KMS
    
    c, s = np.cos(theta_jup_rad), np.sin(theta_jup_rad)
    
    x_rot =  X_in * c + Y_in * s
    y_rot = -X_in * s + Y_in * c
    z_rot =  Z_in
    
    vx_rot = ( VX_in * c + VY_in * s) + y_rot
    vy_rot = (-VX_in * s + VY_in * c) - x_rot
    vz_rot = VZ_in
    
    return np.array([x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot])

def inercial_a_rotativo(X_in, Y_in, Z_in, VX_in, VY_in, VZ_in, t):
    c, s = np.cos(t), np.sin(t)
    x_r =  X_in * c + Y_in * s
    y_r = -X_in * s + Y_in * c
    z_r =  Z_in
    vx_r = ( VX_in * c + VY_in * s) + y_r
    vy_r = (-VX_in * s + VY_in * c) - x_r
    vz_r = VZ_in
    return x_r, y_r, z_r, vx_r, vy_r, vz_r

def rotativo_a_inercial(x_r, y_r, z_r, t):
    X = x_r * np.cos(t) - y_r * np.sin(t)
    Y = x_r * np.sin(t) + y_r * np.cos(t)
    return X, Y, z_r

def jacobi_nasa_tisserand(a_au, e, i_deg):
    a_can = a_au / R_JUP_AU
    i_rad = np.radians(i_deg)
    return (1.0 / a_can) + 2.0 * np.sqrt(a_can * (1.0 - e**2)) * np.cos(i_rad)

def jacobi_potencial_efectivo(S_rot):
    x, y, z, vx, vy, vz = S_rot
    r1 = np.sqrt((x - x_sol)**2 + (y - y_sol)**2 + (z - z_sol)**2)
    r2 = np.sqrt((x - x_jup)**2 + (y - y_jup)**2 + (z - z_jup)**2)
    V_ef = -0.5 * (x**2 + y**2) - (mu1 / r1) - (mu2 / r2)
    return -2.0 * V_ef - (vx**2 + vy**2 + vz**2)

theta_jup_0 = np.radians(11.2)
S0_nasa_rot = inercial_km_a_rotativo_norm(estado_horizons_0, theta_jup_0)
CJ_nasa_tisserand = jacobi_nasa_tisserand(A_AU_LAOCOON, EC_LAOCOON, IN_DEG_LAOCOON)

# ==========================================
# 4. EFEMÉRIDES 3D EN REGIÓN L5 (SIN RUNGE-KUTTA) Y ESTADO INICIAL EXACTO
# ==========================================
def efemerides_l5_sin_rk4(t_array):
    """
    Calcula la trayectoria 3D kepleriana proyectada en el punto troyano L5 (-60 deg)
    y devuelve tanto las coordenadas inerciales (X, Y, Z) como el vector de estado
    rotativo en t = 0 sobre el mismo plano inclinado 3D.
    """
    a = A_AU_LAOCOON / R_JUP_AU
    e = EC_LAOCOON
    i = np.radians(IN_DEG_LAOCOON)
    Omega = np.radians(OM_DEG_LAOCOON)
    omega = np.radians(W_DEG_LAOCOON)
    
    M_t = np.radians(MA_DEG_LAOCOON) + 1.0 * t_array
    
    E_t = M_t.copy()
    for _ in range(25):
        E_t = E_t - (E_t - e * np.sin(E_t) - M_t) / (1.0 - e * np.cos(E_t))
        
    nu_t = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E_t / 2.0),
                            np.sqrt(1.0 - e) * np.cos(E_t / 2.0))
    r_t = a * (1.0 - e * np.cos(E_t))
    h = np.sqrt(mu1 * a * (1.0 - e**2))
    
    x_orb = r_t * np.cos(nu_t)
    y_orb = r_t * np.sin(nu_t)
    vx_orb = -(mu1 / h) * np.sin(nu_t)
    vy_orb = (mu1 / h) * (e + np.cos(nu_t))
    
    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_w, sin_w = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    
    R11 = cos_O * cos_w - sin_O * sin_w * cos_i
    R12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    R21 = sin_O * cos_w + cos_O * sin_w * cos_i
    R22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    R31, R32 = sin_w * sin_i, cos_w * sin_i
    
    X_hel = R11 * x_orb + R12 * y_orb
    Y_hel = R21 * x_orb + R22 * y_orb
    Z_hel = R31 * x_orb + R32 * y_orb
    
    VX_hel = R11 * vx_orb + R12 * vy_orb
    VY_hel = R21 * vx_orb + R22 * vy_orb
    VZ_hel = R31 * vx_orb + R32 * vy_orb
    
    # Alinear el plano completo con Júpiter en t = 0 en la región de L5 (-60 deg)
    ang_pos_0 = np.arctan2(Y_hel[0], X_hel[0])
    d_ang = np.radians(-60.0) - ang_pos_0
    c_a, s_a = np.cos(d_ang), np.sin(d_ang)
    
    X_in = X_hel * c_a - Y_hel * s_a
    Y_in = X_hel * s_a + Y_hel * c_a
    Z_in = Z_hel
    
    VX_in = VX_hel * c_a - VY_hel * s_a
    VY_in = VX_hel * s_a + VY_hel * c_a
    VZ_in = VZ_hel
    
    # Vector de estado rotativo en t = 0 correspondiente a este plano en L5
    xr0, yr0, zr0, vxr0, vyr0, vzr0 = inercial_a_rotativo(
        X_in[0] - mu, Y_in[0], Z_in[0], VX_in[0], VY_in[0] - mu, VZ_in[0], 0.0
    )
    S0_plano_L5 = np.array([xr0, yr0, zr0, vxr0, vyr0, vzr0])
    
    return X_in, Y_in, Z_in, S0_plano_L5

# ==========================================
# 5. INTEGRADOR RK4 Y TANDA DE REFINAMIENTO SOBRE EL PLANO EN L5
# ==========================================
def derivar_sistema_3d(t, S):
    x, y, z, vx, vy, vz = S
    r1 = np.sqrt((x - x_sol)**2 + (y - y_sol)**2 + (z - z_sol)**2)
    r2 = np.sqrt((x - x_jup)**2 + (y - y_jup)**2 + (z - z_jup)**2)
    
    ax = 2.0 * vy + x - (mu1 * (x - x_sol) / r1**3) - (mu2 * (x - x_jup) / r2**3)
    ay = -2.0 * vx + y - (mu1 * y / r1**3) - (mu2 * y / r2**3)
    az = - (mu1 * z / r1**3) - (mu2 * z / r2**3)
    
    return np.array([vx, vy, vz, ax, ay, az])

def rk4_step_3d(f, t, S, dt):
    k1 = f(t, S)
    k2 = f(t + 0.5 * dt, S + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, S + 0.5 * dt * k2)
    k4 = f(t + dt, S + dt * k3)
    return S + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

periodo_jup = 2.0 * np.pi
anios = 30.0
t_max = anios * periodo_jup
dt = 0.005
pasos = int(t_max / dt)
t_eval = np.linspace(0, t_max, pasos)

# Trayectoria de referencia en L5 (SIN Runge-Kutta) y estado inicial coplanar
X_nasa_full, Y_nasa_full, Z_nasa_full, S0_objetivo = efemerides_l5_sin_rk4(t_eval)
CJ_nasa_full = np.full(pasos, CJ_nasa_tisserand)

# Perturbaciones suaves alrededor del estado objetivo en L5
S0_base_mismo_plano = S0_objetivo + np.array([-0.03, -0.03, 0.005, -0.03, 0.03, -0.005])
factores_finos = [0.70, 0.88, 0.96, 1.00]
colores = ["limegreen", "orange", "deeppink", "red"]
simulaciones_teo = []

print("="*78)
print("REFINAMIENTO DE ÓRBITA (RK4) EN EL PUNTO TROYANO L5 (-60°)")
print("="*78)
print(f"Objetivo Referencia L5 (Sin RK4) -> C_J Tisserand = {CJ_nasa_tisserand:.8f}\n")

for idx, alpha in enumerate(factores_finos, start=1):
    S0_prueba = (1.0 - alpha) * S0_base_mismo_plano + alpha * S0_objetivo
    
    tray_rot = np.zeros((pasos, 6))
    CJ_arr   = np.zeros(pasos)
    tray_rot[0] = S0_prueba
    CJ_arr[0]   = jacobi_potencial_efectivo(S0_prueba)
    
    for i in range(1, pasos):
        tray_rot[i] = rk4_step_3d(derivar_sistema_3d, t_eval[i-1], tray_rot[i-1], dt)
        CJ_arr[i]   = jacobi_potencial_efectivo(tray_rot[i])
        
    X_in, Y_in, Z_in = rotativo_a_inercial(tray_rot[:, 0], tray_rot[:, 1], tray_rot[:, 2], t_eval)
    
    dist_media = np.mean(np.sqrt((X_in - X_nasa_full)**2 + (Y_in - Y_nasa_full)**2 + (Z_in - Z_nasa_full)**2))
    diff_CJ    = abs(CJ_arr[0] - CJ_nasa_tisserand)
    
    etiqueta = f"RK4 Fino {idx} ({alpha*100:.0f}%)" if alpha < 1.0 else "RK4 Óptimo (100%)"
    simulaciones_teo.append({
        "etiqueta": etiqueta,
        "color": colores[idx - 1],
        "lw": 1.1 if alpha < 1.0 else 2.2,
        "alpha_vis": 0.55 if alpha < 1.0 else 0.95,
        "X": X_in, "Y": Y_in, "Z": Z_in,
        "CJ": CJ_arr
    })
    
    print(f"{etiqueta}:")
    print(f"  CI [x0, y0, z0]    : [{S0_prueba[0]:.5f}, {S0_prueba[1]:.5f}, {S0_prueba[2]:.5f}]")
    print(f"  CI [vx0, vy0, vz0] : [{S0_prueba[3]:.5f}, {S0_prueba[4]:.5f}, {S0_prueba[5]:.5f}]")
    print(f"  C_J obtenida       : {CJ_arr[0]:.8f} (Diferencia con Ref L5: {diff_CJ:.6f})")
    print(f"  Distancia orbital media a Ref L5: {dist_media:.5f} AU_norm\n")

print("="*78)

# ==========================================
# 6. SUBMUESTREO Y ANIMACIÓN 3D + CONVERGENCIA DE JACOBI
# ==========================================
fps = 30
anios_por_segundo = 2.0
pasos_por_frame = int((anios_por_segundo * periodo_jup) / (dt * fps))

t_anim = t_eval[::pasos_por_frame]
anios_anim = t_anim / periodo_jup

X_sol, Y_sol, Z_sol = rotativo_a_inercial(x_sol, y_sol, np.zeros_like(t_anim), t_anim)
X_jup, Y_jup, Z_jup = rotativo_a_inercial(x_jup, y_jup, np.zeros_like(t_anim), t_anim)
X_L5,  Y_L5,  Z_L5  = rotativo_a_inercial(x_L5,  y_L5,  np.zeros_like(t_anim), t_anim)

X_nasa = X_nasa_full[::pasos_por_frame]
Y_nasa = Y_nasa_full[::pasos_por_frame]
Z_nasa = Z_nasa_full[::pasos_por_frame]
CJ_nasa_anim = CJ_nasa_full[::pasos_por_frame]

for sim in simulaciones_teo:
    sim["X_a"]  = sim["X"][::pasos_por_frame]
    sim["Y_a"]  = sim["Y"][::pasos_por_frame]
    sim["Z_a"]  = sim["Z"][::pasos_por_frame]
    sim["CJ_a"] = sim["CJ"][::pasos_por_frame]

fig = plt.figure(figsize=(15.5, 7.5), dpi=110)

# --- Panel Izquierdo: Órbita 3D ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim(-1.3, 1.3)
ax1.set_ylim(-1.3, 1.3)
ax1.set_zlim(-0.15, 0.15)
ax1.set_title("Superposición Orbital 3D en Punto L5 (RK4 vs Referencia L5)", fontsize=10, fontweight='bold')
ax1.set_xlabel("X (Inercial)")
ax1.set_ylabel("Y (Inercial)")
ax1.set_zlabel("Z (Inclinación)")

theta = np.linspace(0, 2*np.pi, 200)
ax1.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 'k:', alpha=0.25, label='Órbita Júpiter')

sol_p,     = ax1.plot([], [], [], 'o', color='gold', markersize=12, label='Sol')
jup_p,     = ax1.plot([], [], [], 'o', color='peru', markersize=8, label='Júpiter')
l5_p,      = ax1.plot([], [], [], 'b*', markersize=10, label='Punto L5')
triangulo, = ax1.plot([], [], [], 'k--', alpha=0.25, lw=1)

nasa_p,    = ax1.plot([], [], [], 'co', markersize=7, markeredgecolor='k', label='Referencia L5 (Sin RK4)')
nasa_cola, = ax1.plot([], [], [], 'c--', lw=2.0, alpha=0.95)

teo_puntos = []
teo_colas  = []
for sim in simulaciones_teo:
    p, = ax1.plot([], [], [], 'o', color=sim["color"], markersize=4 if sim["lw"] < 1.5 else 6)
    c, = ax1.plot([], [], [], '-', color=sim["color"], lw=sim["lw"], alpha=sim["alpha_vis"], label=sim["etiqueta"])
    teo_puntos.append(p)
    teo_colas.append(c)

ax1.legend(loc='upper right', fontsize=7.2)

# --- Panel Derecho: Constante de Jacobi vs Tiempo ---
ax2 = fig.add_subplot(122)
ax2.set_xlim(0, anios)
ax2.set_ylim(2.978, 2.988)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_title("Constante de Jacobi ($C_J$) en L5: Convergencia Fina", fontsize=10.5, fontweight='bold')
ax2.set_xlabel("Tiempo (Años jovianos)")
ax2.set_ylabel("Constante de Jacobi ($C_J$)")

line_nasa, = ax2.plot([], [], 'c--', lw=2.4, label=f'Ref. Tisserand L5 ($C_J = {CJ_nasa_tisserand:.5f}$)')
pt_nasa,   = ax2.plot([], [], 'co', markersize=6, markeredgecolor='k')

lines_teo = []
pts_teo   = []
for sim in simulaciones_teo:
    l, = ax2.plot([], [], '-', color=sim["color"], lw=sim["lw"], alpha=sim["alpha_vis"],
                  label=f"{sim['etiqueta']} ($C_J = {sim['CJ'][0]:.5f}$)")
    pt, = ax2.plot([], [], 'o', color=sim["color"], markersize=4)
    lines_teo.append(l)
    pts_teo.append(pt)

ax2.legend(loc='upper right', fontsize=8)

texto_tiempo = ax2.text(0.03, 0.90, '', transform=ax2.transAxes, fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

largo_cola = 45

def init():
    for art in [sol_p, jup_p, l5_p, triangulo, nasa_p, nasa_cola] + teo_puntos + teo_colas:
        art.set_data([], [])
        art.set_3d_properties([])
    for l in [line_nasa, pt_nasa] + lines_teo + pts_teo:
        l.set_data([], [])
    texto_tiempo.set_text('')
    return [sol_p, jup_p, l5_p, triangulo, nasa_p, nasa_cola, line_nasa, pt_nasa, texto_tiempo] + teo_puntos + teo_colas + lines_teo + pts_teo

def update(frame):
    ini = max(0, frame - largo_cola)
    
    xs, ys, zs = [X_sol[frame]], [Y_sol[frame]], [Z_sol[frame]]
    xj, yj, zj = [X_jup[frame]], [Y_jup[frame]], [Z_jup[frame]]
    xl, yl, zl = [X_L5[frame]],  [Y_L5[frame]],  [Z_L5[frame]]
    
    sol_p.set_data(xs, ys); sol_p.set_3d_properties(zs)
    jup_p.set_data(xj, yj); jup_p.set_3d_properties(zj)
    l5_p.set_data(xl, yl);  l5_p.set_3d_properties(zl)
    
    triangulo.set_data([xs[0], xj[0], xl[0], xs[0]], [ys[0], yj[0], yl[0], ys[0]])
    triangulo.set_3d_properties([zs[0], zj[0], zl[0], zs[0]])
    
    nasa_p.set_data([X_nasa[frame]], [Y_nasa[frame]])
    nasa_p.set_3d_properties([Z_nasa[frame]])
    nasa_cola.set_data(X_nasa[ini:frame], Y_nasa[ini:frame])
    nasa_cola.set_3d_properties(Z_nasa[ini:frame])
    
    line_nasa.set_data(anios_anim[:frame], CJ_nasa_anim[:frame])
    pt_nasa.set_data([anios_anim[frame]], [CJ_nasa_anim[frame]])
    
    for idx, sim in enumerate(simulaciones_teo):
        teo_puntos[idx].set_data([sim["X_a"][frame]], [sim["Y_a"][frame]])
        teo_puntos[idx].set_3d_properties([sim["Z_a"][frame]])
        teo_colas[idx].set_data(sim["X_a"][ini:frame], sim["Y_a"][ini:frame])
        teo_colas[idx].set_3d_properties(sim["Z_a"][ini:frame])
        
        lines_teo[idx].set_data(anios_anim[:frame], sim["CJ_a"][:frame])
        pts_teo[idx].set_data([anios_anim[frame]], [sim["CJ_a"][frame]])
        
    texto_tiempo.set_text(f"Tiempo: {t_anim[frame]:.1f} $t_0$ ({anios_anim[frame]:.1f} años)")
    
    return [sol_p, jup_p, l5_p, triangulo, nasa_p, nasa_cola, line_nasa, pt_nasa, texto_tiempo] + teo_puntos + teo_colas + lines_teo + pts_teo

ani = animation.FuncAnimation(fig, update, frames=len(t_anim), init_func=init, interval=33, blit=True)

plt.tight_layout()
plt.show()