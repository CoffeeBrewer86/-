import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

def load_settings(settings_file="settings.txt"):
    """读取settings文件，返回参数字典。"""
    settings = {}
    if os.path.exists(settings_file):
        with open(settings_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().rstrip(",")  # 去掉多余逗号
                    try:
                        # 尝试转成 float 或 int
                        if "." in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass
                    settings[key] = val
    return settings

def _read_icc_bytes(icc_path: str):
    with open(icc_path, "rb") as f:
        return f.read()

def save_figure_as_p3_linear(fig, out_path: str, icc_path: str, dpi: int = None):
    """
    将当前 Matplotlib Figure 按“P3-linear”方式保存为 PNG，并内嵌 ICC。
    不做任何颜色变换：直接把 fig 的 RGB 像素当作 P3-linear 写入。
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)   # (H,W,4)
    rgb = buf[..., :3]                                           # 取 RGB
    icc_bytes = _read_icc_bytes(icc_path)
    im = Image.fromarray(rgb, mode="RGB")
    if dpi is None:
        dpi = int(fig.dpi)
    im.save(out_path, format="PNG", icc_profile=icc_bytes, dpi=(dpi, dpi))

def make_colorband_image(colors_float, height: int = 80):
    """
    colors_float: (N, 3) in [0,1], 已经是 linear Display-P3 数值
    返回: (H, N, 3) uint8，直接可写 PNG
    """
    colors = np.clip(np.asarray(colors_float, dtype=np.float32), 0.0, 1.0)
    band = np.tile(colors[None, :, :], (height, 1, 1))               # (H, N, 3)
    band_u8 = (band * 255.0 + 0.5).astype(np.uint8)
    return band_u8

def save_colorband_p3_linear_png(colors_float, out_path: str, icc_path: str, height: int = 80, dpi: int = 300):
    """
    将 (N,3) 的 P3-linear 色带保存为 PNG，并内嵌 ICC。
    """
    band_u8 = make_colorband_image(colors_float, height=height)
    icc_bytes = _read_icc_bytes(icc_path)
    img = Image.fromarray(band_u8, mode="RGB")
    # 嵌入 P3-linear ICC；注意：像素本身已经是 P3-linear，无需再做任何色彩转换
    img.save(out_path, format="PNG", icc_profile=icc_bytes, dpi=(dpi, dpi))

# ---- 1) D65 × CIE1931 → XYZ(λ) ----
def load_d65_cie(d65_csv, cie_csv, wl_min=380, wl_max=830, d_lambda=1):
    d65 = pd.read_csv(d65_csv)
    wl_d65 = pd.to_numeric(d65.iloc[:,0], errors="coerce").values
    spd    = pd.to_numeric(d65.iloc[:,1], errors="coerce").values

    cie = pd.read_csv(cie_csv)
    wl_cie = pd.to_numeric(cie.iloc[:,0], errors="coerce").values
    xbar   = pd.to_numeric(cie.iloc[:,1], errors="coerce").values
    ybar   = pd.to_numeric(cie.iloc[:,2], errors="coerce").values
    zbar   = pd.to_numeric(cie.iloc[:,3], errors="coerce").values

    wl = np.arange(wl_min, wl_max+1, d_lambda)
    S  = np.interp(wl, wl_d65, spd)
    x  = np.interp(wl, wl_cie, xbar)
    y  = np.interp(wl, wl_cie, ybar)
    z  = np.interp(wl, wl_cie, zbar)

    XYZ = np.vstack([S*x, S*y, S*z]).T  # (N,3)
    return wl, XYZ, S, float(d_lambda)

# ====== 修改：XYZ -> P3-linear（去掉错误的 M_P3_to_sRGB） ======
M_XYZ_to_P3 = np.array([
    [ 2.4934969, -0.9313836, -0.4027107],
    [-0.8294889,  1.7626640,  0.0236247],
    [ 0.0358458, -0.0761724,  0.9568845]
], dtype=float)

def xyz_to_p3_table(xyz_table):
    """
    输入: 每个波长处的 XYZ(λ)（线性）
    输出: 对应的 linear Display-P3(λ)（线性）
    """
    return xyz_table @ M_XYZ_to_P3.T

# ---- 3) Read UV-Vis txt ----
def load_uvvis_txt(path):
    rows, spec, skip = [], False, False
    with open(path, "r") as f:
        for line in f:
            if "# Spectra" in line:
                spec, skip = True, True
                continue
            if not spec:
                continue
            if skip:
                skip = False
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    wl = float(parts[0]); eps = float(parts[1])
                    rows.append((wl, eps))
                except:
                    pass
    arr = np.array(rows, dtype=float)
    return arr[:,0], arr[:,1]

# ---- 4) Main simulation ----
def simulate_solution(d65_csv, cie_csv, uvvis_txt,
                      c_min=0.0, c_max=0.05, steps=100, path_cm=1.0,
                      wl_min=380, wl_max=830, d_lambda=1,
                      icc_path: str = None,
                      save_prefix: str = None):
    wl, XYZ, S_in, dλ = load_d65_cie(d65_csv, cie_csv, wl_min, wl_max, d_lambda)
    RGB_lambda = xyz_to_p3_table(XYZ)
    
    wl_uv, eps = load_uvvis_txt(uvvis_txt)
    eps_g = np.interp(wl, wl_uv, eps, left=0.0, right=0.0)

    # Blank
    RGB_blank = RGB_lambda.sum(axis=0) * dλ
    P_blank   = np.sum(S_in) * dλ
    RGB_blank[RGB_blank == 0] = 1e-12

    concs = np.linspace(c_min, c_max, steps)
    colors_real, colors_opt, rgb_rel_raw = [], [], []

    for c in concs:
        T = 10.0 ** (-eps_g * c * path_cm)
        RGB_c = (RGB_lambda * T[:,None]).sum(axis=0) * dλ
        rgb_rel = RGB_c / RGB_blank
        rgb_rel_raw.append(rgb_rel.copy())

        # ---- 1) Realistic swatch ----
        rgb_disp1 = rgb_rel.copy()
        mx = float(np.max(rgb_disp1))
        if mx > 1.0:
            rgb_disp1 = rgb_disp1 / mx
        rgb_disp1 = np.where(rgb_disp1 < 0, 0.0, rgb_disp1)
        rgb_disp1 = np.clip(rgb_disp1, 0.0, 1.0)
        colors_real.append(rgb_disp1)

        # ---- 2) Optimized swatch (p^(1/3) scaling) ----
        P_c = np.sum(S_in * T) * dλ
        p = P_c / P_blank if P_blank > 0 else 1.0
        factor = p ** (-2/3)
        rgb_disp2 = rgb_rel * factor
        mx2 = float(np.max(rgb_disp2))
        if mx2 > 1.0:
            rgb_disp2 = rgb_disp2 / mx2
        rgb_disp2 = np.where(rgb_disp2 < 0, 0.0, rgb_disp2)
        rgb_disp2 = np.clip(rgb_disp2, 0.0, 1.0)
        colors_opt.append(rgb_disp2)

    colors_real = np.array(colors_real)
    colors_opt = np.array(colors_opt)
    rgb_rel_raw = np.array(rgb_rel_raw)

    # ---- Output dataframes ----
    df_curves = pd.DataFrame({
        "c": concs,
        "R_raw": rgb_rel_raw[:,0],
        "G_raw": rgb_rel_raw[:,1],
        "B_raw": rgb_rel_raw[:,2]
    })
    df_real = pd.DataFrame(np.column_stack([concs, colors_real]), columns=["c","R","G","B"])
    df_opt  = pd.DataFrame(np.column_stack([concs, colors_opt]),  columns=["c","R","G","B"])

    # ---- Plots ----
    fig, axs = plt.subplots(2,1, figsize=(6,2.4), sharex=True,dpi=150)
    axs[0].imshow([colors_real], extent=[concs.min(), concs.max(), 0, 1], aspect="auto")
    axs[0].set_yticks([]); axs[0].set_title("真实透射色带（D65白光校准）")
    axs[1].imshow([colors_opt], extent=[concs.min(), concs.max(), 0, 1], aspect="auto")
    axs[1].set_yticks([]); axs[1].set_title("优化透射色带 (深色区提亮)")
    axs[1].set_xlabel("Concentration (mol/L)")
    plt.tight_layout()
    plt.show()

    # ====== 新增：保存“第一张图”为 P3-linear + ICC（不做任何校正） ======
    if icc_path and save_prefix:
        save_figure_as_p3_linear(fig, f"{save_prefix}_figure1_p3linear.png", icc_path)
        print(f"[OK] 已保存第一张图：{save_prefix}_figure1_p3linear.png （嵌入ICC：{icc_path}）")

    # Raw RGB curves
    plt.figure(figsize=(8,4),dpi=100)
    plt.plot(concs, rgb_rel_raw[:,0], label="R raw",c="r")
    plt.plot(concs, rgb_rel_raw[:,1], label="G raw",c="g")
    plt.plot(concs, rgb_rel_raw[:,2], label="B raw",c="b")
    plt.legend(); plt.grid(alpha=0.3)
    plt.xlabel("Concentration (mol/L)")
    plt.ylabel("RGB / RGB_blank")
    plt.title("Raw RGB responses vs concentration")
    plt.show()

    # ====== 新增：按需保存为 P3-linear PNG（仅色带图像） ======
    if icc_path and save_prefix:
        try:
            save_colorband_p3_linear_png(colors_real, f"{save_prefix}_band_real_p3linear.png", icc_path, height=80, dpi=300)
            save_colorband_p3_linear_png(colors_opt,  f"{save_prefix}_band_opt_p3linear.png",  icc_path, height=80, dpi=300)
            print(f"[OK] 已保存：{save_prefix}_band_real_p3linear.png / {save_prefix}_band_opt_p3linear.png （嵌入ICC：{icc_path}）")
        except Exception as e:
            print("[WARN] 保存 P3-linear PNG 失败：", e)
    print("警告：matplotlib默认使用带Gamma的sRGB渲染，颜色请以保存文件为准")
    return df_curves, df_real, df_opt

def run():
    # 默认参数
    defaults = {
        "c_min": 0.0,
        "c_max": 0.001,
        "steps": 1000,
        "path_cm": 1.0,
    }

    # 读取settings
    settings = load_settings("settings.txt")

    # 应用读取到的设置，未读取到就打印提示
    params = {}
    for k, v in defaults.items():
        if k in settings:
            params[k] = settings[k]
        else:
            print(f"没有读取到设定 {k}，采用默认值 {v}")
            params[k] = v

    # 交互输入 uvvis_txt 路径
    uvvis_txt = input("请输入 uvvis_txt 文件路径: ").strip()
    if not os.path.exists(uvvis_txt):
        print(f"文件 {uvvis_txt} 不存在，退出。")
        return

    # 调用原函数
    df_curves, df_real, df_opt = simulate_solution(
        d65_csv="dataset/CIE_std_illum_D65.csv",
        cie_csv="dataset/CIE_xyz_1931_2deg.csv",
        uvvis_txt=uvvis_txt,
        c_min=params["c_min"],
        c_max=params["c_max"],
        steps=params["steps"],
        path_cm=params["path_cm"],
        wl_min=380,
        wl_max=830,
        d_lambda=1,
        icc_path="dataset/DisplayP3-Linear.icc",
        save_prefix="png/"
    )

run()
