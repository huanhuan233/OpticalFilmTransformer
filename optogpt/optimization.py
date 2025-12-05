# bfgs_optimization.py
# 专门用于对已知多层膜结构进行BFGS厚度优化的程序
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==============================
# 材料折射率数据（从生成器正式_v3.ipynb获取）
# ==============================
# TiO2 折射率数据
lam_tab_tio2 = np.array([380.0,425.0,450.0,475.0,500.0,525.0,550.0,575.0,600.0,
                         625.0,650.0,675.0,750.0,775.0,800.0,825.0,850.0,900.0,
                         1000.0,1060.0])
n_tab_tio2   = np.array([2.55,2.49,2.469,2.444,2.422,2.402,2.385,2.37,2.351,
                         2.343,2.337,2.331,2.322,2.317,2.313,2.311,2.309,2.305,
                         2.300,2.299])

def n_tio2(lam_nm):
    """TiO2折射率的线性插值计算"""
    return np.interp(lam_nm, lam_tab_tio2, n_tab_tio2)

# SiO2 折射率数据
lam_tab_sio2 = np.array([300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,900.0,1000.0])
n_tab_sio2   = np.array([1.478 ,1.472 ,1.467 ,1.463 ,1.459 ,1.455 ,1.452 ,1.450 ,1.446 ,1.437 ,1.434])

def n_sio2(lam_nm):
    """SiO2折射率的线性插值计算"""
    return np.interp(lam_nm, lam_tab_sio2, n_tab_sio2)

# MgF2 折射率数据
lam_tab_mgf2 = np.array([248.0, 550.0, 1550.0])
n_tab_mgf2   = np.array([1.40 , 1.38 , 1.36  ])

def n_mgf2(lam_nm):
    """MgF2折射率的线性插值计算"""
    return np.interp(lam_nm, lam_tab_mgf2, n_tab_mgf2)

# 玻璃基底折射率（常数）
glass_n_const = 1.5163

# 材料名称到折射率函数的映射
MATERIAL_REFRACTIVE_FUNCTIONS = {
    'MgF2': n_mgf2,
    'SiO2': n_sio2,
    'TiO2': n_tio2
}

# ==============================
# 传输矩阵法（垂直入射）
# ==============================
def tmm_reflectance(material_list, d_list, wavelength, n_inc=1.0, n_sub=glass_n_const):
    """
    计算多层膜在给定波长下的反射率
    
    参数:
    material_list: [material1, material2, ..., materialN] 膜层材料名称列表
    d_list: [d1, d2, ..., dN] 膜层厚度（nm）
    wavelength: 单个波长值（nm）
    n_inc: 入射介质折射率（默认1.0，空气）
    n_sub: 基底折射率（默认1.52，玻璃）
    
    返回:
    反射率值（0-1之间的浮点数）
    """
    k0 = 2 * np.pi / wavelength
    M = np.eye(2, dtype=complex)
    
    # 构建完整折射率序列：入射介质 + 膜层 + 基底
    n_full = [n_inc]
    for material in material_list:
        # 获取当前材料在当前波长下的折射率
        n_func = MATERIAL_REFRACTIVE_FUNCTIONS.get(material)
        if n_func is None:
            raise ValueError(f"未知材料: {material}")
        n_full.append(n_func(wavelength))
    n_full.append(n_sub)
    
    # 计算各层的传输矩阵乘积
    for i in range(1, len(n_full) - 1):
        n1 = n_full[i]
        d = d_list[i - 1]
        delta = k0 * n1 * d
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # 单层传输矩阵
        layer_M = np.array([[cos_d, 1j * sin_d / n1],
                           [1j * n1 * sin_d, cos_d]], dtype=complex)
        M = M @ layer_M
    
    # 计算总导纳和反射率
    Y_sub = n_sub
    Y_inc = n_inc
    
    B = M[0, 0] * Y_sub + M[0, 1]
    C = M[1, 0] * Y_sub + M[1, 1]
    Y_total = B / C
    
    r = (Y_inc - Y_total) / (Y_inc + Y_total)
    return np.abs(r) ** 2

# ==============================  
# BFGS优化函数
# ==============================
def bfgs_optimize_thickness(material_list, initial_d_list, wavelengths, target_R, 
                            n_inc=1.0, n_sub=glass_n_const, bounds=(20.0, 300.0)):
    """
    使用BFGS算法优化已知膜层结构的厚度
    
    参数:
    material_list: [material1, material2, ..., materialN] 膜层材料名称列表
    initial_d_list: [d1, d2, ..., dN] 初始膜层厚度（nm）
    wavelengths: 波长范围数组（nm）
    target_R: 目标反射率数组（与wavelengths同长度）
    n_inc: 入射介质折射率
    n_sub: 基底折射率
    bounds: 厚度约束范围（min, max）
    
    返回:
    optimized_d_list: 优化后的膜层厚度
    optimization_result: 优化器返回的结果对象
    """
    # 定义目标函数
    def objective_function(d_list):
        """计算当前厚度下的目标函数值"""
        # 添加额外的厚度检查，确保所有厚度都在有效范围内
        for i, d in enumerate(d_list):
            if d < bounds[0] or d > bounds[1]:
                # 如果厚度超出范围，返回一个很大的惩罚值
                return 1e20 + np.sum((np.array(d_list) - np.array(bounds[0])) ** 2) + np.sum((np.array(d_list) - np.array(bounds[1])) ** 2)
        
        R = np.array([tmm_reflectance(material_list, d_list, wl, n_inc, n_sub) for wl in wavelengths])
        return np.sum((R - target_R) ** 2)  # 均方误差
    
    # 设置边界条件（确保每一层都在20-300nm范围内）
    thickness_bounds = [bounds for _ in range(len(initial_d_list))]
    print(f"\n厚度限制：每层涂层厚度在 {bounds[0]}-{bounds[1]} nm 之间")
    
    # 执行BFGS优化
    result = minimize(objective_function, initial_d_list, 
                      method='L-BFGS-B', 
                      bounds=thickness_bounds, 
                      options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 100})
    
    return result.x.tolist(), result

# ==============================  
# 辅助函数：计算反射率曲线
# ==============================
def calculate_reflectance_spectrum(material_list, d_list, wavelengths, n_inc=1.0, n_sub=glass_n_const):
    """计算给定波长范围内的反射率光谱"""
    return np.array([tmm_reflectance(material_list, d_list, wl, n_inc, n_sub) for wl in wavelengths])

# ==============================  
# 辅助函数：绘制反射率曲线
# ==============================
def plot_reflectance(wavelengths, R_initial, R_optimized, target_R=None):
    """绘制优化前后的反射率曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, R_initial * 100, 'b--', linewidth=2, label='Initial')
    plt.plot(wavelengths, R_optimized * 100, 'r-', linewidth=2, label='Optimized')
    
    if target_R is not None:
        plt.plot(wavelengths, target_R * 100, 'g:', linewidth=2, label='Target')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance (%)')
    plt.title('Reflectance Spectrum Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, max(max(R_initial), max(R_optimized)) * 120)
    plt.tight_layout()
    plt.savefig('bfgs_optimization_results.png')
    plt.close()
    print("反射率曲线已保存为 'bfgs_optimization_results.png'")

# ==============================  
# 示例用法
# ==============================
if __name__ == "__main__":
    # 示例1：优化4层膜结构（MgF2/SiO2/TiO2/MgF2）
    print("=== BFGS膜层厚度优化示例 ===")
    
    # 波长范围（380-750nm，50个点）
    wavelengths = np.linspace(380, 750, 50)
    
    # 目标：宽带增透（反射率尽可能接近0）
    target_R = np.zeros_like(wavelengths)
    
    # 已知的膜层材料序列
    material_list = ['MgF2', 'SiO2', 'TiO2', 'MgF2']  # MgF2/SiO2/TiO2/MgF2（自顶向下）
    
    # 初始厚度（可以是任意合理值）
    initial_d_list = [60, 100, 50, 90]  # nm
    
    # 执行BFGS优化（明确指定厚度范围为20-300nm）
    print(f"初始膜层厚度: {initial_d_list} nm")
    optimized_d_list, result = bfgs_optimize_thickness(material_list, initial_d_list, wavelengths, target_R, bounds=(20.0, 300.0))
    
    # 计算优化前后的反射率
    R_initial = calculate_reflectance_spectrum(material_list, initial_d_list, wavelengths)
    R_optimized = calculate_reflectance_spectrum(material_list, optimized_d_list, wavelengths)
    
    # 输出结果
    print(f"\n优化结果:")
    print(f"优化是否成功: {result.success}")
    print(f"迭代次数: {result.nit}")
    print(f"最终目标函数值: {result.fun:.6f}")
    
    print(f"\n优化后的膜层厚度: {[round(d, 2) for d in optimized_d_list]} nm")
    
    # 计算平均反射率
    avg_R_initial = np.mean(R_initial) * 100
    avg_R_optimized = np.mean(R_optimized) * 100
    peak_R_initial = np.max(R_initial) * 100
    peak_R_optimized = np.max(R_optimized) * 100
    
    print(f"\n光学性能对比:")
    print(f"初始平均反射率: {avg_R_initial:.2f}%")
    print(f"优化后平均反射率: {avg_R_optimized:.2f}%")
    print(f"初始峰值反射率: {peak_R_initial:.2f}%")
    print(f"优化后峰值反射率: {peak_R_optimized:.2f}%")
    
    # 绘制结果
    plot_reflectance(wavelengths, R_initial, R_optimized, target_R)
    
    # 保存详细结果到文件
    with open('bfgs_optimization_details.txt', 'w') as f:
        f.write("BFGS膜层厚度优化结果\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"波长范围: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm\n")
        f.write(f"膜层结构: {', '.join(material_list)}\n")
        f.write(f"厚度限制: 每层涂层厚度在20-300 nm之间\n\n")
        
        f.write("初始厚度 (nm):\n")
        for i, (mat, d) in enumerate(zip(material_list, initial_d_list), 1):
            f.write(f"  层 {i}: {mat}, {d:.2f} nm\n")
        
        f.write("\n优化后厚度 (nm):\n")
        for i, (mat, d) in enumerate(zip(material_list, optimized_d_list), 1):
            f.write(f"  层 {i}: {mat}, {d:.2f} nm\n")
        
        f.write(f"\n光学性能:")
        f.write(f"\n  初始平均反射率: {avg_R_initial:.2f}%")
        f.write(f"\n  优化后平均反射率: {avg_R_optimized:.2f}%")
        f.write(f"\n  初始峰值反射率: {peak_R_initial:.2f}%")
        f.write(f"\n  优化后峰值反射率: {peak_R_optimized:.2f}%")
    
    print("\n详细结果已保存到 'bfgs_optimization_details.txt'")
    print("\n=== 优化完成 ===")