"""
优化的湍流场生成程序 - 基于TKE和epsilon的调谐版本
生成1000个样本（100组参数 × 10个随机种子）

改进:
1. 输入条件改为TKE和epsilon
2. 使用调谐方程求解每组参数对应的new_alpha和new_kappa_e
3. 网格参数：L = 0.18 * π, N = 256
4. TKE范围(0.01, 1.0), epsilon范围(0.01, 1.0)
5. 每个参数组10个样本，共100×10=1000个样本
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift
from numpy import pi, sqrt, conj
from scipy.special import hyperu
from scipy.optimize import root_scalar
from concurrent.futures import ProcessPoolExecutor
import os
import json
from datetime import datetime
import time

# ================================================================
# 配置参数
# ================================================================
class Config:
    """生成配置"""
    # 生成参数
    N_TKE_SAMPLES = 10          # TKE参数样本数
    N_EPS_SAMPLES = 10          # epsilon参数样本数
    N_SEEDS_PER_PARAM = 10      # 每组参数的随机种子数
    TOTAL_SAMPLES = N_TKE_SAMPLES * N_EPS_SAMPLES * N_SEEDS_PER_PARAM  # 1000
    
    # 参数范围
    TKE_BASELINE = 0.09375
    EPS_BASELINE = 0.5402
    TKE_MIN = 0.5 * TKE_BASELINE              # 最小TKE [m²/s²]
    TKE_MAX = 2.0 * TKE_BASELINE               # 最大TKE [m²/s²]
    EPS_MIN = 0.5 * EPS_BASELINE            # 最小epsilon [m²/s³]
    EPS_MAX = 2.0 * EPS_BASELINE              # 最大epsilon [m²/s³]
    
    # 并行参数
    N_PARALLEL = 20              # 并行处理数
    
    # 物理参数
    L = 0.18 * np.pi            # 域长度 [m]
    N = 256                     # 网格点数
    M = 10000                   # 傅里叶模态数
    
    # 流体性质（空气）
    nu = 1e-5                   # 分子粘度 [m²/s]
    
    # 输出设置
    OUTPUT_ROOT = "TurbulenceData_1000_Tuned"
    SAVE_SPECTRUM_PLOTS = True
    PLOT_INTERVAL = 100
    
    def __init__(self):
        self.dx = self.L / self.N
        self.k_min = 2 * np.pi / self.L
        self.k_max = np.pi / self.dx
        
        # 创建输出目录结构
        self.dir_fields = os.path.join(self.OUTPUT_ROOT, "velocity_fields")
        self.dir_spectra = os.path.join(self.OUTPUT_ROOT, "spectra_plots")
        self.dir_metadata = os.path.join(self.OUTPUT_ROOT, "metadata")
        
        for directory in [self.dir_fields, self.dir_spectra, self.dir_metadata]:
            os.makedirs(directory, exist_ok=True)
        
        print("=" * 70)
        print("湍流场生成配置 - 调谐版本")
        print("=" * 70)
        print(f"TKE参数数: {self.N_TKE_SAMPLES}")
        print(f"Epsilon参数数: {self.N_EPS_SAMPLES}")
        print(f"每组参数种子数: {self.N_SEEDS_PER_PARAM}")
        print(f"目标样本总数: {self.TOTAL_SAMPLES}")
        print(f"网格尺寸: {self.N} × {self.N}")
        print(f"域长度: {self.L:.6f} m")
        print(f"TKE范围: [{self.TKE_MIN}, {self.TKE_MAX}] m²/s²")
        print(f"Epsilon范围: [{self.EPS_MIN}, {self.EPS_MAX}] m²/s³")
        print(f"波数范围: [{self.k_min:.2f}, {self.k_max:.2f}] m⁻¹")
        print(f"并行数: {self.N_PARALLEL}")
        print(f"输出目录: {self.OUTPUT_ROOT}")
        print("=" * 70 + "\n")

config = Config()

# ================================================================
# 调谐方程求解
# ================================================================
def solve_tuning_equation(tke, epsilon, nu):
    """
    根据TKE和epsilon求解调谐方程，得到new_alpha和new_kappa_e
    
    参数:
        tke: 湍动能 [m²/s²]
        epsilon: 耗散率 [m²/s³]
        nu: 分子粘度 [m²/s]
    
    返回:
        new_alpha: 调谐后的缩放系数
        new_kappa_e: 调谐后的能量峰值波数 [m⁻¹]
        u_prime: RMS速度 [m/s]
    """
    # 从TKE计算u_prime
    # k = (3/2) * u_prime^2
    u_prime = np.sqrt(2.0 * tke / 3.0)
    
    # 从epsilon和u_prime计算积分长度尺度
    L = u_prime**3 / epsilon
    
    # 计算Kolmogorov波数
    kappa_eta = (epsilon / nu**3)**0.25
    
    # 计算湍流雷诺数
    Re_L = tke**2 / (epsilon * nu)
    
    # 定义调谐方程
    def tuning_equation(z):
        """文献中的调谐方程 (Eq.10)"""
        try:
            U1 = hyperu(7/2, 5/3, z)
            U2 = hyperu(5/2, 2/3, z)
            
            if U2 == 0 or np.isnan(U1) or np.isnan(U2):
                return 1e10
            
            left_side = z * U1 / U2
            right_side = (2/5) * Re_L**(-0.5)
            
            return left_side - right_side
        except:
            return 1e10
    
    # 求解调谐方程
    try:
        sol = root_scalar(tuning_equation, bracket=[1e-10, 1000], method='brentq', xtol=1e-10)
        z_solution = sol.root
    except:
        # 如果求解失败，使用默认值
        print(f"  ⚠️  调谐方程求解失败 (TKE={tke:.4f}, eps={epsilon:.4f})，使用默认参数")
        z_solution = 0.5
    
    # 计算new_alpha和new_kappa_e
    try:
        U_new = hyperu(5/2, 2/3, z_solution)
        new_alpha = 4 / (np.sqrt(np.pi) * U_new)
        new_kappa_e = kappa_eta * np.sqrt(z_solution / 2)
    except:
        # 备用默认值
        new_alpha = 1.453
        new_kappa_e = 40 * np.sqrt(5/12)
    
    return new_alpha, new_kappa_e, u_prime, kappa_eta

# ================================================================
# 生成参数组合
# ================================================================
def generate_parameter_combinations():
    """生成TKE和epsilon的参数组合"""
    # 在对数空间生成均匀分布的参数
    tke_values = np.linspace(config.TKE_MIN, config.TKE_MAX, 
                             config.N_TKE_SAMPLES)
    eps_values = np.linspace(config.EPS_MIN, config.EPS_MAX, 
                             config.N_EPS_SAMPLES)
    
    # 生成所有组合
    param_combinations = []
    param_id = 0
    for tke in tke_values:
        for eps in eps_values:
            param_combinations.append({
                'param_id': param_id,
                'tke': tke,
                'epsilon': eps
            })
            param_id += 1
    
    return param_combinations

# ================================================================
# 能谱模型
# ================================================================
def von_karman_pao_tuned(k, alpha, kappa_e, u_prime, kappa_eta):
    """调谐后的Von Kármán-Pao能谱模型"""
    term1 = alpha * (u_prime**2 / kappa_e)
    term2 = (k / kappa_e)**4
    term3 = (1 + (k / kappa_e)**2)**(-17/6)
    exp_term = np.exp(-2 * (k / kappa_eta)**2)
    return term1 * term2 * term3 * exp_term

# ================================================================
# 随机模态生成
# ================================================================
def generate_random_modes(seed, k_m, u_m, M):
    """生成2D随机傅里叶模态"""
    rng = np.random.RandomState(seed)
    
    # 波矢方向
    phi = rng.uniform(0, 2 * np.pi, M)
    k_x = k_m * np.cos(phi)
    k_y = k_m * np.sin(phi)
    k_vec = np.stack((k_x, k_y), axis=-1)
    
    # 速度方向 (正交于k)
    alpha_angle = rng.uniform(0, 2 * np.pi, M)
    sigma = np.zeros((M, 2))
    
    for i in range(M):
        # 选择初始向量
        if abs(k_vec[i, 0]) > 0.9:
            a = np.array([0, 1])
        else:
            a = np.array([1, 0])
        
        # 90度旋转（正交化）
        b1 = np.array([-k_vec[i, 1], k_vec[i, 0]])
        b1 /= np.linalg.norm(b1)
        
        # 组合
        sigma[i] = np.cos(alpha_angle[i]) * b1 + np.sin(alpha_angle[i]) * a
    
    # 相位
    psi = rng.uniform(0, 2 * np.pi, M)
    
    return k_vec, sigma, psi

# ================================================================
# 速度场生成
# ================================================================
def compute_velocity_field(seed, k_m, u_m, M, L, N):
    """计算速度场"""
    # 生成随机模态
    k_vec, sigma, psi = generate_random_modes(seed, k_m, u_m, M)
    
    # 网格
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    # 初始化速度场
    u_field = np.zeros((N, N, 2), dtype=np.float64)
    
    # 累加所有模态贡献
    for i in range(M):
        phase = k_vec[i, 0] * X + k_vec[i, 1] * Y + psi[i]
        u_field += 2.0 * u_m[i] * np.cos(phase)[..., None] * sigma[i]
    
    return u_field

# ================================================================
# 能谱计算
# ================================================================
def cal_spectrum_iso(u_field, L):
    """计算各向同性能谱"""
    N = u_field.shape[0]
    nt = N**2
    
    # FFT
    u_hat = fftn(u_field[..., 0]) / nt
    v_hat = fftn(u_field[..., 1]) / nt
    
    u_hat = fftshift(u_hat)
    v_hat = fftshift(v_hat)
    
    # 能量
    tkef = 0.5 * (u_hat * conj(u_hat) + v_hat * conj(v_hat)).real
    
    k0 = 2.0 * pi / L
    knorm = k0
    
    kappa_kr = knorm * np.arange(0, N)
    tke_kr = np.zeros(len(kappa_kr))
    
    # 径向积分
    for i in range(-N//2, N//2):
        for j in range(-N//2, N//2):
            rk = sqrt(i**2 + j**2)
            kr = int(np.round(rk))
            if kr < len(tke_kr):
                tke_kr[kr] += tkef[i + N//2, j + N//2]
    
    tke_kr = tke_kr / knorm
    knyquist = knorm * N / 2.0
    
    return knyquist, knorm, kappa_kr, tke_kr

# ================================================================
# 单个样本生成
# ================================================================
def generate_single_sample(task):
    """生成单个样本"""
    param_id = task['param_id']
    tke = task['tke']
    epsilon = task['epsilon']
    seed = task['seed']
    sample_id = task['sample_id']
    
    try:
        start_time = time.time()
        
        # 求解调谐方程
        new_alpha, new_kappa_e, u_prime, kappa_eta = solve_tuning_equation(
            tke, epsilon, config.nu)
        
        # 生成波数网格
        modes = np.arange(1, config.M + 1)
        k_m = config.k_min + (config.k_max - config.k_min) * (modes - 1) / (config.M - 1)
        
        # 计算能谱
        E_k = von_karman_pao_tuned(k_m, new_alpha, new_kappa_e, u_prime, kappa_eta)
        dk = np.gradient(k_m)
        u_m = np.sqrt(E_k * dk)
        
        # 生成速度场
        u_field = compute_velocity_field(seed, k_m, u_m, config.M, config.L, config.N)
        
        # 计算实际能量
        energy = 0.5 * np.mean(u_field[..., 0]**2 + u_field[..., 1]**2)
        
        # 保存速度场
        filename = os.path.join(config.dir_fields, f"field_{sample_id:04d}.npy")
        np.save(filename, u_field.astype(np.float64))
        
        # 保存能谱图（每100个样本）
        has_plot = False
        if config.SAVE_SPECTRUM_PLOTS and (sample_id % config.PLOT_INTERVAL == 0):
            save_spectrum_plot(u_field, sample_id, tke, epsilon, new_alpha, 
                             new_kappa_e, energy, k_m, E_k)
            has_plot = True
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': True,
            'sample_id': sample_id,
            'param_id': param_id,
            'seed': seed,
            'file': filename,
            'time': elapsed_time,
            'tke': tke,
            'epsilon': epsilon,
            'energy': energy,
            'alpha': new_alpha,
            'kappa_e': new_kappa_e,
            'u_prime': u_prime,
            'has_plot': has_plot
        }
        
    except Exception as e:
        return {
            'success': False,
            'sample_id': sample_id,
            'param_id': param_id,
            'seed': seed,
            'error': str(e)
        }

# ================================================================
# 能谱图保存
# ================================================================
def save_spectrum_plot(u_field, sample_id, tke, epsilon, alpha, kappa_e, 
                       energy, k_m, E_k_theory):
    """保存能谱对比图"""
    # 计算实际能谱
    knyquist, knorm, kappa_kr, tke_kr = cal_spectrum_iso(u_field, config.L)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 左图：对数坐标
    plt.subplot(1, 2, 1)
    plt.loglog(kappa_kr[1:], tke_kr[1:], 'b-', linewidth=2, label='Numerical', alpha=0.7)
    plt.loglog(k_m, E_k_theory, 'r--', linewidth=2, label='Theoretical (Tuned VKP)')
    plt.axvline(x=kappa_e, color='g', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'κ_e = {kappa_e:.2f}')
    plt.xlabel(r'Wavenumber κ [m$^{-1}$]', fontsize=11)
    plt.ylabel(r'Energy Spectrum E(κ) [m$^3$/s$^2$]', fontsize=11)
    plt.title(f'Sample {sample_id} - Log Scale', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim(config.k_min, knyquist)
    
    # 右图：线性坐标
    plt.subplot(1, 2, 2)
    plt.semilogy(kappa_kr[1:], tke_kr[1:], 'b-', linewidth=2, label='Numerical', alpha=0.7)
    plt.semilogy(k_m, E_k_theory, 'r--', linewidth=2, label='Theoretical (Tuned VKP)')
    plt.axvline(x=kappa_e, color='g', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'κ_e = {kappa_e:.2f}')
    plt.xlabel(r'Wavenumber κ [m$^{-1}$]', fontsize=11)
    plt.ylabel(r'Energy Spectrum E(κ) [m$^3$/s$^2$]', fontsize=11)
    plt.title(f'Sample {sample_id} - Linear Scale', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 添加参数信息
    textstr = '\n'.join((
        f'TKE = {tke:.4f} m²/s²',
        f'ε = {epsilon:.4f} m²/s³',
        f'α = {alpha:.4f}',
        f'κ_e = {kappa_e:.2f} m⁻¹',
        f'Energy = {energy:.4f} m²/s²',
        f'Seed = {sample_id % config.N_SEEDS_PER_PARAM}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.gcf().text(0.98, 0.95, textstr, transform=plt.gcf().transFigure,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=props, fontsize=9)
    
    plt.tight_layout()
    filename = os.path.join(config.dir_spectra, f"spectrum_{sample_id:04d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ================================================================
# 主生成函数
# ================================================================
def generate_samples():
    """生成所有样本"""
    print("🚀 开始生成样本...\n")
    
    # 生成参数组合
    param_combinations = generate_parameter_combinations()
    print(f"✅ 生成了 {len(param_combinations)} 组参数组合")
    print(f"   TKE: {config.N_TKE_SAMPLES} 个值")
    print(f"   Epsilon: {config.N_EPS_SAMPLES} 个值\n")
    
    # 生成所有任务
    tasks = []
    sample_id = 0
    for params in param_combinations:
        for seed_offset in range(config.N_SEEDS_PER_PARAM):
            seed = params['param_id'] * config.N_SEEDS_PER_PARAM + seed_offset
            tasks.append({
                'sample_id': sample_id,
                'param_id': params['param_id'],
                'tke': params['tke'],
                'epsilon': params['epsilon'],
                'seed': seed
            })
            sample_id += 1
    
    print(f"✅ 生成了 {len(tasks)} 个任务\n")
    print("=" * 70)
    
    # 并行处理
    successful_samples = []
    failed_samples = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=config.N_PARALLEL) as executor:
        # 提交所有任务
        futures = [executor.submit(generate_single_sample, task) for task in tasks]
        
        # 处理结果
        for i, future in enumerate(futures):
            result = future.result()
            
            if result['success']:
                successful_samples.append(result)
                plot_msg = "📊" if result.get('has_plot', False) else ""
                print(f"[{i+1:4d}/{len(tasks)}] ✅ Sample {result['sample_id']:4d} | "
                      f"Param {result['param_id']:3d} | Seed {result['seed']:4d} | "
                      f"TKE={result['tke']:.3e} | ε={result['epsilon']:.3e} | "
                      f"Time: {result['time']:.2f}s {plot_msg}")
            else:
                failed_samples.append(result)
                print(f"[{i+1:4d}/{len(tasks)}] ❌ Sample {result['sample_id']:4d} | "
                      f"Error: {result['error']}")
            
            # 每100个样本打印进度
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(tasks) - i - 1)
                print(f"\n📊 进度: {i+1}/{len(tasks)} | "
                      f"剩余时间: {remaining/60:.1f}min\n")
    
    print("\n" + "=" * 70)
    
    # 保存元数据
    save_metadata(successful_samples, failed_samples, param_combinations, start_time)
    
    return successful_samples, failed_samples

# ================================================================
# 元数据保存
# ================================================================
def save_metadata(successful_samples, failed_samples, param_combinations, start_time):
    """保存生成元数据"""
    total_time = time.time() - start_time
    
    metadata = {
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time/3600:.2f} hours",
            'target_samples': config.TOTAL_SAMPLES,
            'successful_samples': len(successful_samples),
            'failed_samples': len(failed_samples)
        },
        'configuration': {
            'N_grid': config.N,
            'L_domain': config.L,
            'M_modes': config.M,
            'nu': config.nu,
            'tke_range': [config.TKE_MIN, config.TKE_MAX],
            'eps_range': [config.EPS_MIN, config.EPS_MAX],
            'n_tke_samples': config.N_TKE_SAMPLES,
            'n_eps_samples': config.N_EPS_SAMPLES,
            'n_seeds_per_param': config.N_SEEDS_PER_PARAM
        },
        'parameter_combinations': param_combinations,
        'successful_samples': successful_samples,
        'failed_samples': failed_samples
    }
    
    # 保存JSON
    json_file = os.path.join(config.dir_metadata, 'generation_metadata.json')
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 保存训练配置
    training_config = {
        'data_info': {
            'num_samples': len(successful_samples),
            'data_dir': config.dir_fields,
            'file_pattern': 'field_*.npy',
            'data_shape': [config.N, config.N, 2],
            'data_type': 'float64'
        },
        'input_parameters': {
            'type': 'TKE_and_epsilon',
            'tke_range': [config.TKE_MIN, config.TKE_MAX],
            'eps_range': [config.EPS_MIN, config.EPS_MAX]
        },
        'tuning_method': 'von_karman_pao_with_literature_tuning_equation'
    }
    
    config_file = os.path.join(config.dir_metadata, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"\n✅ 元数据已保存:")
    print(f"  - {json_file}")
    print(f"  - {config_file}")

# ================================================================
# 生成总结
# ================================================================
def print_summary(successful_samples, failed_samples, start_time):
    """打印生成总结"""
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("生成完成总结")
    print("=" * 70)
    print(f"目标样本数: {config.TOTAL_SAMPLES}")
    print(f"成功生成: {len(successful_samples)} ✅")
    print(f"失败数量: {len(failed_samples)} ❌")
    print(f"成功率: {len(successful_samples)/config.TOTAL_SAMPLES*100:.1f}%")
    print(f"\n总用时: {total_time/3600:.2f} 小时")
    print(f"平均每个样本: {total_time/len(successful_samples):.2f} 秒")
    
    if successful_samples:
        # 能量统计
        energies = [s['energy'] for s in successful_samples]
        tkes = [s['tke'] for s in successful_samples]
        print(f"\n能量统计:")
        print(f"  平均能量: {np.mean(energies):.4e} m²/s²")
        print(f"  标准差: {np.std(energies):.4e} m²/s²")
        print(f"  范围: [{np.min(energies):.4e}, {np.max(energies):.4e}] m²/s²")
        print(f"\nTKE统计:")
        print(f"  范围: [{np.min(tkes):.4e}, {np.max(tkes):.4e}] m²/s²")
        
        # 参数统计
        alphas = [s['alpha'] for s in successful_samples]
        kappa_es = [s['kappa_e'] for s in successful_samples]
        print(f"\n调谐参数统计:")
        print(f"  α 范围: [{np.min(alphas):.4f}, {np.max(alphas):.4f}]")
        print(f"  κ_e 范围: [{np.min(kappa_es):.2f}, {np.max(kappa_es):.2f}] m⁻¹")
    
    print(f"\n数据位置:")
    print(f"  速度场: {config.dir_fields}")
    print(f"  能谱图: {config.dir_spectra}")
    print(f"  元数据: {config.dir_metadata}")
    
    if failed_samples:
        print(f"\n失败样本: {len(failed_samples)}")
        for result in failed_samples[:5]:
            print(f"  Sample {result['sample_id']}: {result['error']}")
        if len(failed_samples) > 5:
            print(f"  ... 还有 {len(failed_samples)-5} 个失败样本")
    
    print("\n" + "=" * 70)

# ================================================================
# 主程序
# ================================================================
if __name__ == "__main__":
    try:
        # 生成样本
        successful_samples, failed_samples = generate_samples()
        
        # 打印总结
        start_time = time.time() - sum(s['time'] for s in successful_samples)
        print_summary(successful_samples, failed_samples, start_time)
        
        print("\n✅ 全部完成！")
        print(f"生成了 {len(successful_samples)} 个样本，存储在: {config.OUTPUT_ROOT}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  生成被用户中断")
        print(f"已生成的样本保存在: {config.OUTPUT_ROOT}")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
