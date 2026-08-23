"""
白噪声场生成程序 - 与湍流样本一一对应
生成1000个白噪声场，每个对应一个湍流样本的随机种子

对应关系：
- 湍流样本: TurbulenceData_1000_Tuned/velocity_fields/field_XXXX.npy
- 白噪声场: WhiteNoiseData_1000/noise_fields/noise_XXXX.npy
- 使用相同的seed确保配对关系

白噪声场特性：
- 高斯分布：均值=0，标准差=1
- 空间相关性：δ(x-x') (空间白噪声)
- 维度：256 × 256 (匹配湍流场的网格)
"""

import numpy as np
import os
import json
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor

# ================================================================
# 配置参数
# ================================================================
class NoiseConfig:
    """白噪声生成配置"""
    # 生成参数
    N_TKE_SAMPLES = 10          # TKE参数样本数
    N_EPS_SAMPLES = 10          # epsilon参数样本数
    N_SEEDS_PER_PARAM = 10      # 每组参数的随机种子数
    TOTAL_SAMPLES = N_TKE_SAMPLES * N_EPS_SAMPLES * N_SEEDS_PER_PARAM  # 1000
    
    # 网格参数（与湍流场匹配）
    N = 256                     # 网格点数
    
    # 白噪声参数
    NOISE_MEAN = 0.0            # 均值
    NOISE_STD = 1.0             # 标准差
    
    # 并行参数
    N_PARALLEL = 20              # 并行处理数
    
    # 输出设置
    OUTPUT_ROOT = "WhiteNoiseData_1000"
    
    def __init__(self):
        # 创建输出目录结构
        self.dir_noise = os.path.join(self.OUTPUT_ROOT, "noise_fields")
        self.dir_metadata = os.path.join(self.OUTPUT_ROOT, "metadata")
        
        for directory in [self.dir_noise, self.dir_metadata]:
            os.makedirs(directory, exist_ok=True)
        
        print("=" * 70)
        print("白噪声场生成配置")
        print("=" * 70)
        print(f"TKE参数数: {self.N_TKE_SAMPLES}")
        print(f"Epsilon参数数: {self.N_EPS_SAMPLES}")
        print(f"每组参数种子数: {self.N_SEEDS_PER_PARAM}")
        print(f"目标噪声场总数: {self.TOTAL_SAMPLES}")
        print(f"网格尺寸: {self.N} × {self.N}")
        print(f"噪声分布: N({self.NOISE_MEAN}, {self.NOISE_STD}²)")
        print(f"并行数: {self.N_PARALLEL}")
        print(f"输出目录: {self.OUTPUT_ROOT}")
        print("=" * 70 + "\n")

config = NoiseConfig()

# ================================================================
# 白噪声场生成函数
# ================================================================
def generate_white_noise(seed, shape, mean=0.0, std=1.0):
    """
    生成高斯白噪声场
    
    参数:
        seed: 随机种子
        shape: 噪声场形状 (N, N)
        mean: 均值
        std: 标准差
    
    返回:
        noise: 白噪声场 [N, N]
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(mean, std, shape)
    return noise

# ================================================================
# 单个样本生成函数
# ================================================================
def generate_single_noise(task):
    """
    生成单个白噪声场
    
    参数:
        task: 包含 sample_id, param_id, seed 的字典
    
    返回:
        result: 生成结果字典
    """
    sample_id = task['sample_id']
    param_id = task['param_id']
    seed = task['seed']
    
    start_time = time.time()
    
    try:
        # 生成白噪声场
        noise_field = generate_white_noise(
            seed=seed,
            shape=(config.N, config.N),
            mean=config.NOISE_MEAN,
            std=config.NOISE_STD
        )
        
        # 保存噪声场
        filename = os.path.join(config.dir_noise, f"noise_{sample_id:04d}.npy")
        np.save(filename, noise_field.astype(np.float64))
        
        # 计算统计信息
        noise_mean = float(np.mean(noise_field))
        noise_std = float(np.std(noise_field))
        noise_min = float(np.min(noise_field))
        noise_max = float(np.max(noise_field))
        
        elapsed_time = time.time() - start_time
        
        result = {
            'success': True,
            'sample_id': sample_id,
            'param_id': param_id,
            'seed': seed,
            'filename': filename,
            'statistics': {
                'mean': noise_mean,
                'std': noise_std,
                'min': noise_min,
                'max': noise_max
            },
            'time': elapsed_time
        }
        
        return result
        
    except Exception as e:
        result = {
            'success': False,
            'sample_id': sample_id,
            'param_id': param_id,
            'seed': seed,
            'error': str(e),
            'time': time.time() - start_time
        }
        return result

# ================================================================
# 生成任务列表
# ================================================================
def generate_task_list():
    """
    生成与湍流样本对应的任务列表
    
    返回:
        tasks: 任务列表
    """
    tasks = []
    sample_id = 0
    
    # 遍历所有参数组合
    for param_id in range(config.N_TKE_SAMPLES * config.N_EPS_SAMPLES):
        for seed_offset in range(config.N_SEEDS_PER_PARAM):
            # 使用与湍流生成脚本完全相同的seed计算方式
            seed = param_id * config.N_SEEDS_PER_PARAM + seed_offset
            
            tasks.append({
                'sample_id': sample_id,
                'param_id': param_id,
                'seed': seed
            })
            sample_id += 1
    
    return tasks

# ================================================================
# 主生成函数
# ================================================================
def generate_all_noise():
    """生成所有白噪声场"""
    print("🚀 开始生成白噪声场...\n")
    
    # 生成任务列表
    tasks = generate_task_list()
    print(f"✅ 生成了 {len(tasks)} 个任务\n")
    print("=" * 70)
    
    # 并行处理
    successful_samples = []
    failed_samples = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=config.N_PARALLEL) as executor:
        # 提交所有任务
        futures = [executor.submit(generate_single_noise, task) for task in tasks]
        
        # 处理结果
        for i, future in enumerate(futures):
            result = future.result()
            
            if result['success']:
                successful_samples.append(result)
                print(f"[{i+1:4d}/{len(tasks)}] ✅ Noise {result['sample_id']:4d} | "
                      f"Param {result['param_id']:3d} | Seed {result['seed']:4d} | "
                      f"Mean={result['statistics']['mean']:+.6f} | "
                      f"Std={result['statistics']['std']:.6f} | "
                      f"Time: {result['time']:.3f}s")
            else:
                failed_samples.append(result)
                print(f"[{i+1:4d}/{len(tasks)}] ❌ Noise {result['sample_id']:4d} | "
                      f"Error: {result['error']}")
            
            # 每100个样本打印进度
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(tasks) - i - 1)
                print(f"\n📊 进度: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.1f}%) | "
                      f"已用时间: {elapsed/60:.1f}min | "
                      f"剩余时间: {remaining/60:.1f}min\n")
    
    print("\n" + "=" * 70)
    
    # 保存元数据
    save_metadata(successful_samples, failed_samples, start_time)
    
    return successful_samples, failed_samples

# ================================================================
# 元数据保存
# ================================================================
def save_metadata(successful_samples, failed_samples, start_time):
    """保存生成元数据"""
    total_time = time.time() - start_time
    
    # 计算统计信息
    if successful_samples:
        all_means = [s['statistics']['mean'] for s in successful_samples]
        all_stds = [s['statistics']['std'] for s in successful_samples]
        all_mins = [s['statistics']['min'] for s in successful_samples]
        all_maxs = [s['statistics']['max'] for s in successful_samples]
        
        statistics_summary = {
            'mean': {
                'average': float(np.mean(all_means)),
                'std': float(np.std(all_means)),
                'min': float(np.min(all_means)),
                'max': float(np.max(all_means))
            },
            'std': {
                'average': float(np.mean(all_stds)),
                'std': float(np.std(all_stds)),
                'min': float(np.min(all_stds)),
                'max': float(np.max(all_stds))
            },
            'range': {
                'min': float(np.min(all_mins)),
                'max': float(np.max(all_maxs))
            }
        }
    else:
        statistics_summary = {}
    
    metadata = {
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time/60:.2f} minutes",
            'target_samples': config.TOTAL_SAMPLES,
            'successful_samples': len(successful_samples),
            'failed_samples': len(failed_samples)
        },
        'configuration': {
            'N_grid': config.N,
            'noise_distribution': {
                'type': 'Gaussian',
                'mean': config.NOISE_MEAN,
                'std': config.NOISE_STD
            },
            'n_tke_samples': config.N_TKE_SAMPLES,
            'n_eps_samples': config.N_EPS_SAMPLES,
            'n_seeds_per_param': config.N_SEEDS_PER_PARAM
        },
        'statistics_summary': statistics_summary,
        'correspondence': {
            'description': 'Each noise field corresponds to a turbulence field',
            'naming_convention': 'noise_XXXX.npy corresponds to field_XXXX.npy',
            'seed_calculation': 'seed = param_id * N_SEEDS_PER_PARAM + seed_offset'
        },
        'successful_samples': successful_samples,
        'failed_samples': failed_samples
    }
    
    # 保存JSON
    json_file = os.path.join(config.dir_metadata, 'noise_metadata.json')
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 保存简化的映射文件
    mapping = []
    for sample in successful_samples:
        mapping.append({
            'sample_id': sample['sample_id'],
            'param_id': sample['param_id'],
            'seed': sample['seed'],
            'noise_file': f"noise_{sample['sample_id']:04d}.npy",
            'turbulence_file': f"field_{sample['sample_id']:04d}.npy"
        })
    
    mapping_file = os.path.join(config.dir_metadata, 'noise_turbulence_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✅ 元数据已保存:")
    print(f"  - {json_file}")
    print(f"  - {mapping_file}")

# ================================================================
# 生成总结
# ================================================================
def print_summary(successful_samples, failed_samples, start_time):
    """打印生成总结"""
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("白噪声场生成完成总结")
    print("=" * 70)
    print(f"目标样本数: {config.TOTAL_SAMPLES}")
    print(f"成功生成: {len(successful_samples)} ✅")
    print(f"失败数量: {len(failed_samples)} ❌")
    print(f"成功率: {len(successful_samples)/config.TOTAL_SAMPLES*100:.1f}%")
    print(f"\n总用时: {total_time/60:.2f} 分钟")
    print(f"平均每个样本: {total_time/len(successful_samples):.3f} 秒")
    
    if successful_samples:
        # 统计信息
        all_means = [s['statistics']['mean'] for s in successful_samples]
        all_stds = [s['statistics']['std'] for s in successful_samples]
        
        print(f"\n白噪声场统计:")
        print(f"  均值 - 平均: {np.mean(all_means):.6f}, 标准差: {np.std(all_means):.6f}")
        print(f"  标准差 - 平均: {np.mean(all_stds):.6f}, 标准差: {np.std(all_stds):.6f}")
        print(f"  理论期望: 均值=0.0, 标准差=1.0")
    
    print(f"\n数据位置:")
    print(f"  白噪声场: {config.dir_noise}")
    print(f"  元数据: {config.dir_metadata}")
    
    print(f"\n对应关系:")
    print(f"  noise_XXXX.npy ↔ field_XXXX.npy")
    print(f"  (相同的XXXX表示配对样本)")
    
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
        # 生成白噪声场
        successful_samples, failed_samples = generate_all_noise()
        
        # 打印总结
        start_time = time.time() - sum(s['time'] for s in successful_samples)
        print_summary(successful_samples, failed_samples, start_time)
        
        print("\n✅ 白噪声场生成完成！")
        print(f"生成了 {len(successful_samples)} 个噪声场，存储在: {config.OUTPUT_ROOT}")
        print(f"\n💡 使用说明:")
        print(f"  - 每个 noise_XXXX.npy 对应湍流样本 field_XXXX.npy")
        print(f"  - 使用相同的随机种子确保配对关系")
        print(f"  - 可用于FNO训练：输入=噪声场+条件，输出=湍流场")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  生成被用户中断")
        print(f"已生成的噪声场保存在: {config.OUTPUT_ROOT}")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
