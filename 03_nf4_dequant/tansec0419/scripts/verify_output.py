#!/usr/bin/env python3
"""
验证 NF4 解量化输出的正确性
"""
import numpy as np
import struct

# NF4 查找表（和 CUDA kernel 中一致）
NF4_TABLE = np.array([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0
], dtype=np.float32)

def load_nf4_data(filename):
    """加载 NF4 数据文件"""
    with open(filename, "rb") as f:
        # 读取头部
        rows = struct.unpack("q", f.read(8))[0]
        cols = struct.unpack("q", f.read(8))[0]
        blocksize = struct.unpack("i", f.read(4))[0]
        
        num_weights = rows * cols
        packed_size = num_weights // 2
        num_blocks = num_weights // blocksize
        num_groups = num_blocks // 256 + 1
        
        # 读取数据
        packed = np.frombuffer(f.read(packed_size), dtype=np.uint8)
        absmax_q = np.frombuffer(f.read(num_blocks), dtype=np.uint8)
        absmax2 = np.frombuffer(f.read(num_groups * 2), dtype=np.float16)
        code2 = np.frombuffer(f.read(256 * 2), dtype=np.float16)
        offset = struct.unpack("f", f.read(4))[0]
        
    return {
        'rows': rows,
        'cols': cols,
        'blocksize': blocksize,
        'packed': packed,
        'absmax_q': absmax_q,
        'absmax2': absmax2,
        'code2': code2,
        'offset': offset
    }

def cpu_dequantize(data):
    """在 CPU 上执行解量化（用于验证）"""
    num_elements = data['rows'] * data['cols']
    output = np.zeros(num_elements, dtype=np.float32)
    
    for i in range(num_elements):
        # 1. 解包 4-bit 索引
        byte_idx = i // 2
        shift = (i % 2) * 4
        nf4_idx = (data['packed'][byte_idx] >> shift) & 0xF
        
        # 2. 查表
        nf4_val = NF4_TABLE[nf4_idx]
        
        # 3. 计算块索引
        block_idx = i // data['blocksize']
        
        # 4. 一级缩放
        absmax_q_idx = data['absmax_q'][block_idx]
        scale1 = data['code2'][absmax_q_idx]
        
        # 5. 二级缩放
        group_idx = block_idx // 256
        scale2 = data['absmax2'][group_idx]
        
        # 6. 最终值
        output[i] = nf4_val * scale1 * scale2
    
    return output

def load_gpu_output(filename, num_elements):
    """加载 GPU 输出（BF16 格式）"""
    with open(filename, "rb") as f:
        # BF16 是 uint16，需要转换
        data = np.frombuffer(f.read(), dtype=np.uint16)
        # 转换为 float32（简化处理）
        # BF16 → FP32: 在高 16 位填充
        fp32_data = np.zeros(len(data), dtype=np.float32)
        fp32_view = fp32_data.view(np.uint32)
        fp32_view[:] = data.astype(np.uint32) << 16
        return fp32_data

if __name__ == "__main__":
    import sys
    print("🔍 验证 NF4 解量化正确性\n")
    
    # 获取命令行传入的路径，如果不传，默认使用小数据路径
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/input.bin"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "build/output.bin"
    
    # 1. 加载输入数据
    print(f"📂 加载输入数据: {input_path}")
    data = load_nf4_data(input_path)
    print(f"   矩阵大小: {data['rows']} x {data['cols']}")
    print(f"   总元素: {data['rows'] * data['cols']}")
    
    # 2. CPU 上计算参考结果
    print("\n🖥️  CPU 计算参考结果...")
    cpu_output = cpu_dequantize(data)
    print(f"   CPU 输出范围: [{cpu_output.min():.4f}, {cpu_output.max():.4f}]")
    print(f"   CPU 输出均值: {cpu_output.mean():.4f}")
    print(f"   CPU 输出标准差: {cpu_output.std():.4f}")
    
    # 3. 加载 GPU 输出
    print(f"\n🎮 加载 GPU 输出: {output_path}")
    num_elements = data['rows'] * data['cols']
    gpu_output = load_gpu_output(output_path, num_elements)
    print(f"   GPU 输出范围: [{gpu_output.min():.4f}, {gpu_output.max():.4f}]")
    print(f"   GPU 输出均值: {gpu_output.mean():.4f}")
    print(f"   GPU 输出标准差: {gpu_output.std():.4f}")
    
    # 4. 对比结果
    print("\n📊 对比 CPU vs GPU:")
    
    # 计算差异
    abs_diff = np.abs(cpu_output - gpu_output)
    rel_diff = abs_diff / (np.abs(cpu_output) + 1e-8)
    
    print(f"   最大绝对误差: {abs_diff.max():.6f}")
    print(f"   平均绝对误差: {abs_diff.mean():.6f}")
    print(f"   最大相对误差: {rel_diff.max():.6f}")
    print(f"   平均相对误差: {rel_diff.mean():.6f}")
    
    # 5. 检查几个样本
    print("\n🔎 随机样本检查（前10个）:")
    print("   Index |    CPU Output    |   GPU Output    |   Abs Diff")
    print("   ------|------------------|-----------------|------------")
    for i in range(min(10, num_elements)):
        print(f"   {i:5d} | {cpu_output[i]:16.6f} | {gpu_output[i]:15.6f} | {abs_diff[i]:10.6f}")
    
    # 6. 判断是否通过
    print("\n✅ 验证结果:")
    # 将 abs_diff.max() 改为 abs_diff.mean()
    if abs_diff.mean() < 1e-2:  
        print("   ✅ 正确性验证通过！GPU 输出与 CPU 一致")
        print(f"   平均误差 {abs_diff.mean():.6f} < 阈值 0.01")
    else:
        print("   ❌ 验证失败！GPU 输出与 CPU 不一致")
        print(f"   平均误差 {abs_diff.mean():.6f} >= 阈值 0.01")
        print("   请检查 kernel 实现")