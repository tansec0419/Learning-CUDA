#!/usr/bin/env python3
"""
生成测试用的 NF4 数据文件
"""
import numpy as np
import struct
import argparse

def generate_nf4_data(rows, cols, blocksize, output_file):
    print(f"生成 NF4 测试数据...")
    print(f"  矩阵大小: {rows} x {cols}")
    print(f"  块大小: {blocksize}")
    
    num_weights = rows * cols
    packed_size = num_weights // 2
    
    # 生成随机数据
    packed = np.random.randint(0, 256, packed_size, dtype=np.uint8)
    
    num_blocks = num_weights // blocksize
    absmax_q = np.random.randint(0, 255, num_blocks, dtype=np.uint8)
    
    num_groups = num_blocks // 256 + 1
    absmax2 = np.random.randn(num_groups).astype(np.float16)
    
    code2 = np.random.randn(256).astype(np.float16)
    
    offset = np.float32(0.0)
    
    # 写入文件
    with open(output_file, "wb") as f:
        # 头部
        f.write(struct.pack("q", rows))
        f.write(struct.pack("q", cols))
        f.write(struct.pack("i", blocksize))
        
        # 数据
        packed.tofile(f)
        absmax_q.tofile(f)
        absmax2.tofile(f)
        code2.tofile(f)
        f.write(struct.pack("f", offset))
    
    print(f"✅ 已保存到 {output_file}")
    print(f"  文件大小: {packed_size + num_blocks + num_groups*2 + 256*2 + 20} 字节")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--blocksize", type=int, default=64)
    parser.add_argument("--output", type=str, default="data/input.bin")
    
    args = parser.parse_args()
    generate_nf4_data(args.rows, args.cols, args.blocksize, args.output)