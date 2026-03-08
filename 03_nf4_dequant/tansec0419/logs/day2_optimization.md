# Day 2 优化总结

## V2 优化策略
1. 每线程处理 2 个元素（减少线程数）
2. Packed Store（2×BF16 → 1×uint32）
3. Shared Memory 缓存 NF4_TABLE

## 性能提升
- GPU 时间: 163.97 μs → 114.21 μs (**1.44x**)
- DRAM 带宽: 26.48% → 33.21% (+25.4%)
- 8192² 峰值带宽: 105.4 GB/s

## ncu 指标对比
| 指标 | V1 | V2 | 变化 |
|------|----|----|------|
| gpu__time_duration | 163.97 μs | 114.21 μs | -30.3% ✅ |
| dram__throughput | 26.48% | 33.21% | +25.4% ✅ |
| Grid Size | 65536 | 32768 | -50% |

## 下一步
- [ ] 测试 V3 向量化读取
- [ ] 尝试不同 block size
- [ ] 分析 uncoalesced load
