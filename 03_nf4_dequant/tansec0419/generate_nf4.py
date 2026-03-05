import numpy as np
import struct

rows = 4096
cols = 4096
blocksize = 64

packed_size = rows * cols // 2
packed = np.random.randint(0,256,packed_size,dtype=np.uint8)

num_blocks = (rows * cols) // blocksize
absmax_q = np.random.randint(0,255,num_blocks,dtype=np.uint8)

num_groups = num_blocks // 256 + 1
absmax2 = np.random.randn(num_groups).astype(np.float16)

code2 = np.random.randn(256).astype(np.float16)

offset = np.float32(0.0)

with open("input.bin","wb") as f:
    
    f.write(struct.pack("q",rows))
    f.write(struct.pack("q",cols))
    f.write(struct.pack("i",blocksize))
    
    packed.tofile(f)
    absmax_q.tofile(f)
    absmax2.tofile(f)
    code2.tofile(f)
    
    f.write(struct.pack("f",offset))