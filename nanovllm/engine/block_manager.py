from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 用于构建链式hash值
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # 链式hash实现
    # 基于token_ids和prefix生成了一个hash值
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 非加密hash，适合缓存索引
        h = xxhash.xxh64()
        # 前缀不为空，则前一个块prefix作为8字节数据放入
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 判断 hash 条目是否还指向自身，防止误删已覆盖
        # 啥情况下会覆盖？
        # 比如同样开头的两个句子“你好世界”，“你好小明”
        # 假设被分为[“你好”，“世界”]、[“你好”，“小明”]
        # 两个“你好”会被映射到相同的位置，比如先映射了A的你好，那么B的你好在缓存的时候就把它覆盖掉
        # 现在假设A的物理块已经回收了，要分配A的你好所在的物理块
        # 此时这里就会判断这个你好是否是A的你好，防止误删B缓存的你好，使B的整个前缀缓存崩溃
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    # 回收已分配的物理块，移出已使用队列，移入空闲队列
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 探测是否有足够的空闲块 同时做缓存探测
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 哈希表中没有这个内容或哈希命中但发生碰撞，链式哈希查找失败！
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 缓存块正在被使用，才能不消耗空闲块
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 如果空闲块不够分配，拒绝调度该序列
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        # 足够分配，则返回已缓存的块数
        return num_cached_blocks

    # seq 需分配物理块的序列
    # num_cached_blocks 前缀缓存命中的块
    # 分为两部分处理，一部分是前缀缓存命中的块，另一部分是未命中的块
    # 采用链式构造前缀缓存
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            # 最巧妙的机制，计算hash值的时候，带上前一个块hash计算
            # 从而保证当前构建的hash是基于完整上下文信息的hash，而非单独块hash
            h = self.compute_hash(token_ids, h)
            # 如果缓存命中的块，走hash查找路径，不走空闲队列的FIFO
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            # 引用计数 如果物理块正在被使用，引用计数加一
            # 如果没有使用，把物理块移除空闲队列，放入使用队列
            # 最后把引用的物理块信息放入当前seq的信息中
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 处理没有缓存命中的块 FIFO
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 缓存的token数：缓存命中物理块数乘物理块size
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 逆序遍历，从最后一个块开始释放
    # 前面的块是共享的前缀，而后面的块更可能是新加入的部分
    # 逆序会先释放后面的块
    # block的前缀hash是链式依赖不影响hash悬空，因为释放部分不涉及hash
    # hash的清理是在reallocate的时候
    # deallocate是一个不可分割的同步操作，释放的顺序并不影响物理块的提前释放
    # 但是会影响物理块在空闲队列的顺序
    # 前缀块通常保留有更多共享，更容易被复用
    # 如果排在后面，存活时间更长，更有机会在还没被清除hash时命中
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 若引用计数为0，释放该物理块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
x
    def can_append(self, seq: Sequence) -> bool:
        # 巧妙的布尔表达式
        # 结合seq超出blocksize的部分是否为刚好为1
        # 每次新来一个token时，都会调用检测一次是否需要新的物理块
        # 比如从0增长到1时，后面的表达式为true，则需要分配新的物理块
        # 此时还需检测空闲块是否大于等于一，刚好符合
        # 若为其他值，由于此时物理块已分配，所以不用分配新的，只要检测是否大于等于0
        # 由于空闲块数非负，必通过
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 实际的分配方法，而非检测
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 完成推理的 token 所在物理块写入 hash 使得其他序列可以完成前缀缓存查找
    def hash_blocks(self, seq: Sequence):
        # 计算本序列的起始和结束块索引
        # 若没有跨过新的块，则结束，无需写入hash
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 取前一个块的hash
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        # 遍历需要建立hash的块，链式建立hash
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
