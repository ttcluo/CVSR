import queue as Queue
import threading
import torch
import time
from torch.utils.data import DataLoader


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Reference: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Reference: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher():
    """CUDA prefetcher.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

         # PCIe 监控参数
        self.pcie_monitor = opt.get('pcie_monitor', False)
        self.rx_total = 0  # 接收数据总量 (bytes)
        self.tx_total = 0  # 发送数据总量 (bytes)
        self.start_time = time.time()
        self.batch_counter = 0

        # 预加载第一个batch
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch 是字典

            # 确保在正确的 CUDA 流中操作
            with torch.cuda.stream(self.stream):
                # 处理字典中的所有张量
                for k, v in self.batch.items():
                    if torch.is_tensor(v):
                        # ===== PCIe 优化 1: 确保数据在 CPU 上 =====
                        if v.device.type != 'cpu':
                            v = v.cpu()

                        # ===== PCIe 优化 2: 确保数据是 pinned memory =====
                        if not v.is_pinned():
                            v = v.pin_memory()

                        # ===== PCIe 优化 3: 异步传输到 GPU =====
                        self.batch[k] = v.to(device=self.device, non_blocking=True)

                        # 记录 PCIe 传输量
                        if self.pcie_monitor:
                            self.rx_total += v.element_size() * v.nelement()

            # 增加计数器
            self.batch_counter += 1

        except StopIteration:
            self.batch = None

    def next(self):
        # 等待当前流完成
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch

        # 预取下一个 batch
        self.preload()

        # 打印 PCIe 带宽信息
        if self.pcie_monitor and batch is not None:
            elapsed = time.time() - self.start_time
            if elapsed > 1:  # 至少1秒后报告
                rx_gbps = (self.rx_total / elapsed) / (1024 ** 3)
                tx_gbps = (self.tx_total / elapsed) / (1024 ** 3)

                print(f"[Prefetcher] Batch {self.batch_counter} | "
                      f"PCIe带宽: RX={rx_gbps:.2f} GB/s, TX={tx_gbps:.2f} GB/s | "
                      f"总传输: {self.rx_total/(1024**2):.1f} MB")

                # 重置计数器
                self.rx_total = 0
                self.tx_total = 0
                self.start_time = time.time()

        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.rx_total = 0
        self.tx_total = 0
        self.start_time = time.time()
        self.batch_counter = 0
        self.preload()

        # 打印重置信息
        if self.pcie_monitor:
            print("[Prefetcher] Reset with PCIe monitoring enabled")
