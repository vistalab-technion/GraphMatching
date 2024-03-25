""" Dataset partitioning helper """
from random import Random
import torch.distributed as dist


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        left_out_elements_amount = len(indexes)
        if left_out_elements_amount > 0:
            # we ensure equal partitions, to avoid a scenario in which workers have different number of batches,
            # resulting in deadlocks
            print(f"Data partitioner leaves out {left_out_elements_amount} elements of the data. "
                  "To make better usage of the data, try to adjust number of workers or the length of the data")

    def use(self, rank):
        return Partition(self.data, self.partitions[rank])

    def get_partition_data(self, rank):
        partition = self.use(rank)
        return [elem for elem in partition]


def split_dataset(dataset, batch_size, size):
    bsz = int(batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    return [partition.get_partition_data(rank) for rank in range(size)], bsz

def partition_dataset(dataset, batch_size, rank=None, size=None):
    if rank is None:
        rank = dist.get_rank()
        size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(rank)
    return partition, bsz


def average_gradients(model):
    size = float(dist.get_world_size())
    if size == 1:
        return
    for i, param in enumerate(model.parameters()):
        try:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        except AttributeError:
            # print(f"No grad detected for param with index={i}")
            continue
        param.grad.data /= size
