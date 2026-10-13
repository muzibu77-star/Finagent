"""Multi-process accumulation must reproduce the single-process update."""

import multiprocessing
import os
import socket
import unittest

import torch

from src.training import distributed

SAMPLES_PER_UPDATE = 8


def toy():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    data = [(torch.randn(3), torch.randn(1)) for _ in range(2 * SAMPLES_PER_UPDATE)]
    return model, data


def train(rank: int, world: int) -> list[torch.Tensor]:
    model, data = toy()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position in distributed.local_positions(SAMPLES_PER_UPDATE, rank, world):
            x, y = data[step * SAMPLES_PER_UPDATE + position]
            loss = ((model(x) - y) ** 2).mean()
            (loss / SAMPLES_PER_UPDATE).backward()
            losses.append(float(loss))
        distributed.reduce_scalar(sum(losses), world)
        distributed.sync_gradients(parameters, world)
        optimizer.step()
    distributed.require_identical(parameters, world)
    return [p.detach().clone() for p in parameters]


def worker(rank: int, world: int, port: int, queue) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    assert distributed.setup('gloo') == (rank, world)
    queue.put((rank, [p.tolist() for p in train(rank, world)]))


class DistributedTrainingTests(unittest.TestCase):
    def test_positions_partition_one_update_for_every_world_size(self):
        for world in (1, 2, 4, 8):
            parts = [distributed.local_positions(SAMPLES_PER_UPDATE, r, world) for r in range(world)]
            self.assertEqual(sorted(p for part in parts for p in part), list(range(SAMPLES_PER_UPDATE)))
            self.assertEqual({len(part) for part in parts}, {SAMPLES_PER_UPDATE // world})

    def test_uneven_split_is_rejected_instead_of_changing_the_batch(self):
        with self.assertRaisesRegex(ValueError, 'effective batch would change'):
            distributed.local_positions(SAMPLES_PER_UPDATE, 0, 3)

    def test_single_process_helpers_are_no_ops(self):
        self.assertEqual(distributed.reduce_scalar(2.5, 1), 2.5)
        distributed.sync_gradients([], 1)
        distributed.require_identical([], 1)

    def test_fsdp_units_are_blocks_and_leaf_modules_and_cover_every_frozen_weight(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(2, 2)
                self.adapter = torch.nn.Linear(2, 2, bias=False)

        class Text(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(4, 2)
                self.layers = torch.nn.ModuleList([Layer(), Layer()])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text = Text()
                self.head = torch.nn.Linear(2, 4)
                self.only_adapter = torch.nn.Linear(2, 2, bias=False)

        model = Model()
        adapters = {layer.adapter.weight for layer in model.text.layers} | {model.only_adapter.weight}
        units = distributed.frozen_weight_units(model, adapters)
        self.assertEqual(units, [model.text.embed, *model.text.layers, model.head])

        class Loose(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))
                self.inner = torch.nn.Linear(2, 2)

        loose = Loose()  # a container with its own frozen weight becomes one unit as a whole
        self.assertEqual(distributed.frozen_weight_units(loose, set()), [loose])

    def test_two_processes_match_one_process_and_each_other(self):
        reference = train(0, 1)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        processes = [context.Process(target=worker, args=(rank, 2, port, queue)) for rank in range(2)]
        for process in processes:
            process.start()
        results = dict(queue.get(timeout=120) for _ in processes)
        for process in processes:
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(results[0], results[1])  # processes stay bit-identical
        for expected, actual in zip(reference, results[0]):
            self.assertTrue(torch.allclose(expected, torch.tensor(actual), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
