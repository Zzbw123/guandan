"""One-pass FIFO teacher trainer; only visible observations enter update()."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import re
import torch
from guandan.learning.model import DMCNetwork
from experiments.p5f_protocol import config


def configure_runtime(seed):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TeacherTrainer:
    def __init__(self, configuration, corpus_sha256):
        if (type(configuration) is not dict or
                configuration != config(configuration.get('seed'), configuration.get('objective'))):
            raise ValueError('P5f configuration mismatch')
        if type(corpus_sha256) is not str or re.fullmatch('[0-9a-f]{64}', corpus_sha256) is None:
            raise ValueError('pinned corpus manifest digest required')
        configure_runtime(configuration['seed'])
        self.config = dict(configuration)
        self.corpus_sha256 = corpus_sha256
        self.model = DMCNetwork().to('cuda', dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'],
                                          eps=self.config['adam_eps'], capturable=True)
        self.updates = self.samples = self.candidates = 0
        self.ready = True

    def update(self, requests):
        if not self.ready:
            raise RuntimeError('failed/incomplete update; reload a successful checkpoint')
        if type(requests) is not list or not 1 <= len(requests) <= self.config['batch_size']:
            raise ValueError('batch size outside fixed bounds')
        self.ready = False
        from experiments.p5f_objective import fit_batch
        result = fit_batch(self.model, self.optimizer, requests,
                           objective=self.config['objective'], chunk_size=self.config['chunk_size'])
        self.updates += 1
        start = self.samples
        self.samples += len(requests)
        self.candidates += result['scored_candidates']
        self.ready = True
        return dict(result, update=self.updates, observation_start=start,
                    observation_stop=self.samples, total_candidates=self.candidates)
