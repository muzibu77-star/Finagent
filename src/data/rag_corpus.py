"""Document RAG exposing the same scoped read/search contract as ReportCorpus."""

import json
from pathlib import Path

from src.data.report_corpus import ReportCorpus
from src.retrieval.neural import NeuralIndex


class RagCorpus(ReportCorpus):
    def __init__(self, path: str, snapshot_id: str, index_dir: Path,
                 mode: str = 'cross_encoder'):
        # Reuse all original snapshot, publication and replacement validation.
        super().__init__(path, snapshot_id)
        self.index.close()
        config = json.loads(Path('configs/jd_retrieval.json').read_text())
        # The generation worker owns the GPU. Online retrieval is explicitly CPU.
        self.index = NeuralIndex(self.documents, index_dir, config, device='cpu')
        self.mode = mode
