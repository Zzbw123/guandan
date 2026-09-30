import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.p5f_data import build_corpus, iter_batches, iter_hands, verify_corpus
from guandan.agents import GreedyAgent
from guandan.env.hand_env import HandEnv
from guandan.types import PlayerObservation


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


class CorpusTests(unittest.TestCase):
    DEALS = [(100000, 2, 0), (100001, 5, 2)]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "corpus"

    def build(self):
        return build_corpus(self.directory, self.DEALS)

    def rewrite(self, rows, manifest):
        raw = b"".join(canonical(row) + b"\n" for row in rows)
        with (self.directory / "replays.jsonl.gz").open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as zipped:
                zipped.write(raw)
        manifest["replays_sha256"] = hashlib.sha256(
            (self.directory / "replays.jsonl.gz").read_bytes()).hexdigest()
        manifest["logical_sha256"] = hashlib.sha256(raw).hexdigest()
        (self.directory / "manifest.json").write_bytes(canonical(manifest) + b"\n")

    def test_replay_requests_fifo_remainder_and_resume(self):
        manifest = self.build()
        self.assertEqual(manifest, verify_corpus(self.directory))
        hands = list(iter_hands(self.directory))
        self.assertEqual(2, len(hands))
        requests = []
        for metadata, hand_requests in hands:
            self.assertNotIn("replay", metadata)
            self.assertEqual(metadata["steps"], len(hand_requests))
            self.assertEqual(metadata["legal_candidates"],
                             sum(len(legal) for _, legal in hand_requests))
            for obs, legal in hand_requests:
                self.assertIsInstance(obs, PlayerObservation)
                self.assertIsInstance(legal, list)
                self.assertGreater(len(legal), 0)
                self.assertIn(GreedyAgent().act(obs, legal), legal)
            requests.extend(hand_requests)
        self.assertEqual(manifest["observations"], len(requests))
        self.assertEqual(manifest["candidates"], sum(len(legal) for _, legal in requests))
        batch_size = max(2, len(requests) // 3)
        if len(requests) % batch_size == 0:
            batch_size += 1
        batches = list(iter_batches(self.directory, batch_size))
        self.assertEqual(requests, [request for batch in batches for request in batch])
        self.assertEqual(requests[batch_size:],
                         [request for batch in iter_batches(self.directory, batch_size, 1)
                          for request in batch])
        self.assertLess(len(batches[-1]), batch_size)
        self.assertEqual([], list(iter_batches(self.directory, batch_size, len(batches))))
        with self.assertRaisesRegex(ValueError, "start_batch"):
            list(iter_batches(self.directory, batch_size, len(batches) + 1))

    def test_exclusive_and_invalid_deals(self):
        for deals in ([(100000, 2, 0), (100000, 3, 1)],
                      [(99999, 2, 0)], [(100000, 1, 0)], [(100000, 2, 4)]):
            with self.subTest(deals=deals), self.assertRaises(ValueError):
                build_corpus(self.directory, deals)
            self.assertFalse(self.directory.exists())
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()

    def test_repeated_build_is_byte_deterministic(self):
        first = self.build()
        other = Path(self.temp.name) / "other"
        second = build_corpus(other, self.DEALS)
        self.assertEqual(first, second)
        self.assertEqual((self.directory / "replays.jsonl.gz").read_bytes(),
                         (other / "replays.jsonl.gz").read_bytes())

    def test_file_manifest_and_teacher_tampering(self):
        manifest = self.build()
        path = self.directory / "replays.jsonl.gz"
        original = path.read_bytes()
        path.write_bytes(original + b"x")
        with self.assertRaisesRegex(ValueError, "file hash"):
            verify_corpus(self.directory)
        path.write_bytes(original)
        manifest_path = self.directory / "manifest.json"
        manifest["observations"] += 1
        manifest_path.write_bytes(canonical(manifest) + b"\n")
        with self.assertRaises(ValueError):
            verify_corpus(self.directory)
        manifest["observations"] -= 1
        manifest_path.write_bytes(canonical(manifest) + b"\n")
        with gzip.open(path, "rb") as source:
            rows = [json.loads(line) for line in source]
        # A modified legal teacher choice must fail even if both hashes are repaired.
        steps = rows[0]["replay"]["steps"]
        self.assertGreater(len(steps), 0)
        steps[0]["action"]["kind"] = "pass"
        self.rewrite(rows, manifest)
        verify_corpus(self.directory)
        with self.assertRaisesRegex(ValueError, "greedy trajectory"):
            list(iter_hands(self.directory))

    def test_rehashed_tail_truncation_is_rejected(self):
        manifest = self.build()
        path = self.directory / "replays.jsonl.gz"
        with gzip.open(path, "rb") as source:
            rows = [json.loads(line) for line in source]
        self.rewrite(rows[:-1], manifest)
        with self.assertRaisesRegex(ValueError, "tail count"):
            verify_corpus(self.directory)
        with self.assertRaisesRegex(ValueError, "tail count"):
            list(iter_batches(self.directory, 64))


if __name__ == "__main__":
    unittest.main()
