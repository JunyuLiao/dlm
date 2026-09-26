import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.v18_finalize import inspect_archive


class FinalizeTests(unittest.TestCase):
    def test_archive_accepts_only_private_dataset_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'private.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                item = tarfile.TarInfo('ruler/cells/cell/attempt00.json')
                item.size = 2
                archive.addfile(item, io.BytesIO(b'{}'))
            self.assertEqual(inspect_archive(path), 1)
            with tarfile.open(path, 'w:gz') as archive:
                item = tarfile.TarInfo('ruler/../outside')
                item.size = 1
                archive.addfile(item, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'unexpected entry'):
                inspect_archive(path)


if __name__ == '__main__':
    unittest.main()
