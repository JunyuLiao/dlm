from pathlib import Path
import hashlib
import tempfile
import unittest
from scripts.v27_vllm_bind import fingerprint, _method_fields
from scripts.v30_new_config import migrate_source_identity


class Tests(unittest.TestCase):
    def test_reviewed_nested_pins_change_without_changing_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new';rel='experiments/numerical_qk_reuse/v21.py'
            for directory,data in ((old,b'old'),(new,b'new')):
                path=directory/rel;path.parent.mkdir(parents=True);path.write_bytes(data)
            cfg=dict(beta=3,source_hashes={str(old/rel):hashlib.sha256(b'old').hexdigest()})
            cfg['fingerprint']=fingerprint(cfg)
            outer=dict(parent_config=cfg,carry_first=True);outer['fingerprint']=fingerprint(outer)
            result=migrate_source_identity(outer,old,new)
            self.assertEqual(_method_fields(result),_method_fields(outer))
            self.assertNotEqual(result['fingerprint'],outer['fingerprint'])
            self.assertEqual(result['parent_config']['source_hashes'],{str((new/rel).resolve()):hashlib.sha256(b'new').hexdigest()})
            (old/rel).write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError,'Original pinned'):migrate_source_identity(outer,old,new)

    def test_unreviewed_kernel_or_same_basename_elsewhere_rejected(self):
        for rel in ('experiments/numerical_qk_reuse/cached_executor.py','other/v21.py'):
            with tempfile.TemporaryDirectory() as tmp:
                old,new=Path(tmp)/'old',Path(tmp)/'new'
                for directory,data in ((old,b'a'),(new,b'b')):
                    path=directory/rel;path.parent.mkdir(parents=True);path.write_bytes(data)
                cfg=dict(source_hashes={str(old/rel):hashlib.sha256(b'a').hexdigest()});cfg['fingerprint']=fingerprint(cfg)
                with self.assertRaisesRegex(ValueError,'Unreviewed source change'):migrate_source_identity(cfg,old,new)


if __name__=='__main__':unittest.main()
