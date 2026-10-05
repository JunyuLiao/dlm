"""A deployment path rebind cannot legitimize changed method bytes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.v27_vllm_bind import fingerprint, main, rebind_config


def seal(config):
    config['fingerprint'] = fingerprint(config)
    return config


class FrozenConfigRebindTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.old = self.root / 'old_deploy'
        self.new = self.root / 'new_deploy'
        self.old.mkdir()
        self.new.mkdir()
        self.hashes = {}
        for relative, data in [('core/v20.py', b'unchanged v20'), ('core/v21.py', b'unchanged v21')]:
            for root in (self.old, self.new):
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(data)
            self.hashes[relative] = hashlib.sha256(data).hexdigest()
        self.external = self.root / 'external.so'
        self.external.write_bytes(b'unchanged external binary')
        external_hash = hashlib.sha256(self.external.read_bytes()).hexdigest()
        parent = seal(dict(decision_interval=6, threshold=-3.874, carry_first=True,
                           source_hashes={str(self.old/'core/v20.py'): self.hashes['core/v20.py'],
                                          str(self.external): external_hash}))
        nested = seal(dict(variant='test-list-child', source_hashes={
            str(self.old/'core/v20.py'): self.hashes['core/v20.py']}))
        self.config = seal(dict(parent_config=parent, condition='main',
                                plugin='experiments.numerical_qk_reuse.v21:install',
                                values=[nested, {'binary_path': str(self.external), 'shape': [1, 256]}],
                                source_hashes={str(self.old/'core/v21.py'): self.hashes['core/v21.py']}))

    def test_rebinds_all_nested_paths_bottom_up_and_preserves_original(self):
        original = copy.deepcopy(self.config)
        rebound = rebind_config(self.config, self.old, self.new)
        self.assertEqual(self.config, original)
        self.assertEqual(rebound['source_hashes'], {
            str(self.new/'core/v21.py'): self.hashes['core/v21.py']})
        self.assertEqual(rebound['parent_config']['source_hashes'][str(self.new/'core/v20.py')],
                         self.hashes['core/v20.py'])
        self.assertIn(str(self.new/'core/v20.py'), rebound['values'][0]['source_hashes'])
        for node in (rebound, rebound['parent_config'], rebound['values'][0]):
            self.assertEqual(node['fingerprint'], fingerprint(node))
        self.assertNotEqual(rebound['fingerprint'], original['fingerprint'])
        self.assertEqual(rebound['parent_config']['decision_interval'], 6)
        self.assertEqual(rebound['parent_config']['threshold'], -3.874)
        self.assertTrue(rebound['parent_config']['carry_first'])
        self.assertEqual(rebound['values'][1], original['values'][1])
        # Returned nested collections are independent from the original.
        rebound['values'][1]['shape'].append(9)
        self.assertEqual(self.config, original)

    def test_external_sources_remain_verbatim_and_are_verified(self):
        rebound = rebind_config(self.config, self.old, self.new)
        self.assertEqual(rebound['parent_config']['source_hashes'][str(self.external)],
                         self.config['parent_config']['source_hashes'][str(self.external)])
        self.external.write_bytes(b'changed external binary')
        with self.assertRaisesRegex(ValueError, 'external source byte identity drift'):
            rebind_config(self.config, self.old, self.new)

    def test_changed_original_or_rebound_bytes_are_rejected(self):
        original = copy.deepcopy(self.config)
        for root, label in ((self.old, 'original'), (self.new, 'rebound')):
            path = root / 'core/v20.py'
            path.write_bytes(b'changed v20')
            with self.assertRaisesRegex(ValueError, label+' source byte identity drift'):
                rebind_config(self.config, self.old, self.new)
            path.write_bytes(b'unchanged v20')
        self.assertEqual(self.config, original)

    def test_same_changed_bytes_in_both_deploys_do_not_replace_frozen_hash(self):
        for root in (self.old, self.new):
            (root/'core/v20.py').write_bytes(b'identically changed v20')
        with self.assertRaisesRegex(ValueError, 'original source byte identity drift'):
            rebind_config(self.config, self.old, self.new)

    def test_bad_top_or_nested_fingerprint_is_rejected(self):
        for location in ('top', 'parent', 'list'):
            config = copy.deepcopy(self.config)
            if location == 'top':
                config['fingerprint'] = '0'*64
            elif location == 'parent':
                config['parent_config']['fingerprint'] = '0'*64
                seal(config)  # Even a valid outer hash cannot hide the bad child.
            else:
                config['values'][0]['fingerprint'] = '0'*64
                seal(config)
            with self.assertRaisesRegex(ValueError, 'original config fingerprint drift'):
                rebind_config(config, self.old, self.new)

    def test_missing_source_and_invalid_hash_are_rejected(self):
        (self.new/'core/v20.py').unlink()
        with self.assertRaisesRegex(ValueError, 'source is unavailable'):
            rebind_config(self.config, self.old, self.new)
        (self.new/'core/v20.py').write_bytes(b'unchanged v20')
        config = copy.deepcopy(self.config)
        config['source_hashes'][str(self.old/'core/v21.py')] = 'not-a-hash'
        seal(config)
        with self.assertRaisesRegex(ValueError, 'frozen SHA256'):
            rebind_config(config, self.old, self.new)

    def test_shared_prefix_external_directory_is_not_rebound(self):
        sibling = self.root / 'old_deploy_other'
        sibling.mkdir()
        external = sibling / 'x.py'
        external.write_bytes(b'external')
        config = seal(dict(source_hashes={str(external): hashlib.sha256(b'external').hexdigest()}))
        rebound = rebind_config(config, self.old, self.new)
        self.assertEqual(rebound, config)

    def test_source_path_collision_is_rejected(self):
        sources = {str(self.old/'core/v21.py'): self.hashes['core/v21.py'],
                   str(self.new/'core/v21.py'): self.hashes['core/v21.py']}
        config = seal(dict(source_hashes=sources))
        with self.assertRaisesRegex(ValueError, 'source path collision'):
            rebind_config(config, self.old, self.new)

    def test_relative_and_traversal_paths_are_rejected(self):
        for source in ('core/v21.py', str(self.old/'..'/'external.so')):
            config = seal(dict(source_hashes={source: hashlib.sha256(b'external').hexdigest()}))
            with self.assertRaisesRegex(ValueError, 'absolute paths|escapes the old deployment'):
                rebind_config(config, self.old, self.new)

    def test_missing_top_fingerprint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fingerprinted frozen config'):
            rebind_config({}, self.old, self.new)

    def test_canonical_fingerprint_matches_method_guard_json_encoding(self):
        config = dict(text='中文', settings=[0.5, True, None], nested={'x': 3})
        expected = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                             allow_nan=False).encode()).hexdigest()
        self.assertEqual(fingerprint(config), expected)
        config['fingerprint'] = expected
        self.assertEqual(fingerprint(config), expected)

    def test_keyword_roots_and_same_root_binding(self):
        rebound = rebind_config(self.config, old_root=self.old, new_root=self.old)
        self.assertEqual(rebound, self.config)

    def test_cli_writes_new_binding_without_overwriting_frozen_input(self):
        source, output = self.root/'frozen.json', self.root/'bound.json'
        text = json.dumps(self.config)
        source.write_text(text, encoding='utf-8')
        main([str(source), str(self.old), str(self.new), str(output)])
        self.assertEqual(json.loads(output.read_text()), rebind_config(self.config, self.old, self.new))
        self.assertEqual(source.read_text(), text)
        with self.assertRaises(FileExistsError):
            main([str(source), str(self.old), str(self.new), str(source)])
        self.assertEqual(source.read_text(), text)


if __name__ == '__main__':
    unittest.main()
