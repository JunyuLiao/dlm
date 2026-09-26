import json
import tempfile
import unittest
from pathlib import Path

from scripts.v18_bridge import ARMS, compare_bridge, inventory_identity
from scripts.v18_protocol import sha


class BridgeTests(unittest.TestCase):
    def test_inventory_requires_full_tensor_set(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'inventory.json'
            path.write_text(json.dumps({'tensors': {'one': {'sha256': '0' * 64}}}))
            with self.assertRaisesRegex(ValueError, '1047'):
                inventory_identity(path)

    def test_private_bridge_compare_and_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tensors = {f't{i}': {'shape': [1], 'dtype': 'BF16', 'sha256': f'{i:064x}'} for i in range(1047)}
            protocols, ledgers, privates, inventories = [], [], [], []
            for side in range(2):
                inventory_path = root / f'inventory{side}.json'
                inventory_path.write_text(json.dumps({'tensors': tensors,
                                                      'index_sha256': ('a' if side == 0 else 'b') * 64}))
                inventory = inventory_identity(inventory_path)
                inventories.append(inventory_path)
                private = root / f'private{side}'
                privates.append(private)
                schedule, events = [], []
                for arm in ARMS:
                    for rid in ('q0', 'q1'):
                        cell = arm + rid
                        schedule.append({'arm': arm, 'id': rid, 'seed': 101, 'role': 'attempt0',
                                         'repeat': 0, 'cell_id': cell})
                        receipt = {'id': rid, 'seed': 101, 'completion_tokens': [1, 2],
                                   'prompt_token_hash': 'prompt', 'termination_reason': 'eos',
                                   'total_decoder_calls': 2,
                                   'per_canvas': [{'decoder_calls': 2, 'schedule_steps': [48, 47],
                                                   'native_stop_final_call': True,
                                                   'iteration_cap_final_call': False}]}
                        path = private / 'cells' / cell / 'attempt00.json'
                        path.parent.mkdir(parents=True)
                        path.write_text(json.dumps(receipt))
                        events.append({'event': 'run', 'arm': arm, 'id': rid, 'seed': 101,
                                       'role': 'attempt0', 'repeat': 0, 'cell_id': cell,
                                       'ok': True, 'private_receipt': '/original/' + cell + '/attempt00.json',
                                       'completion_token_hash': sha('[1,2]'), 'termination': 'eos',
                                       'per_canvas_calls': [2]})
                proto = {'host': f'host{side}', 'gpu_uuid': f'gpu{side}', 'ids': ['q0', 'q1'],
                         'seeds': [101], 'arms': list(ARMS), 'manifest_sha256': 'b' * 64,
                         'calibration_sha256': 'c' * 64, 'policy_file_sha256': 'd' * 64,
                         'binary_sha256': {'kernel': 'e' * 64, 'bridge': 'f' * 64},
                         'bridge_source_sha256': '0' * 64, 'math_identity': {},
                         'nonindex_model_metadata_hashes': {},
                         'source_hash_values': {}, 'tensor_inventory': inventory,
                         'schedule': schedule}
                p, l = root / f'protocol{side}.json', root / f'ledger{side}.jsonl'
                p.write_text(json.dumps(proto))
                l.write_text('\n'.join(json.dumps(e) for e in events) + '\n')
                protocols.append(p)
                ledgers.append(l)
            kwargs = dict(private_a=privates[0], private_b=privates[1],
                          inventory_a=inventories[0], inventory_b=inventories[1])
            passed = compare_bridge(protocols[0], ledgers[0], protocols[1], ledgers[1], **kwargs)
            self.assertTrue(passed['qualified'])
            self.assertTrue(passed['repacked_index_differs'])
            missing = privates[1] / 'cells' / 'U50q0' / 'attempt00.json'
            original = json.loads(missing.read_text())
            malformed = json.loads(missing.read_text())
            del malformed['per_canvas'][0]['native_stop_final_call']
            missing.write_text(json.dumps(malformed))
            rejected = compare_bridge(protocols[0], ledgers[0], protocols[1], ledgers[1], **kwargs)
            self.assertFalse(rejected['qualified'])
            self.assertIn('missing_trajectory_evidence', rejected['issues'])
            missing.write_text(json.dumps(original))
            tampered = privates[1] / 'cells' / 'U50q0' / 'attempt00.json'
            row = json.loads(tampered.read_text())
            row['completion_tokens'] = [9]
            tampered.write_text(json.dumps(row))
            result = compare_bridge(protocols[0], ledgers[0], protocols[1], ledgers[1], **kwargs)
            self.assertFalse(result['qualified'])
            self.assertIn('trajectory_mismatch', result['issues'])


if __name__ == '__main__':
    unittest.main()
