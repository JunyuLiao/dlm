"""CPU draft guards; never import or execute peer source."""
import copy,unittest
from scripts import v30_peer_reproduction_plan as p

class Draft(unittest.TestCase):
    def test_draft_valid_and_not_launchable(self):
        spec=p.validate_spec(p.draft_spec());self.assertFalse(spec['execution_enabled']);self.assertFalse(spec['formal_launch_enabled']);self.assertFalse(spec['performance_claim_allowed'])
    def test_eight_seeds_multiple_questions(self):
        spec=p.draft_spec();self.assertEqual(len(spec['seeds']),8);self.assertEqual(spec['pilot']['timed_requests'],144);self.assertEqual(spec['full_panel']['timed_requests'],2136)
    def test_no_gate_or_temporal_and_global_only(self):
        method=p.draft_spec()['method'];self.assertIsNone(method['query_sensitivity']);self.assertFalse(method['temporal']);self.assertEqual(method['global_layers'],[5,11,17,23,29]);self.assertEqual(method['local'],'original_dense')
    def test_mutated_draft_fails(self):
        for field in ('seeds','settings','method','thresholds','arms','execution_enabled','peer_source_commit'):
            spec=p.draft_spec();spec[field]=None
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_spec(spec)
    def test_private_material_not_present(self):
        import json
        text=json.dumps(p.draft_spec());self.assertNotIn('/home/',text);self.assertNotIn('E:/',text)
    def test_wrong_peer_pin_rejected_before_git(self):
        with self.assertRaises(ValueError):p.source_inventory('.',commit='a'*40)

if __name__=='__main__':unittest.main()
