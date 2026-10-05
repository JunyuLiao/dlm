import unittest
from scripts.v30_export_paused_records import numeric,sanitize


class ExportTests(unittest.TestCase):
    def row(self):
        return dict(arm='method',deploy_commit='a'*40,wall_s=3.,decode_span_s=2.,denoise_forward_count=4,
            prompt='PRIVATE',prompt_hash='PRIVATE_HASH',gold='PRIVATE',output='PRIVATE',index=12345,
            receipts=dict(adapter=dict(begins=4,private_path='PRIVATE'),method=dict(
                effective_method=dict(score_period=64,consumer='fa4',risk_state='dense_prefix'),
                attention_calls=20,support_build=dict(path='PRIVATE',fingerprint='PRIVATE_HASH'))))
    def test_private_strings_indices_and_hashes_are_not_exported(self):
        import json
        out=sanitize(self.row(),'aime','b0_method',0,'q000')
        self.assertNotIn('PRIVATE',json.dumps(out));self.assertNotIn('12345',json.dumps(out))
        self.assertEqual(out['S_per_N_s'],.5)
        self.assertEqual(out['effective_method']['consumer'],'fa4')
        self.assertEqual(out['receipt_numeric']['method']['attention_calls'],20)
    def test_path_like_enum_and_unrecognized_source_fail(self):
        row=self.row();row['receipts']['method']['effective_method']['consumer']='/private/path'
        with self.assertRaises(ValueError):sanitize(row,'aime','b0_method',0,'q000')
        row=self.row();row['deploy_commit']='PRIVATE'
        with self.assertRaises(ValueError):sanitize(row,'aime','b0_method',0,'q000')
    def test_numeric_nested_fields_reject_nonfinite(self):
        self.assertEqual(numeric(dict(capture=0,prompt_hash=42,secret=[1,2])),dict(capture=0))
        self.assertEqual(numeric([dict(calls=3,path='PRIVATE'),'PRIVATE']),[dict(calls=3)])
        with self.assertRaises(ValueError):numeric(float('nan'))


if __name__=='__main__':unittest.main()
