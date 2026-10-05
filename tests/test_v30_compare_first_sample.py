import copy
import unittest
from scripts.v30_compare_first_sample import compare_request


def request():
    names=('canvas','step_tensor','is_encoder_phase','sc_embeds','history_len_tensor','decode_slots','logits','argmax_canvas','confident_tensor','history')
    values=dict(rng='PRIVATE_RNG',tensors={n:dict(sha256='PRIVATE_'+n,shape=[1],dtype='x') for n in names})
    return dict(request_ordinal=0,init=[dict(before=copy.deepcopy(values),after=copy.deepcopy(values))],
                sample=[dict(before=copy.deepcopy(values),after=copy.deepcopy(values))])


class Tests(unittest.TestCase):
    def test_export_is_boolean_and_does_not_leak_private_hashes(self):
        left=request();right=copy.deepcopy(left)
        self.assertIsNone(compare_request(left,right)['first_observed_difference'])
        right['sample'][0]['before']['tensors']['logits']['sha256']='different'
        result=compare_request(left,right)
        self.assertEqual(result['first_observed_difference'],'sampler_input_logits_equal')
        self.assertNotIn('PRIVATE',str(result));self.assertNotIn('different',str(result))

    def test_ignored_uninitialized_storage_is_not_declared_cause(self):
        left=request();right=copy.deepcopy(left)
        for n in ('history','argmax_canvas','confident_tensor'):
            right['sample'][0]['before']['tensors'][n]['sha256']='different'
        self.assertIsNone(compare_request(left,right)['first_observed_difference'])

    def test_incomplete_or_mismatched_requests_rejected(self):
        left=request();right=request();right['request_ordinal']=1
        with self.assertRaises(ValueError):compare_request(left,right)
        right=request();right['sample']=[]
        with self.assertRaises(ValueError):compare_request(left,right)


if __name__=='__main__':unittest.main()
