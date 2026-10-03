import unittest
import numpy as np
from scripts.v30_regroup_key_screen import probe_tiles, support_key_order, screen_need


class KeyScreenTests(unittest.TestCase):
    def check_coverage(self,need,result):
        order=result['order'];h,q,pt=need.shape
        for head in range(h):
            self.assertTrue(np.array_equal(np.sort(order[head]),np.arange(q)))
            self.assertTrue(np.array_equal(order[head]//128,np.arange(q)//128))
            for slot,row in enumerate(order[head]):
                self.assertTrue(np.all(~need[head,row] | result['support'][head,slot//64]))
        self.assertLessEqual(result['candidate_cost']['total_tiles'],result['natural_cost']['total_tiles'])
        self.assertLessEqual(result['candidate_cost']['max_cta_tiles'],result['natural_cost']['max_cta_tiles'])

    def test_nonnested_equal_counts_group_by_set_and_cover_rows(self):
        need=np.zeros((2,256,13),dtype=bool)
        need[:,::2,0]=True;need[:,1::2,12]=True
        result=screen_need(need);self.check_coverage(need,result)
        self.assertEqual(result['accepted_blocks'],4)
        self.assertEqual(result['candidate_cost']['total_tiles'],8)
        self.assertEqual(result['natural_cost']['total_tiles'],16)

    def test_identical_support_stays_identity_including_small_prefix(self):
        for pt in (1,2,5,64):
            for value in (False,True):
                need=np.full((1,128,pt),value,dtype=bool);r=screen_need(need)
                self.check_coverage(need,r);self.assertEqual(r['moved_rows'],0)

    def test_random_and_adversarial_unsampled_support_never_worsens_gate(self):
        rng=np.random.default_rng(30)
        for density in (.01,.2,.7):
            need=rng.random((3,256,97))<density
            r=screen_need(need);self.check_coverage(need,r)
            self.assertTrue(np.array_equal(r['order'],screen_need(need)['order']))

    def test_key_ties_are_original_row_order(self):
        need=np.zeros((1,256,17),dtype=bool);need[0,:,8]=True
        self.assertTrue(np.array_equal(support_key_order(need)[0],np.arange(256)))
        self.assertEqual(probe_tiles(6),tuple(range(6)))

    def test_invalid_geometry_and_probe_rejected(self):
        for pt in (0,-1,1.5):
            with self.assertRaises(ValueError):probe_tiles(pt)
        for need in (np.zeros((1,127,4),dtype=bool),np.zeros((1,128,4),dtype=np.int32)):
            with self.assertRaises(ValueError):screen_need(need)


if __name__ == '__main__': unittest.main()
