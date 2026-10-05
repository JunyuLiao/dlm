import copy
import unittest
from scripts.v30_peer_qualification_contract import FORMAL, strict_formal_ready, validate_test_receipt


class QualificationContractTests(unittest.TestCase):
    def complete(self):
        return dict(complete=True,phase="all_strict_scored_public_exports_ready_for_root_review",
                    completed={k:dict(requests=n,workers=32,source_commit=s) for k,(n,s) in FORMAL.items()})

    def test_waits_for_scores_and_rejects_failed_predecessor(self):
        self.assertFalse(strict_formal_ready(dict(complete=False,phase="cpu_scoring")))
        with self.assertRaises(RuntimeError):strict_formal_ready(dict(complete=False,phase="failed_preserved_review_required"))
        self.assertTrue(strict_formal_ready(self.complete()))

    def test_wrong_source_incomplete_requests_or_workers_rejected(self):
        for field,value in (("source_commit","wrong"),("requests",1887),("workers",31)):
            s=self.complete();s['completed']['longbench'][field]=value
            with self.assertRaises(ValueError):strict_formal_ready(s)
        s=self.complete();del s['completed']['aime']
        with self.assertRaises(ValueError):strict_formal_ready(s)

    def test_tests_cannot_be_skipped_duplicated_or_from_wrong_kernel(self):
        spec=dict(peer_commit='p',kernel_sha256='k',must_pass_cases=2,cases_per_function=dict(a=1,b=1))
        receipt=dict(peer_source_commit='p',kernel_sha256='k',returncode=0,tests=[dict(name='a',result='passed'),dict(name='b',result='passed')])
        self.assertTrue(validate_test_receipt(receipt,spec))
        for change in ('skip','duplicate','kernel','missing','exit','other_test'):
            r=copy.deepcopy(receipt)
            if change=='skip':r['tests'][0]['result']='skipped'
            elif change=='duplicate':r['tests'][1]['name']='a'
            elif change=='kernel':r['kernel_sha256']='other'
            elif change=='missing':r['tests'].pop()
            elif change=='other_test':r['tests'][1]['name']='c'
            else:r['returncode']=1
            with self.assertRaises(ValueError):validate_test_receipt(r,spec)


if __name__=='__main__':unittest.main()
