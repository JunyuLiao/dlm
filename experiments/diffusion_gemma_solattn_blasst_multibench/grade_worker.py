"""Run ONLY inside the network-disabled grading container, never on the host."""
import base64
import json
import pickle
import zlib
from pathlib import Path
from lcb_runner.evaluation import codegen_metrics


def main():
    rows=json.loads(Path('/input.json').read_text())
    samples=[]
    for row in rows:
        r=row['lcb'];public=json.loads(r['public_test_cases'])
        try: private=json.loads(r['private_test_cases'])
        except (ValueError,TypeError):
            private=json.loads(pickle.loads(zlib.decompress(base64.b64decode(r['private_test_cases']))))
        cases=public+private
        samples.append({'input_output':json.dumps(dict(inputs=[t['input'] for t in cases],
            outputs=[t['output'] for t in cases],fn_name=json.loads(r['metadata']).get('func_name')))})
    metrics,results,metadata=codegen_metrics(samples,[[r['code']] for r in rows],k_list=[1],num_process_evaluate=2,timeout=6)
    grades=[dict(id=r['id'],code_hash=r['code_hash'],score=float(all(v>0 for v in results[i][0])),
        tests=results[i][0],metadata=metadata[i]) for i,r in enumerate(rows)]
    print('CONTROLLED_GRADES='+json.dumps(dict(metrics=metrics,grades=grades)))


if __name__=='__main__': main()
