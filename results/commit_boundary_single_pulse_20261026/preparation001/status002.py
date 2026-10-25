from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
for root in sorted(list((R/'boundary').glob('attempt*'))+list((R/'panel').glob('attempt*'))):
 print(str(root.relative_to(R)))
 for n in ['controller_stopped.json','execution_complete.json','qualification_complete.json','prelaunch_superseded.json']:
  p=root/n
  if p.exists():print(n,p.read_text()[:350])
 for n in ['controller_heartbeat.json','heartbeat.json']:
  p=root/'private'/n
  if p.exists():print(n,p.read_text())
 for d in ['executions','runs']:
  path=root/d
  if path.exists():
   print('receipts',len(list(path.glob('**/receipt.json'))),'audit',len(list(path.glob('**/audit*.json'))),'pairs',len(list(path.glob('**/pair*.json'))))
   for p in path.glob('**/failure.json'):print('FAILURE',str(p.relative_to(R)),p.read_text()[:1000])
