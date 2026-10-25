from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
for track in ['boundary','panel']:
 root=R/track/'attempt001';print(track)
 for n in ['controller_stopped.json','execution_complete.json','qualification_complete.json']:
  p=root/n
  if p.exists():print(n,p.read_text()[:500])
 for n in ['controller_heartbeat.json','heartbeat.json']:
  p=root/'private'/n
  if p.exists():print(n,p.read_text())
 for path in [root/'executions',root/'runs']:
  if path.exists():
   print('receipts',len(list(path.glob('**/receipt.json'))),'audit cells',len(list(path.glob('**/audit*.json'))),'pairs',len(list(path.glob('**/pair*.json'))))
   for p in path.glob('**/failure.json'):print('FAILURE',str(p.relative_to(R)),p.read_text()[:1200])
