from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parents[1];OLD=R.parent/'response_behavior_and_stopping_headroom_20261025'
for track in ['boundary','panel']:
 a=R/track/'attempt001'
 for d in ['code','tests','private']:(a/d).mkdir(parents=True,exist_ok=False)
 for n in ['runtime_paths.json','external_sources.json','scorer_sources.json']:shutil.copyfile(OLD/'track_r/attempt001/private'/n,a/'private'/n)
 for n in ['model_fingerprints.json','runtime_fingerprints.json']:shutil.copyfile(OLD/'track_r/attempt001'/n,a/n)
a=R/'boundary/attempt001'
for n in ['bank.py','runtime.py','measure.py','worker.py']:shutil.copyfile(OLD/'track_h/attempt001/code'/n,a/'code'/n)
shutil.copyfile(OLD/'track_h/attempt001/private/manifest.json',a/'private/manifest.json')
