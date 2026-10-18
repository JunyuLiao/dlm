from scripts.v31_refresh_replay_bootstrap import boot_observer


def test_patch_precedes_observer_import():
    steps=[]
    def patch():steps.append('patch');return 'source.py'
    def load():
        assert steps==['patch'];steps.append('import');return 'observer'
    assert boot_observer(patch,load)==('source.py','observer')
    assert steps==['patch','import']
