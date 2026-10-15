from scripts.v31_value_native_study import paired


def test_paired_accuracy_clusters_repeated_seeds_by_problem():
    control={'aime|0|42|0':True,'aime|0|43|0':False,'aime|1|42|0':True,'aime|1|43|0':True}
    candidate={**control,'aime|0|43|0':True}
    result=paired('aime26',control,candidate,{})
    assert result['clusters']==2
    assert result['delta_score_points']==25
    assert result['clustering']=='problem'


def test_ruler_fractional_scores_are_not_binary_and_cluster_by_task():
    control={'ruler|0|42|0':50.,'ruler|1|42|0':100.,'ruler|2|42|0':100.}
    candidate={**control,'ruler|0|42|0':40.}
    result=paired('ruler128k',control,candidate,{0:'task_a',1:'task_a',2:'task_b'})
    assert result['clusters']==2
    assert result['delta_score_points']==-2.5
    assert result['clustering']=='task'
