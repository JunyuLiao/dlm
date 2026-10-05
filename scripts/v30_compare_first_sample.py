"""Export equality booleans only from private tensor/RNG trace manifests.

The first sampler trace synchronizes and does not establish unchanged timing.
Uninitialized argmax/confidence/history storage before its first use is retained
privately for audit but excluded from the first-divergence classification.
"""


def equal_tensor(left, right):
    if not {'sha256','shape','dtype'} <= set(left) or not {'sha256','shape','dtype'} <= set(right):
        raise ValueError('Missing tensor provenance')
    return all(left[k] == right[k] for k in ('sha256','shape','dtype'))


def compare_request(left, right):
    if left['request_ordinal'] != right['request_ordinal']:
        raise ValueError('Request order mismatch')
    for row in (left,right):
        if len(row['init'])!=1 or len(row['sample'])!=1:
            raise ValueError('Incomplete first-sample trace')
    li,ri=left['init'][0],right['init'][0]
    ls,rs=left['sample'][0],right['sample'][0]
    def tensor(a,b,name):
        if name not in a['tensors'] or name not in b['tensors']:raise ValueError('Missing traced tensor')
        return equal_tensor(a['tensors'][name],b['tensors'][name])
    checks = dict(init_rng_before_equal=li['before']['rng']==ri['before']['rng'],
                  initialized_canvas_equal=tensor(li['after'],ri['after'],'canvas'),
                  init_rng_after_equal=li['after']['rng']==ri['after']['rng'],
                  first_sampler_rng_before_equal=ls['before']['rng']==rs['before']['rng'])
    for name in ('canvas','step_tensor','is_encoder_phase','sc_embeds','history_len_tensor','decode_slots','logits'):
        checks['sampler_input_'+name+'_equal']=tensor(ls['before'],rs['before'],name)
    for name in ('canvas','argmax_canvas','step_tensor','is_encoder_phase','confident_tensor','sc_embeds','history_len_tensor'):
        checks['sampler_output_'+name+'_equal']=tensor(ls['after'],rs['after'],name)
    checks['first_sampler_rng_after_equal']=ls['after']['rng']==rs['after']['rng']
    first=next((key for key,value in checks.items() if not value),None)
    # Equal hashes compare observed bytes, not hidden compiler RNG state.
    return dict(request_ordinal=left['request_ordinal'],**checks,first_observed_difference=first,
                earliest_later_difference_located=False,compiled_rng_state_proven_equal=False,
                unused_initial_storage_excluded=['argmax_canvas','confident_tensor','history'],
                performance_claim_allowed=False)
