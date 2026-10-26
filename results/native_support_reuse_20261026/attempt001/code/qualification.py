"""Same-policy active-state comparisons; unused history is not a stopping state."""
def exact_request(a,b):
    keys=['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards','prompt_tokenization_sha256','startup_model_calls_in_request','extra_inside_execute_Python_forwards']
    return {k:a[k]==b[k] for k in keys}

def core_trace_equal(a,b):
    extra={'reference_winner_crossings','actual_native_winner_changes','worst_margin_erosion','mean_absolute_margin_change','signed_margin_quantiles','entropy_by_position','native_margin_by_position','signed_reference_winner_margin_change','reference_pair_margin'}
    def active(row):
        excluded=extra|({'stable_position_fraction'} if row.get('history_ready') is False else set())
        return {k:v for k,v in row.items() if k not in excluded}
    return len(a)==len(b) and all(active(x)==active(y) for x,y in zip(a,b))

