from scripts.blasst_hopper_reproduce import parse_dense, parse_skip


def test_parse_dense_xqa_output():
    text = """batchSize: 64, num_k_heads=4, seqLen: 16384, original attention kernel
dramSolRatio: 89.798401% (0.714068 ms, TOPS = 48.118290)
batchSize: 64, num_k_heads=4, seqLen: 65536, original attention kernel
dramSolRatio: 90.597992% (2.828993 ms, TOPS = 48.582291)
"""
    assert parse_dense(text) == {16384: 0.714068, 65536: 2.828993}


def test_parse_blasst_xqa_output():
    text = """batchSize: 64, num_k_heads=4, seqLen: 65536, skipSoftmaxThreshold: 0.900000
kernel skippedBlockCount: 232700/314572 (73.97%)
dramSolRatio: 146.579330% (1.748548 ms, TOPS = 78.601746)
"""
    assert parse_skip(text) == [
        {
            "sequence_length": 65536,
            "threshold": 0.9,
            "skipped_blocks": 232700,
            "total_blocks": 314572,
            "sparsity_percent": 73.97,
            "time_ms": 1.748548,
        }
    ]
