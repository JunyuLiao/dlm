"""Report the user-authorized eight-arm panel; exact greedy stays diagnostic."""
import v31_value_longbench_report as report
from v31_value_longbench_scope_continue import ARMS


if __name__ == '__main__':
    report.ARMS = ARMS
    report.main()
