# The Ascend profiler for two steps, on the reference ("fixed") and on a heterogeneous ("both") dataset:
# what the spans of hetero_profile cannot show, the kernels, the streams' waits and the collectives.
#
# A trace of one step of this model takes gigabytes per rank, and a profiled run writes one per profiled
# rank on the node that holds it, so this plan profiles FOUR ranks, two per node and two per expert-parallel
# group: enough for the trace report to match their all-to-alls against each other, and for the component
# trace to draw both ranks of each node. profiling.ranks=[] with profiling.rank=-1 profiles all 32 instead,
# which is what filled a node's disk on 2026-10-05. The profiler also drops its raw collection directory
# once parsed (profiling.data_simplification, set in the YAML).
#
# The trace report runs on the nodes (--ranks: per rank the compute, the waits and the idle time of the
# compute stream, the exposed all-to-alls and all-gathers, and the k-th all-to-all matched across the
# profiled ranks of an EP group: the wait for the last rank to arrive against the transfer). Traces stay
# on the nodes; delete them when read:
#   cluster exec 'rm -rf /home/pl/runs/qwen3_vl_30b_perf/<campaign>_*/profile'
# The EP instrument is off: its device events would sit in the traces. hetero_profile stays on, so each
# rank's workload is recorded next to its trace (profiler step N is optimizer step N + 1).
# After the trace report the campaign runs component_trace.py on the nodes, for the busiest and the idlest rank of
# each node: the profiler's own trace with the detected components added as new processes (from the profile alone,
# and from the hooks placed on the profiler's timestamps), and a report that checks one against the other
# (components.txt and components/<node>_* come back; each rank's trace, as large as the original, stays on the node in
# profile/components_full: gather the one to open in Perfetto).
CONFIG=examples/qwen3_vl_30b_perf/train_32dev_a3_hetero.yaml
DATA=/home/pl/data/qwen3_vl_30b_perf
PROFILE="--training.train_iters=8 --ep_instrument.enabled=false --hetero_profile.start_step=2 \
--profiling.enabled=true --profiling.ranks=[6,13,22,29] --profiling.start_step=6 --profiling.end_step=8"
RUNS=(
  "profile_fixed $PROFILE --dataset.data_path=$DATA/hetero_fixed_n640/vlm_conversations.json"
  "profile_both $PROFILE --dataset.data_path=$DATA/hetero_both_n640/vlm_conversations.json"
)
