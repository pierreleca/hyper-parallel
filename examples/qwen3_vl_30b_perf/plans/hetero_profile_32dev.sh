# The Ascend profiler for two steps, on the reference ("fixed") and on a heterogeneous ("both") dataset:
# what the spans of hetero_profile cannot show, the kernels, the streams' waits and the collectives.
#
# Every rank is profiled (profiling.rank=-1): the data-imbalance question is a comparison between ranks, so
# the k-th all-to-all matched across a whole expert-parallel group, and the per-rank compute and wait, need
# all of them. The overhead is then the same on every rank, so the ranks stay comparable with each other;
# the absolute step time is inflated, so it is not comparable with an unprofiled run.
#
# A trace of one step of this model takes gigabytes per rank, and a profiled run writes one per rank on the
# node that holds it, so the two runs leave 2 x 16 traces per node: check the disk before (df -h /home/pl),
# delete them once read, and with ANALYSE_EACH=1 the campaign analyses each run before the next one trains,
# so only one run's traces are on the nodes at a time. The profiler drops its raw collection directory once
# parsed (profiling.data_simplification, in the YAML). If the disk is short, profiling.ranks names a few
# ranks instead, e.g. --profiling.ranks=[6,13,22,29], two per node and two per expert-parallel group.
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
--profiling.enabled=true --profiling.rank=-1 --profiling.start_step=6 --profiling.end_step=8"
RUNS=(
  "profile_fixed $PROFILE --dataset.data_path=$DATA/hetero_fixed_n640/vlm_conversations.json"
  "profile_both $PROFILE --dataset.data_path=$DATA/hetero_both_n640/vlm_conversations.json"
)
