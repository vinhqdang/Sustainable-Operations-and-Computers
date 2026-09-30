"""Build data/traces/alibaba2018_jobs.csv.gz from the Alibaba 2018 cluster trace.

Input: batch_task.csv from cluster-trace-v2018 (https://github.com/alibaba/clusterdata),
columns task_name, instance_num, job_name, task_type, status, start_time, end_time,
plan_cpu (100 = one core), plan_mem. Only terminated tasks with valid timestamps and a
planned CPU request are kept. For each job (all tasks of a job_name, i.e. one DAG):
  a     = earliest task start time (s); the trace records no submission time, so the
          first start is used as the arrival time
  e     = latest task end time (s)
  work  = sum over tasks of instance_num * (end - start) / 3600 * plan_cpu / 100
          (core-hours, using the planned CPU request, not measured utilisation)
  width = largest planned core count of a single task, instance_num * plan_cpu / 100
  dur   = (e - a) / 3600 (hours)"""
import sys
import pandas as pd
from common import DATA

COLS = ["task_name", "instance_num", "job_name", "task_type", "status", "start_time",
        "end_time", "plan_cpu", "plan_mem"]


def build(src):
    d = pd.read_csv(src, names=COLS, usecols=["instance_num", "job_name", "status", "start_time",
                                               "end_time", "plan_cpu"])
    n_all = len(d)
    d = d[(d.status == "Terminated") & (d.start_time > 0) & (d.end_time > d.start_time) & d.plan_cpu.notna()]
    d["core_h"] = d.instance_num * (d.end_time - d.start_time) / 3600 * d.plan_cpu / 100
    d["cores"] = d.instance_num * d.plan_cpu / 100
    g = d.groupby("job_name").agg(a=("start_time", "min"), e=("end_time", "max"),
                                  work=("core_h", "sum"), width=("cores", "max"))
    g["dur"] = (g.e - g.a) / 3600
    print(f"tasks {n_all}, kept {len(d)}, jobs {len(g)}", flush=True)
    return g


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DATA / "traces" / "batch_task.csv"
    build(src).to_csv(DATA / "traces" / "alibaba2018_jobs.csv.gz")
