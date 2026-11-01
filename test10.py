from multiprocessing import Process
import time
import numpy as np
import rebound
import os
import sys
import pickle
import socket
import resource
import threading
from datetime import datetime

try:
    import psutil  # 用于运行中的 CPU / 内存采样；没装也能跑，只是没有时间序列
except ImportError:
    psutil = None

# ===================== 参数 =====================
N_PROC = 12            # 进程数
N_PARTICLES = 16       # 每个进程的测试粒子数
T_END = 5e7            # 积分时长 [yr]，约 3~4 小时（2.8GHz Xeon 单核估算）
SEED = 12345           # 固定随机种子，保证每台机器跑的是同一份工作
OUTPUT_DIR = "bench_result"
LOG_FILE = "bench_log.txt"
MONITOR_INTERVAL = 10  # 资源采样间隔 [s]
N_ACTIVE = True        # True: 测试粒子不产生引力（N_active=5），结果完全相同但更快
# 可在命令行临时改积分时长做快速测试，如: python bench.py 1e4
if len(sys.argv) > 1:
    T_END = float(sys.argv[1])
# ================================================

# 太阳 + 外行星（写死轨道根数，不联网）
# name, mass[Msun], a[AU], e, inc, Omega, omega, M [rad]
PLANETS = [
    ("Jupiter", 9.547919e-4,  5.2023652, 0.04839, 0.022776, 1.7536, 4.77988, 0.60047),
    ("Saturn",  2.858860e-4,  9.5459070, 0.05386, 0.043362, 1.9837, 5.92352, 0.87170),
    ("Uranus",  4.366244e-5, 19.1931301, 0.04726, 0.013485, 1.2918, 1.69265, 2.48346),
    ("Neptune", 5.151389e-5, 30.0773694, 0.00859, 0.030892, 2.3000, 4.82240, 4.47153),
]
a_J, a_S, a_U, a_N = (p[2] for p in PLANETS)

# 12 个进程依次分配到原来的 4 个区间（每个区间 3 个进程）
REGIONS = [("ajas", a_J, a_S), ("asau", a_S, a_U), ("auan", a_U, a_N), ("anak", a_N, 40)]


def log(msg):
    with open(LOG_FILE, "a") as f:  # 先写日志，再打印
        f.write(msg + "\n")
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):  # 终端断开时只写日志，不让计算进程崩掉
        pass


def run(idx, run_dir):
    name, a1, a2 = REGIONS[idx % 4]
    tag = f"{name}_process{idx // 4 + 1}"

    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = "ias15"
    sim.add(m=1.0)
    for _, m, a, e, inc, Om, om, M in PLANETS:
        sim.add(m=m, a=a, e=e, inc=inc, Omega=Om, omega=om, M=M)

    rng = np.random.default_rng([SEED, idx])
    Q = 40
    q = rng.uniform(a1, a2, N_PARTICLES)
    a = (q + Q) * 0.5
    e = 1 - q / a
    for j in range(N_PARTICLES):
        sim.add(primary=sim.particles[0], a=a[j], e=e[j], inc=0.05)
    if N_ACTIVE:
        sim.N_active = 5   # 只有太阳和 4 颗行星产生引力
    sim.move_to_com()

    t0 = time.time()
    cpu0 = time.process_time()
    sim.integrate(T_END)
    elapsed = time.time() - t0
    cpu = time.process_time() - cpu0
    # 本进程的峰值常驻内存（Linux 单位 KB，macOS 单位 byte）
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = maxrss / 1024 if sys.platform.startswith("linux") else maxrss / 1024**2

    # 只保存积分结束时的结果
    orbit_data = [{
        "Index": i, "a": p.a, "e": p.e, "inc": p.inc,
        "Omega": p.Omega, "omega": p.omega, "M": p.M,
    } for i, p in enumerate(sim.particles[5:], start=6)]
    with open(os.path.join(run_dir, f"{tag}.bin"), "wb") as f:
        pickle.dump({"q0": q, "final": orbit_data, "steps": sim.steps_done,
                     "time_s": elapsed, "cpu_s": cpu, "peak_mem_MB": mem_mb}, f)

    # CPU% = CPU 时间 / 墙钟时间；单核跑满 = 100%，明显低于 100% 说明被抢占或超订
    log(f"{tag:<16} time {fmt(elapsed):>14}   steps {sim.steps_done:>12,}   "
        f"CPU {cpu / elapsed * 100:5.1f}%   peak mem {mem_mb:6.1f} MB")


def fmt(s):
    return f"{int(s // 3600)}h {int(s % 3600 // 60)}min {s % 60:.0f}s"


def monitor(procs, csv_path, stop, stats):
    """每隔 MONITOR_INTERVAL 秒采样：本脚本所有进程的 CPU% 和内存 + 整机每个核心的占用率。"""
    ps = []
    for p in procs:
        try:
            ps.append(psutil.Process(p.pid))
        except psutil.Error:
            pass
    for p in ps:
        try:
            p.cpu_percent(None)  # 第一次调用只是建立基准
        except psutil.Error:
            pass
    psutil.cpu_percent(percpu=True)
    n_cpu = psutil.cpu_count()
    t0 = time.time()

    with open(csv_path, "w") as f:
        f.write("elapsed_s,script_cpu_pct,script_cores_used,script_mem_MB,"
                "system_cpu_pct,system_mem_used_pct,"
                + ",".join(f"core{i}_pct" for i in range(n_cpu)) + "\n")
        while not stop.wait(MONITOR_INTERVAL):
            cpu_sum, mem_sum = 0.0, 0.0
            for p in ps:
                try:
                    cpu_sum += p.cpu_percent(None)   # 100 = 占满 1 个核
                    mem_sum += p.memory_info().rss
                except psutil.Error:                 # 进程已结束
                    pass
            cores = psutil.cpu_percent(percpu=True)
            sys_cpu = sum(cores) / len(cores)
            sys_mem = psutil.virtual_memory().percent
            mem_mb = mem_sum / 1024**2
            f.write(f"{time.time() - t0:.0f},{cpu_sum:.1f},{cpu_sum / 100:.2f},{mem_mb:.1f},"
                    f"{sys_cpu:.1f},{sys_mem:.1f}," + ",".join(f"{c:.0f}" for c in cores) + "\n")
            f.flush()
            stats["samples"].append((cpu_sum, mem_mb, sys_cpu))


if __name__ == "__main__":
    # 每次运行单独一个带时间戳的文件夹，不会覆盖之前的结果
    run_dir = os.path.join(OUTPUT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    n_cpu = os.cpu_count()
    log("-----------------------------------")
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S} START | host {socket.gethostname()} | "
        f"{n_cpu} logical CPUs | {N_PROC} processes x {N_PARTICLES} particles | "
        f"t_end {T_END:g} yr | N_active {'5' if N_ACTIVE else 'all'} | output {run_dir}")
    if psutil is None:
        log("WARNING: psutil not installed, no resource time series (pip install psutil)")

    start = time.time()
    procs = [Process(target=run, args=(i, run_dir)) for i in range(N_PROC)]
    for p in procs:
        p.start()

    stop = threading.Event()
    stats = {"samples": []}
    if psutil is not None:
        csv_path = os.path.join(run_dir, "resource_usage.csv")
        mon = threading.Thread(target=monitor, args=(procs, csv_path, stop, stats), daemon=True)
        mon.start()

    for p in procs:
        p.join()
    total = time.time() - start
    stop.set()
    if psutil is not None:
        mon.join()

    # 所有子进程的 CPU 总时间（精确值，来自操作系统统计）
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_total = ru.ru_utime + ru.ru_stime
    avg_cores = cpu_total / total

    log("*" * 60)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S} ALL DONE | total time {fmt(total)}")
    log(f"Total CPU time {fmt(cpu_total)} | avg cores used {avg_cores:.2f} / {N_PROC} processes "
        f"(parallel efficiency {avg_cores / N_PROC * 100:.1f}%)")
    s = stats["samples"]
    if s:
        log(f"Script CPU  avg {sum(x[0] for x in s) / len(s):7.1f}%   max {max(x[0] for x in s):7.1f}%"
            f"   (100% = 1 core)")
        log(f"Script mem  avg {sum(x[1] for x in s) / len(s):7.1f} MB  max {max(x[1] for x in s):7.1f} MB")
        log(f"System CPU  avg {sum(x[2] for x in s) / len(s):7.1f}%   (all cores, incl. other users)")
        log(f"Time series saved to {csv_path}")
    log("*" * 60)