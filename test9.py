from multiprocessing import Process
import time
import numpy as np
import rebound
import os
import sys
import pickle
import socket
from datetime import datetime

# ===================== 参数 =====================
N_PROC = 12            # 进程数
N_PARTICLES = 16       # 每个进程的测试粒子数
T_END = 5e7            # 积分时长 [yr]，约 3~4 小时（2.8GHz Xeon 单核估算）
SEED = 12345           # 固定随机种子，保证每台机器跑的是同一份工作
OUTPUT_DIR = "bench_result"
LOG_FILE = "bench_log.txt"
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
    sim.move_to_com()

    t0 = time.time()
    sim.integrate(T_END)
    elapsed = time.time() - t0

    # 只保存积分结束时的结果
    orbit_data = [{
        "Index": i, "a": p.a, "e": p.e, "inc": p.inc,
        "Omega": p.Omega, "omega": p.omega, "M": p.M,
    } for i, p in enumerate(sim.particles[5:], start=6)]
    with open(os.path.join(run_dir, f"{tag}.bin"), "wb") as f:
        pickle.dump({"q0": q, "final": orbit_data, "steps": sim.steps_done,
                     "time_s": elapsed}, f)

    msg = f"{tag:<16} time {fmt(elapsed):>14}   steps {sim.steps_done:>12,}"
    with open(LOG_FILE, "a") as f:  # 先写日志，再打印
        f.write(msg + "\n")
    print(msg, flush=True)


def fmt(s):
    return f"{int(s // 3600)}h {int(s % 3600 // 60)}min {s % 60:.0f}s"


if __name__ == "__main__":
    # 每次运行单独一个带时间戳的文件夹，不会覆盖之前的结果
    run_dir = os.path.join(OUTPUT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    header = (f"-----------------------------------\n"
              f"{datetime.now():%Y-%m-%d %H:%M:%S} START | host {socket.gethostname()} | "
              f"{N_PROC} processes x {N_PARTICLES} particles | t_end {T_END:g} yr | output {run_dir}")
    print(header)
    with open(LOG_FILE, "a") as f:
        f.write(header + "\n")

    start = time.time()
    procs = [Process(target=run, args=(i, run_dir)) for i in range(N_PROC)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    total = time.time() - start

    msg = f"{datetime.now():%Y-%m-%d %H:%M:%S} ALL DONE | total time {fmt(total)}"
    print("*" * 60)
    print(msg)
    print("*" * 60)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")