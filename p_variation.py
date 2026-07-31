"""
Teil (c) der Aufgabenstellung: Ankunftswahrscheinlichkeit p (und den
Warteschlangen-Deckel K fuer Phase 2b/3) variieren und alle bisherigen
Phasen (1, 2, 2b, 3) fuer jede Kombination wiederholen.

Phase 1/2 haengt NICHT von K ab (nur von p) - wird deshalb pro p nur EINMAL
gerechnet, nicht pro (p, K)-Kombination (das waere unnoetig doppelte Arbeit).
Phase 2b/3 laufen fuer jede Kombination aus P_VALUES x K_VALUES.

Alle Ergebnisse werden an results/runs.csv angehaengt (append_result() ist
ein Upsert - ein erneuter Lauf mit gleichem p/K ersetzt nur die jeweilige
Zeile, alle anderen p/K-Kombinationen bleiben unberuehrt erhalten).
"""

import random

from phase1_2.elevator_environment import ElevatorEnv
from phase2b.elevator_environment_2 import ElevatorEnvV2
import phase1_2.value_iteration as vi1
import phase2b.value_iteration_2 as vi2
import phase3.value_iteration_3 as vi3
from common.evaluation import evaluate, append_result, make_random_policy, make_table_policy

N_FLOORS = 4
GAMMA = 0.9
CAPACITY = 5
EVAL_STEPS = 150
EVAL_EPISODES = 20
P_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
K_VALUES = [2, 3]
# K=3 ist der aktuelle Referenzwert (siehe PHASES.md "Naechste Schritte") - K=2
# bleibt als dokumentiertes Vergleichsbeispiel fuer das Ueberlastungs-Kollaps-
# Artefakt bei hoher Last mit drin (siehe PHASES.md, Abschnitt
# "Systemueberlastung" + Nachtrag). K=4/5 noch nicht getestet.


def make_v2_random_policy(seed):
    rng = random.Random(seed)
    return lambda env: rng.choice(env.available_actions())


def make_v2_table_policy(policy):
    def factory(seed):
        return lambda env: policy[env.get_state()]
    return factory


def run_phase12(p):
    """Phase 1 (Random) + Phase 2 (VI, volle Info/Orakel) - haengt nur von p ab."""
    print(f"\n===== Phase 1/2, p={p} =====")
    V1, policy1 = vi1.value_iteration(n_floors=N_FLOORS, p=p, gamma=GAMMA, theta=1e-4, verbose=False)
    env1_factory = lambda seed: ElevatorEnv(n_floors=N_FLOORS, p=p, seed=seed)

    results = {
        "phase1_random": evaluate(
            label="phase1_random", policy_fn_factory=make_random_policy,
            env_factory=env1_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
            n_floors=N_FLOORS, p=p,
        ),
        "phase2_vi_full_info": evaluate(
            label="phase2_vi_full_info", policy_fn_factory=lambda seed: make_table_policy(policy1),
            env_factory=env1_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
            n_floors=N_FLOORS, p=p,
        ),
    }
    start1 = (1, False, tuple([None] * N_FLOORS), tuple([0] * N_FLOORS))
    print(f"  fertig: V*={V1[start1]:.2f}")

    for r in results.values():
        append_result(r)
    return results


def run_phase2b3(p, K):
    """Phase 2b (VI, realistische Kapazitaet) + Phase 3 (Boolean-Beobachtung,
    ausgefuehrt auf derselben Phase-2b-Umgebung mit demselben K)."""
    print(f"\n===== Phase 2b/3, p={p}, K={K} =====")

    V2, policy2 = vi2.value_iteration(n_floors=N_FLOORS, p=p, gamma=GAMMA, capacity=CAPACITY,
                                       max_wait_per_floor=K, theta=1e-4, verbose=False)
    env2_factory = lambda seed: ElevatorEnvV2(n_floors=N_FLOORS, p=p, capacity=CAPACITY,
                                               max_wait_per_floor=K, seed=seed)
    results = {
        "phase2b_random": evaluate(
            label="phase2b_random", policy_fn_factory=make_v2_random_policy,
            env_factory=env2_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
            n_floors=N_FLOORS, p=p, capacity=CAPACITY, max_wait_per_floor=K,
        ),
        "phase2b_vi_realistic": evaluate(
            label="phase2b_vi_realistic", policy_fn_factory=make_v2_table_policy(policy2),
            env_factory=env2_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
            n_floors=N_FLOORS, p=p, capacity=CAPACITY, max_wait_per_floor=K,
        ),
    }
    start2 = (1, False, tuple([0] * N_FLOORS), tuple([0] * N_FLOORS))
    print(f"  Phase 2b fertig: V*={V2[start2]:.2f}")

    V3, policy3 = vi3.value_iteration(n_floors=N_FLOORS, p=p, gamma=GAMMA, theta=1e-4, verbose=False)
    results["phase3_boolean_obs"] = evaluate(
        label="phase3_boolean_obs", policy_fn_factory=lambda seed: vi3.make_table_policy_v3(policy3),
        env_factory=env2_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=N_FLOORS, p=p, capacity=CAPACITY, max_wait_per_floor=K,
    )
    start3 = (1, False, tuple([0] * N_FLOORS), tuple([0] * N_FLOORS))
    print(f"  Phase 3 fertig: V*_approx={V3[start3]:.2f} (Naeherungs-Kosten, nicht mit obigen V* vergleichbar)")

    for r in results.values():
        append_result(r)
    return results


if __name__ == "__main__":
    all_results = {}
    for p in P_VALUES:
        all_results[p] = {"phase12": run_phase12(p)}
        for K in K_VALUES:
            all_results[p][K] = run_phase2b3(p, K)

    print("\n\n===== Zusammenfassung: avg_cost_per_step =====")
    header = f"{'p':<6}{'phase1':>12}{'phase2':>12}"
    for K in K_VALUES:
        header += f"{'phase2b_K'+str(K):>14}{'phase3_K'+str(K):>14}"
    print(header)
    for p in P_VALUES:
        row = f"{p:<6}"
        row += f"{all_results[p]['phase12']['phase1_random']['avg_cost_per_step']:>12}"
        row += f"{all_results[p]['phase12']['phase2_vi_full_info']['avg_cost_per_step']:>12}"
        for K in K_VALUES:
            row += f"{all_results[p][K]['phase2b_vi_realistic']['avg_cost_per_step']:>14}"
            row += f"{all_results[p][K]['phase3_boolean_obs']['avg_cost_per_step']:>14}"
        print(row)