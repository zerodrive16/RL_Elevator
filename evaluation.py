"""
Gemeinsame Auswertung fuer alle Phasen des Aufzug-Projekts (Random-Baseline,
Value Iteration mit voller Information, spaeter reduzierte Beobachtung).

Jede Phase ruft evaluate() mit denselben (n_floors, p, gamma, steps, n_episodes)
auf und haengt das Ergebnis per append_result() an results/runs.csv an - so
bleiben alle bisherigen Phasen-Ergebnisse erhalten und lassen sich am Ende
gemeinsam vergleichen/plotten (Teil (c) der Aufgabenstellung).
"""

import csv
import os
import random
import statistics

from elevator_environment import ElevatorEnv

RESULTS_DIR = "results"
RUNS_CSV = os.path.join(RESULTS_DIR, "runs.csv")


def make_random_policy(seed):
    """policy_fn(env) -> Aktion, zieht gleichverteilt aus den verfuegbaren
    Aktionen. Eigener RNG, unabhaengig vom Ankunfts-RNG der Umgebung."""
    rng = random.Random(seed)

    def policy_fn(env):
        return rng.choice(env.available_actions())

    return policy_fn


def make_table_policy(policy):
    """policy_fn(env) -> Aktion, Tabellen-Lookup einer (z.B. per Value
    Iteration) vorberechneten Strategie."""

    def policy_fn(env):
        state = (env.floor, env.door_open, tuple(env.waiting_dest), tuple(env.cabin_dest))
        return policy[state]

    return policy_fn


def run_episode(policy_fn, n_floors, p, steps, seed):
    """Fuehrt eine policy_fn ueber `steps` Schritte auf einer frisch
    geseedeten ElevatorEnv aus. Gibt die Kosten pro Schritt und die
    abgeschlossenen Fahrtdauern zurueck."""
    env = ElevatorEnv(n_floors=n_floors, p=p, seed=seed)  # eigene, frische Umgebung pro Episode
    costs = []
    for _ in range(steps):
        action = policy_fn(env)          # naechste Aktion von der uebergebenen Strategie erfragen
        _, cost = env.step(action)
        costs.append(cost)               # Kosten Schritt fuer Schritt sammeln (fuer Summe/Diskontierung)
    return costs, list(env.completed_trips)


def discounted_sum(costs, gamma):
    # Sigma_t gamma^t * cost_t, t=0,1,2,... (Monte-Carlo-Schaetzer fuer den diskontierten
    # unendlichen Horizont, den V* in value_iteration.py exakt berechnet - siehe dortige
    # Herleitung zur "T-Range": bei gamma=0.9 ist der Fehler durch die Begrenzung auf
    # `steps` Schritte statt unendlich vieler vernachlaessigbar klein)
    return sum(c * gamma**t for t, c in enumerate(costs))


def evaluate(label, policy_fn_factory, n_floors, p, gamma, steps, n_episodes, seed_start=0):
    """Mittelt eine Strategie ueber n_episodes unabhaengige Episoden (verschiedene
    Seeds), damit Random- und VI-Strategie fair (gleiche Bedingungen, gemittelt
    statt eine einzelne Stichprobe) vergleichbar sind.

    policy_fn_factory(seed) -> policy_fn(env) -> Aktion. Der seed wird sowohl
    fuer die Strategie (falls sie eigene Zufallsentscheidungen trifft, z.B.
    Random-Strategie) als auch fuer die Umgebung (Ankunftsprozess) verwendet,
    damit jede Episode reproduzierbar ist.
    """
    total_costs, discounted_costs, trip_durations = [], [], []
    # Seeds 0,1,2,...,n_episodes-1 (bzw. ab seed_start): jede Episode bekommt einen
    # eigenen, aber reproduzierbaren Zufallsverlauf des Ankunftsprozesses
    for i in range(n_episodes):
        seed = seed_start + i
        policy_fn = policy_fn_factory(seed)
        costs, trips = run_episode(policy_fn, n_floors, p, steps, seed=seed)
        total_costs.append(sum(costs))                       # undiskontierte Gesamtkosten dieser Episode
        discounted_costs.append(discounted_sum(costs, gamma))  # diskontierte Gesamtkosten dieser Episode
        trip_durations.extend(trips)                          # alle abgeschlossenen Fahrten sammeln

    return {
        "label": label,
        "n_floors": n_floors,
        "p": p,
        "gamma": gamma,
        "steps": steps,
        "n_episodes": n_episodes,
        "avg_total_cost": round(statistics.mean(total_costs), 4),
        "avg_cost_per_step": round(statistics.mean(total_costs) / steps, 4),
        "avg_discounted_cost": round(statistics.mean(discounted_costs), 4),
        "avg_trip_duration": round(statistics.mean(trip_durations), 4) if trip_durations else None,
        "n_completed_trips_total": len(trip_durations),
    }


def append_result(result, path=RUNS_CSV):
    """Haengt eine Ergebniszeile an path an (Header nur beim allerersten
    Schreiben). Ueberschreibt nie vorhandene Zeilen -> alle bisherigen
    Phasen-Ergebnisse bleiben erhalten."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    print(f"Ergebnis '{result['label']}' angehaengt an: {path}")