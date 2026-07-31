"""
Value Iteration fuer Phase 3 (reduzierte Beobachtung - die im Aufgabenblatt
tatsaechlich beschriebene Variante)
==========================================================================

Aufbauend auf Phase 2b (elevator_environment_2.py::ElevatorEnvV2), aber der
Agent beobachtet w_i/z_i jetzt nur noch als Boolean ("wartet dort jemand?" /
"will jemand aus der Kabine dorthin?"), keine exakten Zaehler mehr - exakt
das, was die Aufgabenstellung beschreibt ("nicht wie viele, nur ob").

WICHTIG - "streng genommen POMDP, hier als MDP approximiert" (siehe README/
Session-Diskussion): die WAHRE Umgebung bleibt ElevatorEnvV2 unveraendert
(echte Zaehler, echte Kapazitaet, echte Kosten c = wartend + in_kabine). Die
Value Iteration hier rechnet aber auf einem EIGENEN, kleineren Naeherungs-MDP,
das die Boolean-Beobachtung so behandelt, ALS WAERE SIE der volle Markov-
Zustand (bewusste Vereinfachung, keine echte POMDP-Loesung):

  - Naeherungs-Kostenfunktion c_approx(obs,a) = sum(w_bool) + sum(z_bool)
    ("Anzahl Etagen mit >=1 Wartendem" + "Anzahl Zielrichtungen mit >=1
    Kabineninsassen") - NICHT die echten Kopfzahlen, da der Agent die beim
    Planen gar nicht kennen kann. Das ist eine bewusste Naeherung, keine
    Neudefinition der eigentlichen (mit dem Betreuer abgesprochenen) Kosten.
  - Boarding-Modell: "es wartet jemand" wird beim Einsteigen so behandelt,
    als waere es GENAU eine Person (dieselbe Vereinfachung, die Phase 1/2
    schon hatte) - Ziel wird wie in Phase 2b erst beim Einsteigen gewuerfelt
    (Variante B).

Fuer den VERGLEICH mit den anderen Phasen zaehlt NUR die echte Kostenfunktion:
die hier berechnete Naeherungs-Strategie pi*_approx wird auf der ECHTEN
ElevatorEnvV2 ausgefuehrt (echte Kapazitaet, echte Kosten), dabei aber bei
jedem Schritt der wahre Zustand auf die Boolean-Beobachtung projiziert, bevor
in der Politik-Tabelle nachgeschlagen wird (siehe project_to_observation()).
evaluate()/run_episode() aus evaluation.py liefern dabei automatisch die
ECHTEN Kosten aus ElevatorEnvV2.step() - unabhaengig davon, mit welcher
(Naeherungs-)Kostenfunktion pi*_approx berechnet wurde.

Konsequenz: V*_approx(Startzustand) (das Ergebnis DIESER Value Iteration) ist
NICHT direkt mit V*(Startzustand) aus Phase 1/2/2b vergleichbar (andere
Kostenbasis!). Die Luecke zwischen V*_approx und dem tatsaechlich in der
Simulation gemessenen avg_discounted_cost ist selbst eine interessante
Kennzahl - sie zeigt, wie stark die Naeherung die Realitaet unterschaetzt.
"""

import csv
import itertools
import os
import sys

# Projekt-Root auf sys.path, damit dieses Skript sowohl per `python -m phase3.value_iteration_3`
# als auch direkt per `python phase3/value_iteration_3.py` funktioniert (unabhaengig vom CWD).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_2.value_iteration import available_actions  # unveraendert wiederverwendet


def cost_approx(w, z):
    """Naeherungs-Kostenfunktion NUR fuer die Planung in diesem Modul - siehe
    Docstring oben. w/z sind hier bereits Booleans (0/1), also einfache Summen."""
    return sum(w) + sum(z)


def enumerate_states(n_floors):
    """Alle syntaktisch moeglichen Boolean-Beobachtungen (f, door_open, w, z)
    mit w_i, z_i in {0,1}."""
    states = []
    for f in range(1, n_floors + 1):
        for door_open in (False, True):
            for w in itertools.product((0, 1), repeat=n_floors):
                for z in itertools.product((0, 1), repeat=n_floors):
                    states.append((f, door_open, w, z))
    return states


def action_effect_distribution(f, door_open, w, z, action, n_floors):
    """Erster (bei O stochastischer) Teil des Uebergangs, noch OHNE den
    Ankunftsprozess. Liefert [(prob, (f2, door2, w_mid, z_mid)), ...].

    Bei O: "wartet jemand" (w[idx]==1) wird als GENAU eine Person behandelt,
    die einsteigt und deren Ziel gleichverteilt unter den n_floors-1 anderen
    Etagen gezogen wird (Variante B, wie in Phase 2b) - keine Kapazitaets-
    pruefung noetig, da im Boolean-Modell nie mehr als 1 Person/Etage
    "sichtbar" ist (dieselbe Vereinfachung wie Phase 1/2)."""
    if action == "H":
        return [(1.0, (f + 1, False, w, z))]
    if action == "R":
        return [(1.0, (f - 1, False, w, z))]
    if action == "S":
        return [(1.0, (f, False, w, z))]

    # action == "O"
    idx = f - 1
    z_after_exit = list(z)
    z_after_exit[idx] = 0                       # Aussteigen: Ziel-Flag fuer f zuruecksetzen

    if w[idx] == 0:
        return [(1.0, (f, True, w, tuple(z_after_exit)))]

    w_mid = list(w)
    w_mid[idx] = 0                              # die eine (angenommene) Person steigt ein
    w_mid = tuple(w_mid)

    targets = [j for j in range(1, n_floors + 1) if j != f]
    prob_each = 1.0 / len(targets)
    dist = []
    for t in targets:
        z_new = list(z_after_exit)
        z_new[t - 1] = 1
        dist.append((prob_each, (f, True, w_mid, tuple(z_new))))
    return dist


def arrival_distribution(w_mid, p, n_floors):
    """Verteilung ueber moegliche w' nach dem stochastischen Ankunftsprozess.
    Boolean-Version: ist w_mid[i] schon 1, bleibt es 1 (kann nicht "mehr"
    werden, da nicht beobachtbar); sonst mit Wahrscheinlichkeit p auf 1."""
    per_floor = []
    for i in range(n_floors):
        if w_mid[i] == 1:
            per_floor.append([(1.0, 1)])
        else:
            per_floor.append([(1 - p, 0), (p, 1)])

    result = []
    for combo in itertools.product(*per_floor):
        prob = 1.0
        w_next = []
        for pr, val in combo:
            prob *= pr
            w_next.append(val)
        if prob > 0:
            result.append((prob, tuple(w_next)))
    return result


def value_iteration(n_floors=4, p=0.1, gamma=0.9, theta=1e-4, max_iterations=500, verbose=True):
    states = enumerate_states(n_floors)
    V = {s: 0.0 for s in states}

    if verbose:
        print(f"Zustandsraumgroesse (Naeherungs-MDP): {len(states)}")

    transitions = {}
    actions_of = {}
    for s in states:
        f, door_open, w, z = s
        actions = available_actions(f, door_open, n_floors)
        actions_of[s] = actions
        for a in actions:
            mid_dist = action_effect_distribution(f, door_open, w, z, a, n_floors)
            combined = []
            for prob_mid, (f2, d2, w_mid, z2) in mid_dist:
                for prob_arr, w_next in arrival_distribution(w_mid, p, n_floors):
                    combined.append((prob_mid * prob_arr, (f2, d2, w_next, z2)))
            transitions[(s, a)] = combined

    policy = {}
    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        newV = {}
        for s in states:
            f, door_open, w, z = s
            c = cost_approx(w, z)
            best_value, best_action = None, None
            for a in actions_of[s]:
                expected_future = sum(prob * V[s2] for prob, s2 in transitions[(s, a)])
                value = c + gamma * expected_future
                if best_value is None or value < best_value:
                    best_value, best_action = value, a
            newV[s] = best_value
            policy[s] = best_action
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if verbose:
            print(f"Iteration {iteration:>3}: max|V_neu - V_alt| = {delta:.6f}")
        if delta < theta:
            if verbose:
                print(f"Konvergiert nach {iteration} Iterationen.")
            break
    return V, policy


def _sort_key(s):
    f, door_open, w, z = s
    return (f, door_open, w, z)


def export_policy_csv(V, policy, path="policy.csv"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["f", "door_open", "w_bool", "z_bool", "V_approx", "action"])
        for s in sorted(V.keys(), key=_sort_key):
            f, door_open, w, z = s
            writer.writerow([f, door_open, w, z, f"{V[s]:.4f}", policy[s]])
    print(f"\nVollstaendige Naeherungs-Strategie ({len(V)} Zustaende) geschrieben nach: {path}")


def project_to_observation(state):
    """Bildet den WAHREN ElevatorEnvV2-Zustand (f, door_open, w_counts, z_counts)
    auf die Boolean-Beobachtung ab, mit der pi*_approx nachschlaegt - das ist
    der eigentliche Informationsverlust von Phase 3."""
    f, door_open, w, z = state
    return (f, door_open, tuple(1 if x > 0 else 0 for x in w), tuple(1 if x > 0 else 0 for x in z))


def make_table_policy_v3(policy):
    """policy_fn(env) -> Aktion. env ist die ECHTE ElevatorEnvV2 - der wahre
    Zustand wird vor dem Tabellen-Lookup auf die Beobachtung projiziert."""

    def policy_fn(env):
        return policy[project_to_observation(env.get_state())]

    return policy_fn


if __name__ == "__main__":
    import random

    from phase2b.elevator_environment_2 import ElevatorEnvV2
    from common.evaluation import append_result, evaluate

    n_floors = 4
    p = 0.1
    gamma = 0.9
    capacity = 5
    max_wait_per_floor = 3  # Referenzwert aus Phase 2b - fuer direkten Vergleich beibehalten
    EVAL_STEPS = 150
    EVAL_EPISODES = 20

    V_approx, policy = value_iteration(n_floors=n_floors, p=p, gamma=gamma, theta=1e-4)

    start_obs = (1, False, tuple([0] * n_floors), tuple([0] * n_floors))
    print(f"\nV*_approx(Startzustand) = {V_approx[start_obs]:.4f}  "
          f"(Naeherungs-Kosten, NICHT vergleichbar mit V* aus Phase 1/2/2b - siehe Docstring)")
    print(f"pi*_approx(Startzustand) = {policy[start_obs]}")

    export_policy_csv(V_approx, policy, path=f"results/policy_phase3_boolean_obs_p{p:.2f}.csv")

    # Auswertung auf der ECHTEN Umgebung (Phase 2b, K=3 Referenzwert) - echte Kosten,
    # Politik bekommt nur die Boolean-Projektion des wahren Zustands zu sehen
    env_factory = lambda seed: ElevatorEnvV2(
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor, seed=seed,
    )
    vi_result = evaluate(
        label="phase3_boolean_obs",
        policy_fn_factory=lambda seed: make_table_policy_v3(policy),
        env_factory=env_factory, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor,
    )
    append_result(vi_result)

    print(f"\nPhase 3 (Boolean-Beobachtung) auf der echten Umgebung (K={max_wait_per_floor}):")
    print(f"  avg_cost_per_step={vi_result['avg_cost_per_step']}  "
          f"avg_discounted_cost={vi_result['avg_discounted_cost']}  "
          f"avg_trip_duration={vi_result['avg_trip_duration']}")

    # Zum Vergleich die passende Phase-2b-Zeile (gleiches p/K) aus results/runs.csv lesen,
    # statt Werte hier hart zu kodieren (die sich mit jedem K-Wechsel aendern wuerden)
    import csv as _csv
    try:
        with open("results/runs.csv", newline="", encoding="utf-8") as f_in:
            for row in _csv.DictReader(f_in):
                if (row["label"] == "phase2b_vi_realistic" and float(row["p"]) == p
                        and int(row["max_wait_per_floor"]) == max_wait_per_floor):
                    print(f"  Zum Vergleich Phase 2b VI (K={max_wait_per_floor}, volle Info): "
                          f"avg_cost_per_step={row['avg_cost_per_step']}  "
                          f"avg_discounted_cost={row['avg_discounted_cost']}  "
                          f"avg_trip_duration={row['avg_trip_duration']}")
                    break
    except FileNotFoundError:
        pass