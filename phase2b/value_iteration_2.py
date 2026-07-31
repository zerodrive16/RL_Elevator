"""
Value Iteration fuer Phase 2b (elevator_environment_2.py / ElevatorEnvV2)
==========================================================================

Analog zu value_iteration.py (Phase 1/2), aber an die neue Umgebungsdynamik
angepasst: mehrere Personen pro Etage, echte Kabinenkapazitaet, Fahrtziel
wird erst beim Einsteigen gewuerfelt (Variante B statt Variante A).

available_actions() ist WIRKLICH identisch zu Phase 1/2 (haengt nur von
f/door_open/n_floors ab, nicht von der w/z-Darstellung) - wird von dort
importiert statt dupliziert. Alles andere (enumerate_states, cost,
Uebergangsfunktion) ist neu, weil w_i/z_i jetzt Zaehler statt Einzelwerte
sind UND weil die Aktion "O" jetzt selbst eine Zufallskomponente hat:

    Phase 1/2: Ziel schon bekannt -> Einsteigen ist deterministisch, nur die
               NACH der Aktion neu ankommenden Personen sind zufaellig.
    Phase 2b:  Ziel wird erst beim Einsteigen gewuerfelt -> WELCHE z_j erhoeht
               werden, wenn n_board Personen gleichzeitig einsteigen, ist
               selbst eine Zufallsvariable (Multinomialverteilung ueber die
               n_floors-1 moeglichen Zielrichtungen).

Ein Zeitschritt hat deshalb hier ZWEI hintereinandergeschaltete Zufallsquellen
statt einer: (1) Einsteige-Verteilung (nur bei Aktion O relevant), dann
(2) Ankunfts-Verteilung (wie in Phase 1/2, aber jetzt auf Zaehler-Basis).
Die Gesamtuebergangswahrscheinlichkeit ist das Produkt aus beiden.
"""

import csv
import itertools
import os
import random
import sys
from math import comb

# Projekt-Root auf sys.path, damit dieses Skript sowohl per `python -m phase2b.value_iteration_2`
# als auch direkt per `python phase2b/value_iteration_2.py` funktioniert (unabhaengig vom CWD).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_2.value_iteration import available_actions  # unveraendert wiederverwendet, siehe Docstring oben
from common.evaluation import append_result, evaluate


def cost(w, z):
    """c(s,a): Anzahl offener Anfragen. w/z sind hier bereits echte Zaehler,
    also einfache Summen (kein Filtern auf 'is not None' mehr noetig wie in Phase 1/2)."""
    return sum(w) + sum(z)


def enumerate_cabin_configs(n_floors, capacity):
    """Alle z-Tupel der Laenge n_floors mit z_i >= 0 und sum(z) <= capacity
    (echte Kabinenkapazitaet). Einfachste Umsetzung: alles durchprobieren und
    filtern - bei capacity=5 sind das nur 6**n_floors Kandidaten, unkritisch."""
    return [
        z for z in itertools.product(range(capacity + 1), repeat=n_floors)
        if sum(z) <= capacity
    ]


def enumerate_states(n_floors, max_wait_per_floor, capacity):
    """Alle syntaktisch moeglichen Zustaende s = (f, door_open, w, z).
    w_i in {0,...,max_wait_per_floor} (Warteschlangen-Deckel), z wie oben
    ueber enumerate_cabin_configs (echte Kapazitaetsgrenze)."""
    w_options = list(itertools.product(range(max_wait_per_floor + 1), repeat=n_floors))
    z_options = enumerate_cabin_configs(n_floors, capacity)
    states = []
    for f in range(1, n_floors + 1):
        for door_open in (False, True):
            for w in w_options:
                for z in z_options:
                    states.append((f, door_open, w, z))
    return states


def _compositions(total, parts):
    """Alle Tupel der Laenge `parts` aus nicht-negativen Ganzzahlen, die sich
    zu `total` aufsummieren - z.B. _compositions(2, 3) -> (2,0,0),(0,2,0),...,(1,1,0),...
    Hilfsfunktion fuer die Multinomialverteilung beim gleichzeitigen Einsteigen
    mehrerer Personen (siehe boarding_distribution)."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _multinomial_prob(counts, n_categories):
    """P(genau diese Aufteilung), wenn sum(counts) Personen unabhaengig und
    gleichverteilt je eine von n_categories Zielrichtungen waehlen."""
    n = sum(counts)
    coeff = 1
    remaining = n
    for c in counts:
        coeff *= comb(remaining, c)
        remaining -= c
    return coeff * (1.0 / n_categories) ** n


def boarding_distribution(n_board, z_after_exit, origin_floor, n_floors):
    """Verteilung ueber moegliche z-Zustaende NACH dem Einsteigen von genau
    n_board Personen an Etage origin_floor, deren Ziel jeweils unabhaengig
    und gleichverteilt unter den n_floors-1 anderen Etagen gezogen wird
    (Variante B: Ziel erst beim Einsteigen bekannt). Liefert [(prob, z), ...].
    """
    if n_board == 0:
        return [(1.0, tuple(z_after_exit))]

    targets = [j for j in range(1, n_floors + 1) if j != origin_floor]  # moegliche Zielrichtungen
    m = len(targets)  # = n_floors - 1

    result = []
    for counts in _compositions(n_board, m):          # alle Aufteilungen von n_board auf m Ziele
        prob = _multinomial_prob(counts, m)
        z_new = list(z_after_exit)
        for target_floor, c in zip(targets, counts):
            z_new[target_floor - 1] += c
        result.append((prob, tuple(z_new)))
    return result


def action_effect_distribution(f, door_open, w, z, action, n_floors, capacity):
    """Erster (bei H/R/S deterministischer, bei O stochastischer) Teil des
    Uebergangs - noch OHNE den Ankunftsprozess (siehe arrival_distribution).
    Liefert [(prob, (f2, door2, w_mid, z_mid)), ...]."""
    if action == "H":
        return [(1.0, (f + 1, False, w, z))]
    if action == "R":
        return [(1.0, (f - 1, False, w, z))]
    if action == "S":
        return [(1.0, (f, False, w, z))]

    # action == "O"
    idx = f - 1
    z_after_exit = list(z)
    z_after_exit[idx] = 0                                    # Aussteigen: alle mit Ziel f verlassen die Kabine

    capacity_remaining = capacity - sum(z_after_exit)
    n_board = min(w[idx], capacity_remaining)                # so viele wie Platz ist, Rest bleibt wartend

    w_mid = list(w)
    w_mid[idx] -= n_board
    w_mid = tuple(w_mid)

    dist = boarding_distribution(n_board, z_after_exit, f, n_floors)
    return [(prob, (f, True, w_mid, z_mid)) for prob, z_mid in dist]


def arrival_distribution(w_mid, p, n_floors, max_wait_per_floor):
    """Verteilung ueber moegliche w' nach dem stochastischen Ankunftsprozess.
    Pro Etage unabhaengig: ist die Warteschlange schon am Deckel, kommt mit
    Sicherheit niemand Neues dazu (Warteschlange kann nicht weiter wachsen);
    sonst mit Wahrscheinlichkeit p +1, sonst unveraendert."""
    per_floor = []
    for i in range(n_floors):
        if w_mid[i] >= max_wait_per_floor:
            per_floor.append([(1.0, w_mid[i])])
        else:
            per_floor.append([(1 - p, w_mid[i]), (p, w_mid[i] + 1)])

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


def value_iteration(n_floors=4, p=0.1, gamma=0.9, capacity=5, max_wait_per_floor=2,
                     theta=1e-4, max_iterations=500, verbose=True):
    states = enumerate_states(n_floors, max_wait_per_floor, capacity)
    V = {s: 0.0 for s in states}

    if verbose:
        print(f"Zustandsraumgroesse: {len(states)}")

    # Uebergaenge vorab berechnen (haengen nicht von V ab), wie in value_iteration.py.
    # Gesamtwahrscheinlichkeit = Einsteige-Verteilung (nur bei O) * Ankunfts-Verteilung.
    transitions = {}
    actions_of = {}
    for s in states:
        f, door_open, w, z = s
        actions = available_actions(f, door_open, n_floors)
        actions_of[s] = actions
        for a in actions:
            mid_dist = action_effect_distribution(f, door_open, w, z, a, n_floors, capacity)
            combined = []
            for prob_mid, (f2, d2, w_mid, z2) in mid_dist:
                for prob_arr, w_next in arrival_distribution(w_mid, p, n_floors, max_wait_per_floor):
                    combined.append((prob_mid * prob_arr, (f2, d2, w_next, z2)))
            transitions[(s, a)] = combined

    # Bellman-Update, identisch zur Struktur in value_iteration.py
    policy = {}
    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        newV = {}
        for s in states:
            f, door_open, w, z = s
            c = cost(w, z)
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
        writer.writerow(["f", "door_open", "w", "z", "V", "action"])
        for s in sorted(V.keys(), key=_sort_key):
            f, door_open, w, z = s
            writer.writerow([f, door_open, w, z, f"{V[s]:.4f}", policy[s]])
    print(f"\nVollstaendige Strategie ({len(V)} Zustaende) geschrieben nach: {path}")


def raw_state(env):
    """Zustand im gleichen Format wie die Value-Iteration-Zustaende - im
    Unterschied zu Phase 1/2 hier kein Umkodieren noetig, ElevatorEnvV2.get_state()
    liefert bereits (f, door_open, w, z) mit w/z als Zaehler."""
    return env.get_state()


def simulate_policy(policy, env_cls, n_floors=4, p=0.1, capacity=5, max_wait_per_floor=2, steps=30, seed=42):
    env = env_cls(n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor, seed=seed)
    state = env.get_state()
    print(f"t=0  Zustand={state}")

    total_cost = 0
    for t in range(1, steps + 1):
        action = policy[raw_state(env)]
        state, c = env.step(action)
        total_cost += c
        print(f"t={t:<3} Aktion={action}  Zustand={state}  Kosten={c}")

    print(f"\nGesamtkosten ueber {steps} Schritte (Strategie pi*): {total_cost}")
    print(f"Abgeschlossene Fahrten (Dauer je Person): {env.completed_trips}")
    if env.completed_trips:
        avg = sum(env.completed_trips) / len(env.completed_trips)
        print(f"Durchschnittliche Dauer (Rufknopf -> Ziel erreicht): {avg:.2f} Zeitschritte")


if __name__ == "__main__":
    from phase2b.elevator_environment_2 import ElevatorEnvV2

    n_floors = 4
    p = 0.1
    gamma = 0.9
    capacity = 5
    max_wait_per_floor = 3  # Referenzwert (K=2 zeigte bei hoher Last einen Deckel-bedingten
                             # Ueberlastungs-Kollaps, siehe PHASES.md "Systemueberlastung" + Nachtrag)
    EVAL_STEPS = 150
    EVAL_EPISODES = 20

    V, policy = value_iteration(
        n_floors=n_floors, p=p, gamma=gamma, capacity=capacity, max_wait_per_floor=max_wait_per_floor, theta=1e-4,
    )

    start = (1, False, tuple([0] * n_floors), tuple([0] * n_floors))
    print(f"\nV*(Startzustand) = {V[start]:.4f}")
    print(f"pi*(Startzustand) = {policy[start]}")

    print("\nSimulation unter pi* (Ausgabeformat wie elevator_environment.py):")
    simulate_policy(policy, ElevatorEnvV2, n_floors=n_floors, p=p, capacity=capacity,
                     max_wait_per_floor=max_wait_per_floor, steps=30, seed=42)

    export_policy_csv(V, policy, path=f"results/policy_phase2b_vi_realistic_p{p:.2f}_K{max_wait_per_floor}.csv")

    # policy_fn(env) fuer Random bzw. Tabellen-Lookup - env.get_state() liefert bei
    # ElevatorEnvV2 direkt (f, door_open, w, z) mit w/z als Zaehler, keine Umkodierung
    # noetig (anders als bei Phase 1/2, siehe raw_state() oben)
    def make_v2_random_policy(seed):
        rng = random.Random(seed)
        return lambda env: rng.choice(env.available_actions())

    def make_v2_table_policy(seed):
        return lambda env: policy[env.get_state()]

    env_factory = lambda seed: ElevatorEnvV2(
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor, seed=seed,
    )
    random_result = evaluate(
        label="phase2b_random", policy_fn_factory=make_v2_random_policy,
        env_factory=env_factory, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor,
    )
    vi_result = evaluate(
        label="phase2b_vi_realistic", policy_fn_factory=make_v2_table_policy,
        env_factory=env_factory, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor,
    )
    append_result(random_result)
    append_result(vi_result)

    print(f"\nVergleich ueber {EVAL_EPISODES} Episoden x {EVAL_STEPS} Schritte (p={p}, gamma={gamma}):")
    print(f"  {'':<22}{'avg_cost_per_step':>20}{'avg_discounted_cost':>22}{'avg_trip_duration':>20}")
    for r in (random_result, vi_result):
        print(f"  {r['label']:<22}{r['avg_cost_per_step']:>20}{r['avg_discounted_cost']:>22}{r['avg_trip_duration']:>20}")
    print(f"  (zum Vergleich: V*(Startzustand) = {V[start]:.4f})")