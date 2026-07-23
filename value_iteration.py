"""
Value Iteration fuer das Aufzug-MDP aus elevator_environment.py
================================================================

Berechnet die optimale Kostenfunktion V* und eine optimale (kostenminimale)
Politik pi* per Rueckwaertsinduktion (Bellman-Optimalitaetsgleichung fuer
Kostenminimierung, analog W1_03):

    V*(s) = min_{a in A(s)} [ c(s,a) + gamma * sum_s' P(s'|s,a) * V*(s') ]

Unterschied zu ElevatorEnv: dort wird der stochastische Ankunftsprozess per
Zufallszahlengenerator SIMULIERT (eine Stichprobe pro step()). Value Iteration
braucht dagegen das vollstaendige Modell, also P(s'|s,a) als Verteilung -
deshalb wird der Uebergang hier separat (aber semantisch identisch) als
Wahrscheinlichkeitsverteilung nachgebaut.

ACHTUNG Zustandsraumgroesse: |S| = n_floors * 2 * n_floors**n_floors * 2**n_floors.
  n_floors=3 -> 1.296 Zustaende (schnell)
  n_floors=4 -> 32.768 Zustaende (spuerbar langsamer, siehe __main__)
"""

import itertools

from elevator_environment import ElevatorEnv


def available_actions(f, door_open, n_floors):
    """Identisch zu ElevatorEnv.available_actions(), nur zustandslos aufrufbar.
    f ist wie ueberall in diesem Modul die ECHTE Etagennummer (1..n_floors),
    identisch zur Konvention in ElevatorEnv."""
    if door_open:
        return ["S"]
    actions = ["O"]
    if f < n_floors:
        actions.append("H")
    if f > 1:
        actions.append("R")
    return actions


def cost(w, z):
    """c(s,a): Anzahl offener Anfragen (wartend + in Kabine), wie ElevatorEnv._cost()."""
    wartend = sum(1 for ziel in w if ziel is not None)
    in_kabine = sum(z)
    return wartend + in_kabine


def enumerate_states(n_floors):
    """Alle syntaktisch moeglichen Zustaende s = (f, door_open, w, z).
    f und die Zielwerte in w sind ECHTE Etagennummern 1..n_floors, identisch
    zur Konvention in ElevatorEnv (keine Verschiebung noetig)."""
    waiting_options_per_floor = [
        [None] + [j for j in range(1, n_floors + 1) if j != i]
        for i in range(1, n_floors + 1)
    ]
    states = []
    for f in range(1, n_floors + 1):
        for door_open in (False, True):
            for w in itertools.product(*waiting_options_per_floor):
                for z in itertools.product((0, 1), repeat=n_floors):
                    states.append((f, door_open, w, z))
    return states


def apply_action(f, door_open, w, z, action, n_floors):
    """Deterministischer Teil des Uebergangs (vor dem stochastischen Ankunftsprozess),
    identisch zur Aktionslogik in ElevatorEnv.step(). f ist die echte Etagennummer
    (1..n_floors); w/z sind 0-indizierte Listen (Index i-1 <-> Etage i), genau wie
    waiting_dest/cabin_dest in ElevatorEnv."""
    w = list(w)
    z = list(z)
    idx = f - 1
    if action == "H":
        f, door_open = f + 1, False
    elif action == "R":
        f, door_open = f - 1, False
    elif action == "S":
        door_open = False
    elif action == "O":
        door_open = True
        if z[idx] == 1:               # Aussteigen
            z[idx] = 0
        if w[idx] is not None:        # Einsteigen (Ziel schon bekannt)
            ziel = w[idx]              # echte Zieletage, 1..n_floors
            z[ziel - 1] = 1
            w[idx] = None
    return f, door_open, tuple(w), tuple(z)


def arrival_distribution(w_mid, p, n_floors):
    """Verteilung ueber moegliche w' nach dem stochastischen Ankunftsprozess.
    Liefert eine Liste von (Wahrscheinlichkeit, w')-Paaren. Die Etagen sind
    unabhaengig (Produktverteilung wie in ElevatorEnv.step()), daher
    Aufzaehlung per itertools.product ueber die Einzelverteilungen je Etage."""
    per_floor = []
    for i in range(n_floors):
        etage = i + 1                    # echte Etagennummer dieses Listenindex
        if w_mid[i] is not None:
            per_floor.append([(1.0, w_mid[i])])   # schon belegt -> bleibt unveraendert
        else:
            targets = [j for j in range(1, n_floors + 1) if j != etage]
            dist = [(1 - p, None)] + [(p / len(targets), t) for t in targets]
            per_floor.append(dist)

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


def value_iteration(n_floors=3, p=0.1, gamma=0.9, theta=1e-4, max_iterations=500, verbose=True):
    states = enumerate_states(n_floors)
    V = {s: 0.0 for s in states}

    # Uebergaenge haengen nicht von V ab -> einmalig vorberechnen statt pro Iteration neu bauen
    transitions = {}   # (s, a) -> [(prob, s'), ...]
    actions_of = {}     # s -> A(s)
    for s in states:
        f, door_open, w, z = s
        actions = available_actions(f, door_open, n_floors)
        actions_of[s] = actions
        for a in actions:
            f2, d2, w_mid, z2 = apply_action(f, door_open, w, z, a, n_floors)
            transitions[(s, a)] = [
                (prob, (f2, d2, w_next, z2))
                for prob, w_next in arrival_distribution(w_mid, p, n_floors)
            ]

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
    """Sortierschluessel fuer Zustaende: None (in w) wird als -1 behandelt,
    damit sortiert() nicht None mit int vergleicht (TypeError)."""
    f, door_open, w, z = s
    w_key = tuple(-1 if x is None else x for x in w)
    return (f, door_open, w_key, z)


def print_policy(V, policy):
    """Gibt V*(s) und die greedy (optimale) Aktion pi*(s) fuer ALLE Zustaende aus."""
    print(f"\nV* und pi* fuer alle {len(V)} Zustaende:")
    print(f"{'Zustand (f, Tuer, w, z)':<45} {'V*(s)':>10}   pi*(s)")
    print("-" * 70)
    for s in sorted(V.keys(), key=_sort_key):
        f, door_open, w, z = s
        zustand_str = f"(f={f}, Tuer={'offen' if door_open else 'zu'}, w={w}, z={z})"
        print(f"{zustand_str:<45} {V[s]:>10.4f}   {policy[s]}")


def raw_state(env):
    """Zustand im gleichen (unkodierten) Format wie die Value-Iteration-Zustaende
    (f, door_open, w, z) - im Unterschied zu env.get_state(), das w fuer die
    Anzeige auf 0/Etage+1 umkodiert."""
    return (env.floor, env.door_open, tuple(env.waiting_dest), tuple(env.cabin_dest))


def simulate_policy(policy, n_floors=4, p=0.1, steps=30, seed=42):
    """Fuehrt die optimale (greedy) Politik pi* auf der echten, simulierten
    ElevatorEnv aus. Ausgabeformat identisch zu demo() in elevator_environment.py:
    t=... Aktion=... Zustand=... Kosten=... je Schritt."""
    env = ElevatorEnv(n_floors=n_floors, p=p, seed=seed)
    state = env.get_state()
    print(f"t=0  Zustand={state}")

    total_cost = 0
    for t in range(1, steps + 1):
        action = policy[raw_state(env)]
        state, cost = env.step(action)
        total_cost += cost
        print(f"t={t:<3} Aktion={action}  Zustand={state}  Kosten={cost}")

    print(f"\nGesamtkosten ueber {steps} Schritte (Politik pi*): {total_cost}")
    print(f"Abgeschlossene Fahrten (Dauer je Person): {env.completed_trips}")
    if env.completed_trips:
        avg = sum(env.completed_trips) / len(env.completed_trips)
        print(f"Durchschnittliche Dauer (Rufknopf -> Ziel erreicht): {avg:.2f} Zeitschritte")


if __name__ == "__main__":
    n_floors = 4   # 4 Etagen -> 4 moegliche Ziele fuer Wartende, 4 moegliche Kabinenziele
    V, policy = value_iteration(n_floors=n_floors, p=0.1, gamma=0.9, theta=1e-4)

    start = (1, False, tuple([None] * n_floors), tuple([0] * n_floors))
    print(f"\nV*(Startzustand) = {V[start]:.4f}")
    print(f"pi*(Startzustand) = {policy[start]}")

    print("\nSimulation unter pi* (Ausgabeformat wie elevator_environment.py):")
    simulate_policy(policy, n_floors=n_floors, p=0.1, steps=30, seed=42)
