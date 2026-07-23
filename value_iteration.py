"""
Value Iteration fuer das Aufzug-MDP aus elevator_environment.py
================================================================

Berechnet die optimale Kostenfunktion V* und eine optimale (kostenminimale)
Strategie pi* per Rueckwaertsinduktion (Bellman-Optimalitaetsgleichung fuer
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

import csv
import itertools
import os

from elevator_environment import ElevatorEnv
from evaluation import append_result, evaluate, make_random_policy, make_table_policy


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
    # pro Etage i die moeglichen Werte fuer w_i: "niemand wartet" (None) oder
    # eine der anderen Etagen als Zielwunsch (j != i, da niemand zu sich selbst will)
    waiting_options_per_floor = [
        [None] + [j for j in range(1, n_floors + 1) if j != i]
        for i in range(1, n_floors + 1)
    ]
    # kartesisches Produkt ueber f, door_open, alle w-Kombinationen (product ueber
    # waiting_options_per_floor) und alle z-Kombinationen (0/1 je Etage) -> vollstaendiger Zustandsraum
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
    idx = f - 1  # Listenindex der AKTUELLEN Etage (vor Bewegung), fuer O relevant
    if action == "H":
        f, door_open = f + 1, False        # eine Etage hoch, Tuer schliesst sich dabei
    elif action == "R":
        f, door_open = f - 1, False        # eine Etage runter
    elif action == "S":
        door_open = False                   # nur Tuer schliessen, Etage bleibt gleich
    elif action == "O":
        door_open = True
        if z[idx] == 1:               # Aussteigen: Person mit Ziel = aktuelle Etage verlaesst Kabine
            z[idx] = 0
        if w[idx] is not None:        # Einsteigen (Ziel schon bekannt, kein neues Wuerfeln noetig)
            ziel = w[idx]              # echte Zieletage, 1..n_floors
            z[ziel - 1] = 1            # Kabinenziel-Flag fuer diese Etage setzen
            w[idx] = None              # Rufknopf/Warteflag dieser Etage zuruecksetzen
    return f, door_open, tuple(w), tuple(z)


def arrival_distribution(w_mid, p, n_floors):
    """Verteilung ueber moegliche w' nach dem stochastischen Ankunftsprozess.
    Liefert eine Liste von (Wahrscheinlichkeit, w')-Paaren. Die Etagen sind
    unabhaengig (Produktverteilung wie in ElevatorEnv.step()), daher
    Aufzaehlung per itertools.product ueber die Einzelverteilungen je Etage."""
    # Schritt 1: fuer JEDE Etage einzeln die Verteilung ueber ihren naechsten w-Wert bestimmen
    per_floor = []
    for i in range(n_floors):
        etage = i + 1                    # echte Etagennummer dieses Listenindex
        if w_mid[i] is not None:
            per_floor.append([(1.0, w_mid[i])])   # schon belegt -> bleibt unveraendert (Wahrsch. 1)
        else:
            # mit Wahrsch. (1-p) bleibt die Etage leer (None), mit Wahrsch. p erscheint
            # jemand, dessen Ziel gleichverteilt unter den anderen Etagen gezogen wird
            targets = [j for j in range(1, n_floors + 1) if j != etage]
            dist = [(1 - p, None)] + [(p / len(targets), t) for t in targets]
            per_floor.append(dist)

    # Schritt 2: da die Etagen unabhaengig sind, ergibt sich die Gesamtverteilung als
    # Produkt der Einzelverteilungen - itertools.product zaehlt alle Kombinationen auf,
    # deren gemeinsame Wahrscheinlichkeit sich aus dem Produkt der Einzelwahrscheinlichkeiten ergibt
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
    V = {s: 0.0 for s in states}   # Start: V_0(s) = 0 fuer alle Zustaende (beliebige Initialisierung reicht)

    # Vorbereitung: fuer jeden Zustand s und jede erlaubte Aktion a die vollstaendige
    # Uebergangsverteilung P(s'|s,a) einmal berechnen und cachen. Das haengt nicht von V
    # ab (nur von der Modell-Dynamik) und muesste sonst in jeder der ~100 Iterationen neu
    # berechnet werden - hier also reine Performance-Optimierung.
    transitions = {}   # (s, a) -> [(prob, s'), ...]
    actions_of = {}     # s -> A(s), die in s erlaubten Aktionen
    for s in states:
        f, door_open, w, z = s
        actions = available_actions(f, door_open, n_floors)
        actions_of[s] = actions
        for a in actions:
            # deterministischer Teil der Aktion anwenden, DANACH ueber alle moeglichen
            # stochastischen Ankuenfte (w_mid -> w_next) marginalisieren
            f2, d2, w_mid, z2 = apply_action(f, door_open, w, z, a, n_floors)
            transitions[(s, a)] = [
                (prob, (f2, d2, w_next, z2))
                for prob, w_next in arrival_distribution(w_mid, p, n_floors)
            ]

    # Hauptschleife: synchrones Bellman-Update ueber ALLE Zustaende, bis sich V kaum
    # noch aendert (delta < theta). Wegen gamma<1 ist das eine Kontraktion -> Konvergenz garantiert.
    policy = {}
    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        newV = {}
        for s in states:
            f, door_open, w, z = s
            c = cost(w, z)                       # Sofortkosten c(s,a), hier unabhaengig von a
            best_value, best_action = None, None
            # ueber alle erlaubten Aktionen die mit den geringsten erwarteten Gesamtkosten waehlen (argmin)
            for a in actions_of[s]:
                expected_future = sum(prob * V[s2] for prob, s2 in transitions[(s, a)])
                value = c + gamma * expected_future   # Bellman-Optimalitaetsgleichung fuer diese Aktion
                if best_value is None or value < best_value:
                    best_value, best_action = value, a
            newV[s] = best_value      # V_{k+1}(s) = min_a [...]
            policy[s] = best_action   # gleichzeitig die aktuell beste (greedy) Aktion merken
            delta = max(delta, abs(newV[s] - V[s]))  # groesste Aenderung ueber alle Zustaende hinweg
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


def export_policy_csv(V, policy, path="policy.csv"):
    """Schreibt V*(s) und pi*(s) fuer ALLE Zustaende in eine CSV-Datei
    (eine Zeile je Zustand), zum Durchsuchen/Filtern z.B. in Excel."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["f", "door_open", "w", "z", "V", "action"])
        for s in sorted(V.keys(), key=_sort_key):
            f, door_open, w, z = s
            writer.writerow([f, door_open, w, z, f"{V[s]:.4f}", policy[s]])
    print(f"\nVollstaendige Strategie ({len(V)} Zustaende) geschrieben nach: {path}")


def raw_state(env):
    """Zustand im gleichen (unkodierten) Format wie die Value-Iteration-Zustaende
    (f, door_open, w, z) - im Unterschied zu env.get_state(), das w fuer die
    Anzeige auf 0/Etage+1 umkodiert."""
    return (env.floor, env.door_open, tuple(env.waiting_dest), tuple(env.cabin_dest))


def simulate_policy(policy, n_floors=4, p=0.1, steps=100, seed=42):
    """Fuehrt die optimale (greedy) Strategie pi* auf der echten, simulierten
    ElevatorEnv aus. Ausgabeformat identisch zu demo() in elevator_environment.py:
    t=... Aktion=... Zustand=... Kosten=... je Schritt."""
    env = ElevatorEnv(n_floors=n_floors, p=p, seed=seed)
    state = env.get_state()
    print(f"t=0  Zustand={state}")

    total_cost = 0
    for t in range(1, steps + 1):
        # reines Tabellen-Lookup: die eigentliche Optimierung ist schon in value_iteration()
        # passiert, hier wird nur noch die vorberechnete Strategie ausgefuehrt
        action = policy[raw_state(env)]
        state, cost = env.step(action)
        total_cost += cost
        print(f"t={t:<3} Aktion={action}  Zustand={state}  Kosten={cost}")

    print(f"\nGesamtkosten ueber {steps} Schritte (Strategie pi*): {total_cost}")
    print(f"Abgeschlossene Fahrten (Dauer je Person): {env.completed_trips}")
    if env.completed_trips:
        avg = sum(env.completed_trips) / len(env.completed_trips)
        print(f"Durchschnittliche Dauer (Rufknopf -> Ziel erreicht): {avg:.2f} Zeitschritte")


if __name__ == "__main__":
    n_floors = 4   # 4 Etagen -> 4 moegliche Ziele fuer Wartende, 4 moegliche Kabinenziele
    p = 0.1        # fuer (a)/(b) fest gewaehlt, wird erst in (c) variiert
    gamma = 0.9
    EVAL_STEPS = 150       # siehe Herleitung "T-Range" - deckt gamma^T Restfehler < ~0.1% ab
    EVAL_EPISODES = 20     # unabhaengige Seeds, ueber die gemittelt wird

    # Schritt 1: V* und pi* per Value Iteration berechnen (siehe value_iteration() oben)
    V, policy = value_iteration(n_floors=n_floors, p=p, gamma=gamma, theta=1e-4)

    # Startzustand: Aufzug in Etage 1, Tuer zu, niemand wartet, niemand in der Kabine
    start = (1, False, tuple([None] * n_floors), tuple([0] * n_floors))
    print(f"\nV*(Startzustand) = {V[start]:.4f}")
    print(f"pi*(Startzustand) = {policy[start]}")

    # Schritt 2: eine einzelne Beispiel-Trajektorie unter pi* zur Illustration ausdrucken
    # (fester seed=42 -> nur EIN Zufallsverlauf, kein Mittelwert; siehe Schritt 4 dafuer)
    print("\nSimulation unter pi* (Ausgabeformat wie elevator_environment.py):")
    simulate_policy(policy, n_floors=n_floors, p=p, steps=30, seed=42)

    # Schritt 3: volle Strategie-Tabelle (Phase 2: volle Information, Ziel jeder wartenden
    # Person sofort bekannt) exportieren - phasenspezifischer Dateiname, damit spaetere
    # Phasen (reduzierte Beobachtung) diese Tabelle nicht ueberschreiben.
    export_policy_csv(V, policy, path=f"results/policy_phase2_vi_full_info_p{p:.2f}.csv")

    # Schritt 4: Phase 1 (Random) vs. Phase 2 (VI), unter identischen Bedingungen
    # (gleiches n_floors/p/gamma/steps/episodes, siehe evaluate() in evaluation.py) ausgewertet
    # und beide an results/runs.csv angehaengt, damit alle Phasen am Ende nebeneinander
    # verglichen werden koennen (nichts wird ueberschrieben, siehe append_result()).
    random_result = evaluate(
        label="phase1_random",
        policy_fn_factory=make_random_policy,
        n_floors=n_floors, p=p, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
    )
    vi_result = evaluate(
        label="phase2_vi_full_info",
        # policy ist hier bereits fertig (deterministisch) -> seed wird ignoriert,
        # die Lambda-Signatur muss aber zu policy_fn_factory(seed) passen
        policy_fn_factory=lambda seed: make_table_policy(policy),
        n_floors=n_floors, p=p, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
    )
    append_result(random_result)
    append_result(vi_result)

    # Schritt 5: Kurzvergleich auf der Konsole, inkl. Gegenprobe avg_discounted_cost
    # (Monte-Carlo-Schaetzung von V*) gegen den analytisch berechneten Wert V*(Startzustand)
    print(f"\nVergleich ueber {EVAL_EPISODES} Episoden x {EVAL_STEPS} Schritte (p={p}, gamma={gamma}):")
    print(f"  {'':<22}{'avg_cost_per_step':>20}{'avg_discounted_cost':>22}{'avg_trip_duration':>20}")
    for r in (random_result, vi_result):
        print(f"  {r['label']:<22}{r['avg_cost_per_step']:>20}{r['avg_discounted_cost']:>22}{r['avg_trip_duration']:>20}")
    print(f"  (zum Vergleich: V*(Startzustand) = {V[start]:.4f})")

