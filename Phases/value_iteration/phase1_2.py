"""
Value Iteration fuer Phase 1/2 (Phases/environments/phase1_2.py::ElevatorEnv)
================================================================================

Berechnet die optimale Kostenfunktion V* und eine optimale (kostenminimale)
Strategie pi* per Rueckwaertsinduktion (Bellman-Optimalitaetsgleichung,
siehe Phases/value_iteration/base.py::ValueIterationBase).

Unterschied zu ElevatorEnv: dort wird der stochastische Ankunftsprozess per
Zufallszahlengenerator SIMULIERT (eine Stichprobe pro step()). Value Iteration
braucht dagegen das vollstaendige Modell, also P(s'|s,a) als Verteilung -
deshalb wird der Uebergang hier separat (aber semantisch identisch) als
Wahrscheinlichkeitsverteilung nachgebaut.

Modellbesonderheit dieser Phase ("Variante A"): das Fahrtziel einer wartenden
Person ist schon beim ERSCHEINEN bekannt (nicht erst beim Einsteigen) - w_i
speichert daher direkt die Zieletage (oder None), das Einsteigen selbst
(_boarding_distribution) ist rein deterministisch.

ACHTUNG Zustandsraumgroesse: |S| = n_floors * 2 * n_floors**n_floors * 2**n_floors.
  n_floors=3 -> 1.296 Zustaende (schnell)
  n_floors=4 -> 32.768 Zustaende (spuerbar langsamer, siehe __main__)
"""

import itertools
import os
import sys

# Projekt-Root auf sys.path, damit dieses Skript sowohl per
# `python -m Phases.value_iteration.phase1_2` als auch direkt per
# `python Phases/value_iteration/phase1_2.py` funktioniert (unabhaengig vom CWD).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Phases.value_iteration.base import ValueIterationBase


class Phase12ValueIteration(ValueIterationBase):
    def enumerate_states(self):
        """Alle syntaktisch moeglichen Zustaende s = (f, door_open, w, z).
        f und die Zielwerte in w sind ECHTE Etagennummern 1..n_floors, identisch
        zur Konvention in ElevatorEnv (keine Verschiebung noetig)."""
        # pro Etage i die moeglichen Werte fuer w_i: "niemand wartet" (None) oder
        # eine der anderen Etagen als Zielwunsch (j != i, da niemand zu sich selbst will)
        waiting_options_per_floor = [
            [None] + [j for j in range(1, self.n_floors + 1) if j != i]
            for i in range(1, self.n_floors + 1)
        ]
        # kartesisches Produkt ueber f, door_open, alle w-Kombinationen und alle
        # z-Kombinationen (0/1 je Etage) -> vollstaendiger Zustandsraum
        states = []
        for f in range(1, self.n_floors + 1):
            for door_open in (False, True):
                for w in itertools.product(*waiting_options_per_floor):
                    for z in itertools.product((0, 1), repeat=self.n_floors):
                        states.append((f, door_open, w, z))
        return states

    def cost(self, w, z):
        """c(s,a): Anzahl offener Anfragen (wartend + in Kabine), wie ElevatorEnv._cost()."""
        wartend = sum(1 for ziel in w if ziel is not None)
        in_kabine = sum(z)
        return wartend + in_kabine

    def _boarding_distribution(self, f, w, z):
        """Aussteigen (Ziel f verlaesst Kabine) + Einsteigen - deterministisch,
        da das Fahrtziel der wartenden Person schon bekannt ist (kein neues
        Wuerfeln noetig), identisch zur Aktionslogik in ElevatorEnv.step()."""
        idx = f - 1
        w = list(w)
        z = list(z)
        if z[idx] == 1:                # Aussteigen: Person mit Ziel = aktuelle Etage verlaesst Kabine
            z[idx] = 0
        if w[idx] is not None:         # Einsteigen (Ziel schon bekannt, kein neues Wuerfeln noetig)
            ziel = w[idx]               # echte Zieletage, 1..n_floors
            z[ziel - 1] = 1             # Kabinenziel-Flag fuer diese Etage setzen
            w[idx] = None               # Rufknopf/Warteflag dieser Etage zuruecksetzen
        return [(1.0, (tuple(w), tuple(z)))]

    def arrival_distribution(self, w_mid):
        """Verteilung ueber moegliche w' nach dem stochastischen Ankunftsprozess.
        Liefert eine Liste von (Wahrscheinlichkeit, w')-Paaren. Die Etagen sind
        unabhaengig (Produktverteilung wie in ElevatorEnv.step()), daher
        Aufzaehlung per itertools.product ueber die Einzelverteilungen je Etage."""
        # Schritt 1: fuer JEDE Etage einzeln die Verteilung ueber ihren naechsten w-Wert bestimmen
        per_floor = []
        for i in range(self.n_floors):
            etage = i + 1                    # echte Etagennummer dieses Listenindex
            if w_mid[i] is not None:
                per_floor.append([(1.0, w_mid[i])])   # schon belegt -> bleibt unveraendert (Wahrsch. 1)
            else:
                # mit Wahrsch. (1-p) bleibt die Etage leer (None), mit Wahrsch. p erscheint
                # jemand, dessen Ziel gleichverteilt unter den anderen Etagen gezogen wird
                targets = [j for j in range(1, self.n_floors + 1) if j != etage]
                dist = [(1 - self.p, None)] + [(self.p / len(targets), t) for t in targets]
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

    def start_state(self):
        return (1, False, tuple([None] * self.n_floors), tuple([0] * self.n_floors))

    def state_from_env(self, env):
        """Zustand im gleichen (unkodierten) Format wie die Value-Iteration-Zustaende
        (f, door_open, w, z) - im Unterschied zu env.get_state(), das w fuer die
        Anzeige auf 0/Etage umkodiert."""
        return (env.floor, env.door_open, tuple(env.waiting_dest), tuple(env.cabin_dest))

    def _sort_key(self, s):
        """None (in w) wird als -1 behandelt, damit sortiert() nicht None mit
        int vergleicht (TypeError)."""
        f, door_open, w, z = s
        w_key = tuple(-1 if x is None else x for x in w)
        return (f, door_open, w_key, z)


if __name__ == "__main__":
    from Phases.common.evaluation import append_result, evaluate, make_random_policy
    from Phases.environments.phase1_2 import ElevatorEnv

    n_floors = 4   # 4 Etagen -> 4 moegliche Ziele fuer Wartende, 4 moegliche Kabinenziele
    p = 0.1        # fuer (a)/(b) fest gewaehlt, wird erst in (c) variiert
    gamma = 0.9
    EVAL_STEPS = 150       # siehe Herleitung "T-Range" - deckt gamma^T Restfehler < ~0.1% ab
    EVAL_EPISODES = 20     # unabhaengige Seeds, ueber die gemittelt wird

    # Schritt 1: V* und pi* per Value Iteration berechnen
    vi = Phase12ValueIteration(n_floors=n_floors, p=p, gamma=gamma, theta=1e-4)
    V, policy = vi.run()

    # Startzustand: Aufzug in Etage 1, Tuer zu, niemand wartet, niemand in der Kabine
    start = vi.start_state()
    print(f"\nV*(Startzustand) = {V[start]:.4f}")
    print(f"pi*(Startzustand) = {policy[start]}")

    # Schritt 2: eine einzelne Beispiel-Trajektorie unter pi* zur Illustration ausdrucken
    # (fester seed=42 -> nur EIN Zufallsverlauf, kein Mittelwert; siehe Schritt 4 dafuer)
    print("\nSimulation unter pi* (Ausgabeformat wie demo() in Phases/environments/phase1_2.py):")
    vi.simulate(ElevatorEnv(n_floors=n_floors, p=p, seed=42), steps=30)

    # Schritt 3: volle Strategie-Tabelle (Phase 2: volle Information, Ziel jeder wartenden
    # Person sofort bekannt) exportieren - phasenspezifischer Dateiname, damit spaetere
    # Phasen (reduzierte Beobachtung) diese Tabelle nicht ueberschreiben.
    vi.export_policy_csv(path=f"Results/CSV/policy_phase2_vi_full_info_p{p:.2f}.csv")

    # Schritt 4: Phase 1 (Random) vs. Phase 2 (VI), unter identischen Bedingungen
    # (gleiches n_floors/p/gamma/steps/episodes, siehe evaluate() in evaluation.py) ausgewertet
    # und beide an Results/CSV/runs.csv angehaengt, damit alle Phasen am Ende nebeneinander
    # verglichen werden koennen (nichts wird ueberschrieben, siehe append_result()).
    env_factory = lambda seed: ElevatorEnv(n_floors=n_floors, p=p, seed=seed)
    random_result = evaluate(
        label="phase1_random",
        policy_fn_factory=make_random_policy,
        env_factory=env_factory, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=n_floors, p=p,
    )
    vi_result = evaluate(
        label="phase2_vi_full_info",
        # policy ist hier bereits fertig (deterministisch) -> seed wird ignoriert,
        # die Lambda-Signatur muss aber zu policy_fn_factory(seed) passen
        policy_fn_factory=lambda seed: vi.make_table_policy(),
        env_factory=env_factory, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=n_floors, p=p,
    )
    append_result(random_result)
    append_result(vi_result)

    # Schritt 5: Kurzvergleich auf der Konsole. avg_discounted_cost ist der Durchschnitt
    # ueber die simulierten Episoden (Schritt 4, Messen) - dient als Gegenprobe zum
    # bereits exakt (Schritt 1, Lernen) berechneten V*(Startzustand), nicht dessen Quelle.
    print(f"\nVergleich ueber {EVAL_EPISODES} Episoden x {EVAL_STEPS} Schritte (p={p}, gamma={gamma}):")
    print(f"  {'':<22}{'avg_cost_per_step':>20}{'avg_discounted_cost':>22}{'avg_trip_duration':>20}")
    for r in (random_result, vi_result):
        print(f"  {r['label']:<22}{r['avg_cost_per_step']:>20}{r['avg_discounted_cost']:>22}{r['avg_trip_duration']:>20}")
    print(f"  (zum Vergleich: V*(Startzustand) = {V[start]:.4f})")
