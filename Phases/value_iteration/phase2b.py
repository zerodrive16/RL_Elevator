"""
Value Iteration fuer Phase 2b (Phases/environments/phase2b.py::ElevatorEnvV2)
================================================================================

Analog zu phase1_2.py, aber an die neue Umgebungsdynamik angepasst: mehrere
Personen pro Etage, echte Kabinenkapazitaet, Fahrtziel wird erst beim
Einsteigen gewuerfelt ("Variante B" statt "Variante A" aus Phase 1/2).

available_actions()/action_effect_distribution() fuer H/R/S sind WIRKLICH
identisch zu Phase 1/2 (haengen nur von f/door_open/n_floors ab, nicht von
der w/z-Darstellung) - kommen daher unveraendert aus ValueIterationBase.
Alles andere (enumerate_states, cost, _boarding_distribution,
arrival_distribution) ist neu, weil w_i/z_i jetzt Zaehler statt Einzelwerte
sind UND weil die Aktion "O" jetzt selbst eine Zufallskomponente hat:

    Phase 1/2: Ziel schon bekannt -> Einsteigen ist deterministisch, nur die
               NACH der Aktion neu ankommenden Personen sind zufaellig.
    Phase 2b:  Ziel wird erst beim Einsteigen gewuerfelt -> WELCHE z_j erhoeht
               werden, wenn n_board Personen gleichzeitig einsteigen, ist
               selbst eine Zufallsvariable (Multinomialverteilung ueber die
               n_floors-1 moeglichen Zielrichtungen).

Ein Zeitschritt hat deshalb hier ZWEI hintereinandergeschaltete Zufallsquellen
statt einer: (1) Einsteige-Verteilung (nur bei Aktion O relevant, siehe
_boarding_distribution), dann (2) Ankunfts-Verteilung (wie in Phase 1/2, aber
jetzt auf Zaehler-Basis). Die Gesamtuebergangswahrscheinlichkeit ist das
Produkt aus beiden (kombiniert bereits in ValueIterationBase._build_transitions()).
"""

import itertools
import os
import sys
from math import comb

# Projekt-Root auf sys.path, damit dieses Skript sowohl per
# `python -m Phases.value_iteration.phase2b` als auch direkt per
# `python Phases/value_iteration/phase2b.py` funktioniert (unabhaengig vom CWD).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Phases.value_iteration.base import ValueIterationBase


class Phase2bValueIteration(ValueIterationBase):
    def __init__(self, n_floors=4, p=0.1, gamma=0.9, capacity=5, max_wait_per_floor=2,
                 theta=1e-4, max_iterations=500):
        super().__init__(n_floors=n_floors, p=p, gamma=gamma, theta=theta, max_iterations=max_iterations)
        self.capacity = capacity                      # jetzt WIRKLICH durchgesetzt (siehe _boarding_distribution)
        self.max_wait_per_floor = max_wait_per_floor    # Deckel K, noetig fuer endlichen Zustandsraum

    def enumerate_states(self):
        """Alle syntaktisch moeglichen Zustaende s = (f, door_open, w, z).
        w_i in {0,...,K} (Warteschlangen-Deckel), z ueber _enumerate_cabin_configs
        (echte Kapazitaetsgrenze sum(z) <= capacity)."""
        w_options = list(itertools.product(range(self.max_wait_per_floor + 1), repeat=self.n_floors))
        z_options = self._enumerate_cabin_configs()
        states = []
        for f in range(1, self.n_floors + 1):
            for door_open in (False, True):
                for w in w_options:
                    for z in z_options:
                        states.append((f, door_open, w, z))
        return states

    def _enumerate_cabin_configs(self):
        """Alle z-Tupel der Laenge n_floors mit z_i >= 0 und sum(z) <= capacity
        (echte Kabinenkapazitaet). Einfachste Umsetzung: alles durchprobieren und
        filtern - bei capacity=5 sind das nur 6**n_floors Kandidaten, unkritisch."""
        return [
            z for z in itertools.product(range(self.capacity + 1), repeat=self.n_floors)
            if sum(z) <= self.capacity
        ]

    def cost(self, w, z):
        """c(s,a): Anzahl offener Anfragen. w/z sind hier bereits echte Zaehler,
        also einfache Summen (kein Filtern auf 'is not None' mehr noetig wie in Phase 1/2)."""
        return sum(w) + sum(z)

    @staticmethod
    def _compositions(total, parts):
        """Alle Tupel der Laenge `parts` aus nicht-negativen Ganzzahlen, die sich
        zu `total` aufsummieren - z.B. _compositions(2, 3) -> (2,0,0),(0,2,0),...,(1,1,0),...
        Hilfsfunktion fuer die Multinomialverteilung beim gleichzeitigen Einsteigen
        mehrerer Personen (siehe _boarding_distribution)."""
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in Phase2bValueIteration._compositions(total - first, parts - 1):
                yield (first,) + rest

    @staticmethod
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

    def _boarding_distribution(self, f, w, z):
        """Aussteigen (ALLE Personen mit Kabinenziel f) + Einsteigen (so viele wie
        Platz ist, echte Kapazitaetspruefung) - Ziel wird JETZT gewuerfelt
        (Variante B), jede einsteigende Person unabhaengig und gleichverteilt
        unter den n_floors-1 anderen Etagen."""
        idx = f - 1
        z_after_exit = list(z)
        z_after_exit[idx] = 0                                    # Aussteigen: alle mit Ziel f verlassen die Kabine

        capacity_remaining = self.capacity - sum(z_after_exit)
        n_board = min(w[idx], capacity_remaining)                # so viele wie Platz ist, Rest bleibt wartend

        w_mid = list(w)
        w_mid[idx] -= n_board
        w_mid = tuple(w_mid)
        # verbleibende (w_mid[idx] > 0) Personen bleiben stehen und warten weiter -
        # genau das ist der neue, in Phase 1/2 unmoegliche Fall "Kapazitaet reicht nicht"

        if n_board == 0:
            return [(1.0, (w_mid, tuple(z_after_exit)))]

        targets = [j for j in range(1, self.n_floors + 1) if j != f]  # moegliche Zielrichtungen
        m = len(targets)  # = n_floors - 1

        result = []
        for counts in self._compositions(n_board, m):          # alle Aufteilungen von n_board auf m Ziele
            prob = self._multinomial_prob(counts, m)
            z_new = list(z_after_exit)
            for target_floor, c in zip(targets, counts):
                z_new[target_floor - 1] += c
            result.append((prob, (w_mid, tuple(z_new))))
        return result

    def arrival_distribution(self, w_mid):
        """Verteilung ueber moegliche w' nach dem stochastischen Ankunftsprozess.
        Pro Etage unabhaengig: ist die Warteschlange schon am Deckel, kommt mit
        Sicherheit niemand Neues dazu (Warteschlange kann nicht weiter wachsen);
        sonst mit Wahrscheinlichkeit p +1, sonst unveraendert."""
        per_floor = []
        for i in range(self.n_floors):
            if w_mid[i] >= self.max_wait_per_floor:
                per_floor.append([(1.0, w_mid[i])])
            else:
                per_floor.append([(1 - self.p, w_mid[i]), (self.p, w_mid[i] + 1)])

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
        return (1, False, tuple([0] * self.n_floors), tuple([0] * self.n_floors))

    def state_from_env(self, env):
        """Kein Umkodieren noetig - ElevatorEnvV2.get_state() liefert bereits
        (f, door_open, w, z) mit w/z als Zaehler, identisch zur hier verwendeten
        Zustandsdarstellung (anders als Phase 1/2)."""
        return env.get_state()


if __name__ == "__main__":
    import random

    from Phases.common.evaluation import append_result, evaluate, write_convergence_log
    from Phases.environments.phase2b import ElevatorEnvV2

    n_floors = 4
    p = 0.1
    gamma = 0.9
    capacity = 5
    max_wait_per_floor = 3  # Referenzwert (K=2 zeigte bei hoher Last einen Deckel-bedingten
                             # Ueberlastungs-Kollaps, siehe PHASES.md "Systemueberlastung" + Nachtrag)
    EVAL_STEPS = 150
    EVAL_EPISODES = 20

    vi = Phase2bValueIteration(
        n_floors=n_floors, p=p, gamma=gamma, capacity=capacity,
        max_wait_per_floor=max_wait_per_floor, theta=1e-4,
    )
    print(f"Zustandsraumgroesse: {len(vi.enumerate_states())}")
    V, policy = vi.run()

    start = vi.start_state()
    print(f"\nV*(Startzustand) = {V[start]:.4f}")
    print(f"pi*(Startzustand) = {policy[start]}")

    print("\nSimulation unter pi* (Ausgabeformat wie demo() in Phases/environments/phase2b.py):")
    vi.simulate(
        ElevatorEnvV2(n_floors=n_floors, p=p, capacity=capacity,
                      max_wait_per_floor=max_wait_per_floor, seed=42),
        steps=30,
    )

    vi.export_policy_csv(path=f"Results/CSV/policy_phase2b_vi_realistic_p{p:.2f}_K{max_wait_per_floor}.csv")

    # policy_fn(env) fuer Random bzw. Tabellen-Lookup - env.get_state() liefert bei
    # ElevatorEnvV2 direkt (f, door_open, w, z) mit w/z als Zaehler, keine Umkodierung
    # noetig (anders als bei Phase 1/2, siehe state_from_env() oben)
    def make_v2_random_policy(seed):
        rng = random.Random(seed)
        return lambda env: rng.choice(env.available_actions())

    env_factory = lambda seed: ElevatorEnvV2(
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor, seed=seed,
    )
    random_result = evaluate(
        label="phase2b_random", policy_fn_factory=make_v2_random_policy,
        env_factory=env_factory, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor,
    )
    vi_result = evaluate(
        label="phase2b_vi_realistic", policy_fn_factory=lambda seed: vi.make_table_policy(),
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
