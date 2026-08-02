"""
Value Iteration fuer Phase 3 (reduzierte Beobachtung - die im Aufgabenblatt
tatsaechlich beschriebene Variante)
================================================================================

Aufbauend auf Phase 2b (Phases/environments/phase2b.py::ElevatorEnvV2), aber
der Agent beobachtet w_i/z_i jetzt nur noch als Boolean ("wartet dort
jemand?" / "will jemand aus der Kabine dorthin?"), keine exakten Zaehler mehr
- exakt das, was die Aufgabenstellung beschreibt ("nicht wie viele, nur ob").

WICHTIG - "streng genommen POMDP, hier als MDP approximiert" (siehe README/
PHASES.md): die WAHRE Umgebung bleibt ElevatorEnvV2 unveraendert (echte
Zaehler, echte Kapazitaet, echte Kosten c = wartend + in_kabine). Die Value
Iteration hier rechnet aber auf einem EIGENEN, kleineren Naeherungs-MDP, das
die Boolean-Beobachtung so behandelt, ALS WAERE SIE der volle Markov-Zustand
(bewusste Vereinfachung, keine echte POMDP-Loesung):

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
in der Politik-Tabelle nachgeschlagen wird (siehe state_from_env() /
project_to_observation()). evaluate()/run_episode() aus evaluation.py liefern
dabei automatisch die ECHTEN Kosten aus ElevatorEnvV2.step() - unabhaengig
davon, mit welcher (Naeherungs-)Kostenfunktion pi*_approx berechnet wurde.

Konsequenz: V*_approx(Startzustand) (das Ergebnis DIESER Value Iteration) ist
NICHT direkt mit V* aus Phase 1/2/2b vergleichbar (andere Kostenbasis). Die
Luecke zwischen V*_approx und dem tatsaechlich in der Simulation gemessenen
avg_discounted_cost ist selbst eine interessante Kennzahl - sie zeigt, wie
stark die Naeherung die Realitaet unterschaetzt.
"""

import itertools
import os
import sys

# Projekt-Root auf sys.path, damit dieses Skript sowohl per
# `python -m Phases.value_iteration.phase3` als auch direkt per
# `python Phases/value_iteration/phase3.py` funktioniert (unabhaengig vom CWD).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Phases.value_iteration.base import ValueIterationBase


class Phase3ValueIteration(ValueIterationBase):
    CSV_COLUMNS = ["f", "door_open", "w_bool", "z_bool", "V_approx", "action"]

    def enumerate_states(self):
        """Alle syntaktisch moeglichen Boolean-Beobachtungen (f, door_open, w, z)
        mit w_i, z_i in {0,1}."""
        states = []
        for f in range(1, self.n_floors + 1):
            for door_open in (False, True):
                for w in itertools.product((0, 1), repeat=self.n_floors):
                    for z in itertools.product((0, 1), repeat=self.n_floors):
                        states.append((f, door_open, w, z))
        return states

    def cost(self, w, z):
        """Naeherungs-Kostenfunktion c_approx NUR fuer die Planung in diesem
        Modul - siehe Moduldocstring. w/z sind hier bereits Booleans (0/1),
        also einfache Summen."""
        return sum(w) + sum(z)

    def _boarding_distribution(self, f, w, z):
        """"wartet jemand" (w[idx]==1) wird als GENAU eine Person behandelt,
        die einsteigt und deren Ziel gleichverteilt unter den n_floors-1
        anderen Etagen gezogen wird (Variante B, wie Phase 2b) - keine
        Kapazitaetspruefung noetig, da im Boolean-Modell nie mehr als 1
        Person/Etage "sichtbar" ist (dieselbe Vereinfachung wie Phase 1/2)."""
        idx = f - 1
        z_after_exit = list(z)
        z_after_exit[idx] = 0                       # Aussteigen: Ziel-Flag fuer f zuruecksetzen

        if w[idx] == 0:
            return [(1.0, (w, tuple(z_after_exit)))]

        w_mid = list(w)
        w_mid[idx] = 0                              # die eine (angenommene) Person steigt ein
        w_mid = tuple(w_mid)

        targets = [j for j in range(1, self.n_floors + 1) if j != f]
        prob_each = 1.0 / len(targets)
        dist = []
        for t in targets:
            z_new = list(z_after_exit)
            z_new[t - 1] = 1
            dist.append((prob_each, (w_mid, tuple(z_new))))
        return dist

    def arrival_distribution(self, w_mid):
        """Boolean-Version: ist w_mid[i] schon 1, bleibt es 1 (kann nicht "mehr"
        werden, da nicht beobachtbar); sonst mit Wahrscheinlichkeit p auf 1."""
        per_floor = []
        for i in range(self.n_floors):
            if w_mid[i] == 1:
                per_floor.append([(1.0, 1)])
            else:
                per_floor.append([(1 - self.p, 0), (self.p, 1)])

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

    def project_to_observation(self, state):
        """Bildet den WAHREN ElevatorEnvV2-Zustand (f, door_open, w_counts, z_counts)
        auf die Boolean-Beobachtung ab, mit der pi*_approx nachschlaegt - das ist
        der eigentliche Informationsverlust von Phase 3."""
        f, door_open, w, z = state
        return (f, door_open, tuple(1 if x > 0 else 0 for x in w), tuple(1 if x > 0 else 0 for x in z))

    def state_from_env(self, env):
        """env ist die ECHTE ElevatorEnvV2 - der wahre Zustand wird vor dem
        Tabellen-Lookup auf die Beobachtung projiziert."""
        return self.project_to_observation(env.get_state())


if __name__ == "__main__":
    import csv as _csv

    from Phases.common.evaluation import append_result, evaluate
    from Phases.environments.phase2b import ElevatorEnvV2

    n_floors = 4
    p = 0.1
    gamma = 0.9
    capacity = 5
    max_wait_per_floor = 3  # Referenzwert aus Phase 2b - fuer direkten Vergleich beibehalten
    EVAL_STEPS = 150
    EVAL_EPISODES = 20

    vi = Phase3ValueIteration(n_floors=n_floors, p=p, gamma=gamma, theta=1e-4)
    V_approx, policy = vi.run()

    start_obs = vi.start_state()
    print(f"\nV*_approx(Startzustand) = {V_approx[start_obs]:.4f}  "
          f"(Naeherungs-Kosten, NICHT vergleichbar mit V* aus Phase 1/2/2b - siehe Docstring)")
    print(f"pi*_approx(Startzustand) = {policy[start_obs]}")

    vi.export_policy_csv(path=f"Results/CSV/policy_phase3_boolean_obs_p{p:.2f}.csv")

    # Auswertung auf der ECHTEN Umgebung (Phase 2b, K=3 Referenzwert) - echte Kosten,
    # Politik bekommt nur die Boolean-Projektion des wahren Zustands zu sehen
    env_factory = lambda seed: ElevatorEnvV2(
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor, seed=seed,
    )
    vi_result = evaluate(
        label="phase3_boolean_obs",
        policy_fn_factory=lambda seed: vi.make_table_policy(),
        env_factory=env_factory, gamma=gamma, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=n_floors, p=p, capacity=capacity, max_wait_per_floor=max_wait_per_floor,
    )
    append_result(vi_result)

    print(f"\nPhase 3 (Boolean-Beobachtung) auf der echten Umgebung (K={max_wait_per_floor}):")
    print(f"  avg_cost_per_step={vi_result['avg_cost_per_step']}  "
          f"avg_discounted_cost={vi_result['avg_discounted_cost']}  "
          f"avg_trip_duration={vi_result['avg_trip_duration']}")

    # Zum Vergleich die passende Phase-2b-Zeile (gleiches p/K) aus Results/CSV/runs.csv lesen,
    # statt Werte hier hart zu kodieren (die sich mit jedem K-Wechsel aendern wuerden)
    try:
        with open("Results/CSV/runs.csv", newline="", encoding="utf-8") as f_in:
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
