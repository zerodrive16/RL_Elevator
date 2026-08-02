"""
Gemeinsame Basisklasse fuer alle Value-Iteration-Varianten des Aufzug-Projekts
(Phase 1/2, Phase 2b, Phase 3).
================================================================================

Der Bellman-Update-Loop, das Exportieren der Strategie als CSV, das Ausfuehren
einer fertigen Strategie auf einer echten Umgebung ("simulate") sowie das
Erzeugen einer policy_fn(env) fuer Phases/common/evaluation.py::evaluate()
("make_table_policy") sind in allen drei Phasen strukturell identisch - nur
enumerate_states(), cost(), start_state() und die Uebergangsverteilung
unterscheiden sich (unterschiedliche Zustandsdarstellung je Phase, siehe
jeweilige Subklasse in phase1_2.py/phase2b.py/phase3.py). Diese Klasse haelt
jene gemeinsame Struktur EINMAL; Subklassen ueberschreiben nur die Methoden,
die tatsaechlich modellabhaengig sind (siehe Abschnitt "Von Subklassen zu
implementieren" unten).

Selbst innerhalb des modellabhaengigen Teils ist der Uebergang bei den
Aktionen H/R/S (Bewegung/Tuer schliessen) in allen drei Phasen identisch
(deterministisch, Etage/Tuerstatus aendert sich, w/z bleiben unveraendert) -
nur die Aktion O (Ein-/Aussteigen an der aktuellen Etage) unterscheidet sich
wirklich. `action_effect_distribution()` ist deshalb hier bereits konkret
implementiert und ruft fuer den O-Fall die abstrakte `_boarding_distribution()`
auf, die Subklassen ueberschreiben.

Bellman-Optimalitaetsgleichung fuer Kostenminimierung (Referenzpunkt fuer
`run()` unten):

    V*(s) = min_{a in A(s)} [ c(s,a) + gamma * sum_s' P(s'|s,a) * V*(s') ]
"""

import csv
import os
from abc import ABC, abstractmethod


class ValueIterationBase(ABC):
    # Von Subklassen ueberschreibbar: Spaltenkoepfe der exportierten Policy-CSV
    # (Phase 3 nennt die Spalten z.B. "w_bool"/"z_bool"/"V_approx" statt "w"/"z"/"V").
    CSV_COLUMNS = ["f", "door_open", "w", "z", "V", "action"]

    def __init__(self, n_floors=4, p=0.1, gamma=0.9, theta=1e-4, max_iterations=500):
        self.n_floors = n_floors
        self.p = p
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.V = None
        self.policy = None
        self.history = None

    # ---- von Subklassen zu implementieren (modellabhaengig) -------------------

    @abstractmethod
    def enumerate_states(self):
        """Liefert eine Liste ALLER syntaktisch moeglichen Zustaende s=(f,door_open,w,z)."""

    @abstractmethod
    def cost(self, w, z):
        """c(s,a): Sofortkosten, haengt in allen Phasen nur von (w,z) ab, nicht von a."""

    @abstractmethod
    def _boarding_distribution(self, f, w, z):
        """NUR der Aktion-O-Fall (Aus-/Einsteigen an der aktuellen Etage f) - der
        einzige Teil des Uebergangs, der sich zwischen den Phasen wirklich
        unterscheidet. Liefert [(prob, (w_mid, z_mid)), ...]; Etage/Tuerstatus
        setzt bereits action_effect_distribution() (Etage bleibt f, Tuer geht auf)."""

    @abstractmethod
    def arrival_distribution(self, w_mid):
        """Verteilung ueber moegliche w' NACH dem stochastischen Ankunftsprozess
        (nach dem Ergebnis von action_effect_distribution). Liefert [(prob, w'), ...]."""

    @abstractmethod
    def start_state(self):
        """Zustand 'leerer Aufzug in Etage 1' - Referenzpunkt fuer V*(Startzustand)
        (u.a. fuer die Konvergenz-Historie, siehe run(return_history=True))."""

    @abstractmethod
    def state_from_env(self, env):
        """Bildet den Zustand einer LEBENDEN Environment-Instanz auf die Zustands-
        darstellung ab, mit der policy/V hier indiziert sind - die Bruecke
        zwischen Simulation und Planung. In Phase 3 ist das die verlustbehaftete
        Boolean-Projektion des wahren Zustands (siehe dortige Ueberschreibung)."""

    # ---- gemeinsame, konkrete Logik (identisch fuer alle Phasen) --------------

    def available_actions(self, f, door_open):
        """A(s): identisch in allen Phasen, haengt nur von f/door_open ab."""
        if door_open:
            return ["S"]
        actions = ["O"]
        if f < self.n_floors:
            actions.append("H")
        if f > 1:
            actions.append("R")
        return actions

    def action_effect_distribution(self, f, door_open, w, z, action):
        """Erster Teil des Uebergangs (vor dem Ankunftsprozess). H/R/S sind in
        allen Phasen deterministisch und identisch; nur O ruft die
        phasenspezifische _boarding_distribution() auf."""
        if action == "H":
            return [(1.0, (f + 1, False, w, z))]
        if action == "R":
            return [(1.0, (f - 1, False, w, z))]
        if action == "S":
            return [(1.0, (f, False, w, z))]
        # action == "O"
        return [
            (prob, (f, True, w_mid, z_mid))
            for prob, (w_mid, z_mid) in self._boarding_distribution(f, w, z)
        ]

    def _sort_key(self, s):
        """Sortierschluessel fuer die CSV-Ausgabe. Standard: Zustaende sind schon
        vollstaendig vergleichbar (nur ints/bool/Tupel davon). Phase 1/2
        ueberschreibt das, weil w dort None enthalten kann (nicht mit int
        vergleichbar, sortiert() wuerde sonst TypeError werfen)."""
        return s

    def _build_transitions(self):
        """Baut fuer JEDEN Zustand/jede erlaubte Aktion die vollstaendige
        Uebergangsverteilung P(s'|s,a) (Produkt aus action_effect_distribution
        und arrival_distribution) - haengt nicht von V ab, daher einmalig vorab
        berechnet statt in jeder Bellman-Iteration neu."""
        states = self.enumerate_states()
        transitions = {}
        actions_of = {}
        for s in states:
            f, door_open, w, z = s
            actions = self.available_actions(f, door_open)
            actions_of[s] = actions
            for a in actions:
                combined = []
                for prob_mid, (f2, d2, w_mid, z2) in self.action_effect_distribution(f, door_open, w, z, a):
                    for prob_arr, w_next in self.arrival_distribution(w_mid):
                        combined.append((prob_mid * prob_arr, (f2, d2, w_next, z2)))
                transitions[(s, a)] = combined
        return states, actions_of, transitions

    def run(self, verbose=True, return_history=False):
        """Bellman-Optimalitaetsgleichung fuer Kostenminimierung (siehe Modul-
        Docstring). Synchrones Update ueber ALLE Zustaende, bis sich V kaum noch
        aendert (delta < theta) - wegen gamma<1 eine Kontraktion, Konvergenz
        garantiert. Ergebnis wird in self.V/self.policy/self.history gespeichert
        UND zurueckgegeben."""
        states, actions_of, transitions = self._build_transitions()
        V = {s: 0.0 for s in states}   # V_0(s) = 0 fuer alle Zustaende (beliebige Initialisierung reicht)
        start = self.start_state()
        history = []  # [(iteration, delta, V(Startzustand)), ...] - nur befuellt, falls return_history=True

        policy = {}
        for iteration in range(1, self.max_iterations + 1):
            delta = 0.0
            newV = {}
            for s in states:
                f, door_open, w, z = s
                c = self.cost(w, z)
                best_value, best_action = None, None
                for a in actions_of[s]:
                    expected_future = sum(prob * V[s2] for prob, s2 in transitions[(s, a)])
                    value = c + self.gamma * expected_future
                    if best_value is None or value < best_value:
                        best_value, best_action = value, a
                newV[s] = best_value      # V_{k+1}(s) = min_a [...]
                policy[s] = best_action   # gleichzeitig die aktuell beste (greedy) Aktion merken
                delta = max(delta, abs(newV[s] - V[s]))
            V = newV
            if return_history:
                history.append((iteration, delta, V[start]))
            if verbose:
                print(f"Iteration {iteration:>3}: max|V_neu - V_alt| = {delta:.6f}")
            if delta < self.theta:
                if verbose:
                    print(f"Konvergiert nach {iteration} Iterationen.")
                break

        self.V, self.policy, self.history = V, policy, history
        if return_history:
            return V, policy, history
        return V, policy

    def export_policy_csv(self, path="policy.csv"):
        """Schreibt V*(s) und pi*(s) fuer ALLE Zustaende in eine CSV-Datei
        (eine Zeile je Zustand), zum Durchsuchen/Filtern z.B. in Excel."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(self.CSV_COLUMNS)
            for s in sorted(self.V.keys(), key=self._sort_key):
                f, door_open, w, z = s
                writer.writerow([f, door_open, w, z, f"{self.V[s]:.4f}", self.policy[s]])
        print(f"\nVollstaendige Strategie ({len(self.V)} Zustaende) geschrieben nach: {path}")

    def make_table_policy(self):
        """policy_fn(env) -> Aktion, fuer Phases/common/evaluation.py::evaluate() -
        reines Tabellen-Lookup der hier berechneten Strategie. state_from_env()
        macht das unabhaengig davon, ob env-Zustand 1:1 der Planungs-Zustands-
        darstellung entspricht oder (wie in Phase 3) erst projiziert werden muss."""

        def policy_fn(env):
            return self.policy[self.state_from_env(env)]

        return policy_fn

    def simulate(self, env, steps=30):
        """Fuehrt die fertige Strategie self.policy auf einer bereits instanzierten
        echten Umgebung `env` aus und druckt die Trajektorie - Ausgabeformat
        identisch zu demo() in den Environment-Modulen (t=... Aktion=...
        Zustand=... Kosten=...)."""
        state = env.get_state()
        print(f"t=0  Zustand={state}")

        total_cost = 0
        for t in range(1, steps + 1):
            # reines Tabellen-Lookup: die eigentliche Optimierung ist schon in
            # run() passiert, hier wird nur noch die vorberechnete Strategie ausgefuehrt
            action = self.policy[self.state_from_env(env)]
            state, cost = env.step(action)
            total_cost += cost
            print(f"t={t:<3} Aktion={action}  Zustand={state}  Kosten={cost}")

        print(f"\nGesamtkosten ueber {steps} Schritte (Strategie pi*): {total_cost}")
        print(f"Abgeschlossene Fahrten (Dauer je Person): {env.completed_trips}")
        if env.completed_trips:
            avg = sum(env.completed_trips) / len(env.completed_trips)
            print(f"Durchschnittliche Dauer (Rufknopf -> Ziel erreicht): {avg:.2f} Zeitschritte")
