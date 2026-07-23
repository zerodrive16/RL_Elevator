"""
Aufzugssteuerung als MDP - Schritt 1: Umgebung/Umwelt-Simulation
==================================================================

Bezug zur Vorlesung (W1_03):
- Zustand s_t = (f, d, w_1..w_4, z_1..z_4)
    f  : aktuelles Stockwerk (0..3, entspricht Etage 1..4)
    d  : Tuerstatus (offen / geschlossen)
    w_i: Ziel der wartenden Person an Etage i, 0 falls niemand wartet.
         (Ziel ist ab dem Zeitpunkt des Erscheinens bekannt - nicht erst
          beim Einsteigen - da der Agent laut Vorgabe weiss, wer wohin will)
    z_i: 1, falls eine Person in der Kabine Ziel i hat, sonst 0
- Aktionen A = {H, R, O, S}, zustandsabhaengig eingeschraenkt (A(s))
- Uebergangsfunktion s_{t+1} = f(s_t, a_t, w_t) mit Zufallsvariable w_t
  (Bernoulli(p) je Etage, + zufaelliges Ziel bei Neuankunft)
- Kostenfunktion c(s,a) = Anzahl offener Anfragen (wartend + in Kabine)

Zusaetzlich (fuer die geforderte Zeitmessung, KEIN Teil des MDP-Zustands,
sondern reines Auswertungs-/Logging-Bookkeeping):
- Pro Person wird der Zeitpunkt des Erscheinens (Rufknopf-Druck) gemerkt.
- Bei Ankunft am Ziel wird die Gesamtdauer (Warten + Fahrt) geloggt.
"""

import random


class ElevatorEnv:
    def __init__(self, n_floors=4, p=0.05, capacity=5, seed=None):
        self.n_floors = n_floors
        self.p = p
        self.capacity = capacity  # bindet in dieser Stufe praktisch nie (max. 1 Pers./Ziel)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.floor = 0
        self.door_open = False
        self.waiting_dest = [None] * self.n_floors   # w_i: Ziel oder None
        self.cabin_dest = [0] * self.n_floors         # z_i in {0,1}
        self.waiting_start = [None] * self.n_floors   # Zeitpunkt des Erscheinens je Etage
        self.cabin_start = [None] * self.n_floors     # Startzeit der Person mit Ziel i (in Kabine)
        self.t = 0                                     # aktueller Zeitschritt
        self.completed_trips = []                      # Log: Gesamtdauer jeder abgeschlossenen Fahrt
        return self.get_state()

    def get_state(self):
        w = tuple(0 if d is None else d + 1 for d in self.waiting_dest)  # 0=None, sonst Etage+1
        return (self.floor, self.door_open, w, tuple(self.cabin_dest))

    def available_actions(self):
        if self.door_open:
            return ["S"]
        actions = ["O"]
        if self.floor < self.n_floors - 1:
            actions.append("H")
        if self.floor > 0:
            actions.append("R")
        return actions

    def step(self, action):
        if action not in self.available_actions():
            raise ValueError(f"Aktion {action} in Zustand {self.get_state()} nicht verfuegbar")

        cost = self._cost()

        if action == "H":
            self.floor += 1
            self.door_open = False
        elif action == "R":
            self.floor -= 1
            self.door_open = False
        elif action == "S":
            self.door_open = False
        elif action == "O":
            self.door_open = True
            f = self.floor

            # Aussteigen: Person mit Ziel f verlaesst die Kabine -> Fahrtdauer loggen
            if self.cabin_dest[f] == 1:
                dauer = self.t - self.cabin_start[f]
                self.completed_trips.append(dauer)
                self.cabin_dest[f] = 0
                self.cabin_start[f] = None

            # Einsteigen: Ziel ist bereits bekannt (kein neues Wuerfeln!)
            #
            # VEREINFACHUNG (0-1-Deckel je Zielrichtung): Da das Ziel schon
            # beim Erscheinen der Person gezogen wird (nicht erst beim
            # Einsteigen), kann es vorkommen, dass zwei verschiedene wartende
            # Personen (an unterschiedlichen Etagen) zufaellig dasselbe Ziel
            # haben. Steigt in diesem Fall eine zweite Person mit demselben
            # Ziel ein, waehrend cabin_dest[ziel] bereits 1 ist, wird die
            # Startzeit der ERSTEN Person ueberschrieben - die erste Fahrt
            # wird dann nie geloggt (sie "verschwindet" formal aus dem
            # Zustand). Das ist eine bewusste Vereinfachung dieser ersten
            # Stufe (Kapazitaet 5 wird dadurch ohnehin nie ausgenutzt) und
            # sollte im Bericht so benannt werden.
            if self.waiting_dest[f] is not None:
                ziel = self.waiting_dest[f]
                self.cabin_dest[ziel] = 1
                self.cabin_start[ziel] = self.waiting_start[f]  # Startzeit bleibt erhalten
                self.waiting_dest[f] = None
                self.waiting_start[f] = None

        # stochastischer Ankunftsprozess: pro Etage unabhaengig, Ziel sofort bekannt
        for i in range(self.n_floors):
            if self.waiting_dest[i] is None and self.rng.random() < self.p:
                ziel_kandidaten = [j for j in range(self.n_floors) if j != i]
                self.waiting_dest[i] = self.rng.choice(ziel_kandidaten)
                self.waiting_start[i] = self.t

        self.t += 1
        return self.get_state(), cost

    def _cost(self):
        wartend = sum(1 for d in self.waiting_dest if d is not None)
        in_kabine = sum(self.cabin_dest)
        return wartend + in_kabine


def demo(steps=100, p=0.05, seed=42):
    env = ElevatorEnv(p=p, seed=seed)
    rng = random.Random(seed)
    state = env.get_state()
    print(f"t=0  Zustand={state}")

    total_cost = 0
    for t in range(1, steps + 1):
        action = rng.choice(env.available_actions())
        state, cost = env.step(action)
        total_cost += cost
        print(f"t={t:<3} Aktion={action}  Zustand={state}  Kosten={cost}")

    print(f"\nGesamtkosten ueber {steps} Schritte: {total_cost}")
    print(f"Abgeschlossene Fahrten (Dauer je Person): {env.completed_trips}")
    if env.completed_trips:
        avg = sum(env.completed_trips) / len(env.completed_trips)
        print(f"Durchschnittliche Dauer (Rufknopf -> Ziel erreicht): {avg:.2f} Zeitschritte")


if __name__ == "__main__":
    demo()