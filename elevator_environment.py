"""
Aufzugssteuerung als MDP - Schritt 1: Umgebung/Umwelt-Simulation
==================================================================

Etagen-Konvention (einheitlich, KEINE Verschiebung mehr noetig):
Etagen sind durchgehend 1..4 nummeriert - sowohl self.floor als auch
alle Zielangaben (w_i, "wohin will die Person") verwenden dieselbe
Zaehlung. Intern werden Python-Listen zwar 0-indiziert gespeichert
(Index 0 = Etage 1, ..., Index 3 = Etage 4), aber die gespeicherten
WERTE (Ziel-Etage in waiting_dest, self.floor selbst) sind immer echte
Etagennummern 1..4 - nirgends muss man +1/-1 rechnen, um zu lesen,
was ein Wert bedeutet.

Bezug zur Vorlesung (W1_03):
- Zustand s_t = (f, d, w_1..w_4, z_1..z_4)
    f  : aktuelles Stockwerk, f in {1,2,3,4}
    d  : Tuerstatus (offen / geschlossen)
    w_i: Ziel-Etage der an Etage i wartenden Person (i in {1,2,3,4}),
         0 falls niemand wartet. Ziel ist ab dem Zeitpunkt des
         Erscheinens bekannt (nicht erst beim Einsteigen).
    z_i: 1, falls eine Person in der Kabine Ziel-Etage i hat, sonst 0
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
    def __init__(self, n_floors=4, p=0.3, capacity=5, seed=None):
        self.n_floors = n_floors  # Etagen 1..n_floors
        self.p = p
        self.capacity = capacity  # bindet in dieser Stufe praktisch nie (max. 1 Pers./Ziel)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.floor = 1                                # f in {1,...,n_floors}
        self.door_open = False
        self.waiting_dest = [None] * self.n_floors     # Index i-1 <-> Etage i; Wert = Ziel-Etage oder None
        self.cabin_dest = [0] * self.n_floors           # Index i-1 <-> Etage i; 1 falls jemand mit Ziel i in Kabine
        self.waiting_start = [None] * self.n_floors     # Zeitpunkt des Erscheinens je Etage
        self.cabin_start = [None] * self.n_floors       # Startzeit der Person mit Ziel i (in Kabine)
        self.t = 0                                       # aktueller Zeitschritt
        self.completed_trips = []                        # Log: Gesamtdauer jeder abgeschlossenen Fahrt
        return self.get_state()

    def get_state(self):
        w = tuple(0 if d is None else d for d in self.waiting_dest)  # d ist bereits Etage 1..4, kein +1 noetig
        return (self.floor, self.door_open, w, tuple(self.cabin_dest))

    def available_actions(self):
        if self.door_open:
            return ["S"]
        actions = ["O"]
        if self.floor < self.n_floors:
            actions.append("H")
        if self.floor > 1:
            actions.append("R")
        return actions

    def step(self, action):
        if action not in self.available_actions():
            raise ValueError(f"Aktion {action} in Zustand {self.get_state()} nicht verfuegbar")

        cost = self._cost()
        self.t += 1  # Zeit VOR der Verarbeitung erhoehen, damit self.t exakt der
                     # Nummer dieses step()-Aufrufs entspricht (t=1 beim ersten Aufruf,
                     # deckungsgleich mit der Konsolen-Zeile in demo())

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
            f = self.floor          # aktuelle Etage, 1..4
            idx = f - 1              # zugehoeriger Listen-Index, 0..3

            # Aussteigen: Person mit Ziel f verlaesst die Kabine -> Fahrtdauer loggen
            if self.cabin_dest[idx] == 1:
                dauer = self.t - self.cabin_start[idx]
                self.completed_trips.append(dauer)
                self.cabin_dest[idx] = 0
                self.cabin_start[idx] = None

            # Einsteigen: Ziel ist bereits bekannt (kein neues Wuerfeln!)
            #
            # VEREINFACHUNG (0-1-Deckel je Zielrichtung): Da das Ziel schon
            # beim Erscheinen der Person gezogen wird (nicht erst beim
            # Einsteigen), kann es vorkommen, dass zwei verschiedene wartende
            # Personen (an unterschiedlichen Etagen) zufaellig dasselbe Ziel
            # haben. Steigt in diesem Fall eine zweite Person mit demselben
            # Ziel ein, waehrend cabin_dest fuer dieses Ziel bereits 1 ist,
            # wird die Startzeit der ERSTEN Person ueberschrieben - die erste
            # Fahrt wird dann nie geloggt (sie "verschwindet" formal aus dem
            # Zustand). Das ist eine bewusste Vereinfachung dieser ersten
            # Stufe (Kapazitaet 5 wird dadurch ohnehin nie ausgenutzt) und
            # sollte im Bericht so benannt werden.
            if self.waiting_dest[idx] is not None:
                ziel = self.waiting_dest[idx]        # Ziel-Etage, 1..4
                ziel_idx = ziel - 1
                self.cabin_dest[ziel_idx] = 1
                self.cabin_start[ziel_idx] = self.waiting_start[idx]  # Startzeit bleibt erhalten
                self.waiting_dest[idx] = None
                self.waiting_start[idx] = None

        # stochastischer Ankunftsprozess: pro Etage unabhaengig, Ziel sofort bekannt
        for idx in range(self.n_floors):
            if self.waiting_dest[idx] is None and self.rng.random() < self.p:
                etage = idx + 1
                ziel_kandidaten = [e for e in range(1, self.n_floors + 1) if e != etage]
                self.waiting_dest[idx] = self.rng.choice(ziel_kandidaten)
                self.waiting_start[idx] = self.t

        return self.get_state(), cost

    def _cost(self):
        wartend = sum(1 for d in self.waiting_dest if d is not None)
        in_kabine = sum(self.cabin_dest)
        return wartend + in_kabine


def demo(steps=30, p=0.3, seed=42):
    env = ElevatorEnv(p=p, seed=seed)
    rng = random.Random(seed)
    state = env.get_state()
    print(f"t=0  Zustand={state}")

    total_cost = 0
    for _ in range(steps):
        action = rng.choice(env.available_actions())
        state, cost = env.step(action)
        total_cost += cost
        print(f"t={env.t:<3} Aktion={action}  Zustand={state}  Kosten={cost}")

    print(f"\nGesamtkosten ueber {steps} Schritte: {total_cost}")
    print(f"Abgeschlossene Fahrten (Dauer je Person): {env.completed_trips}")
    if env.completed_trips:
        avg = sum(env.completed_trips) / len(env.completed_trips)
        print(f"Durchschnittliche Dauer (Rufknopf -> Ziel erreicht): {avg:.2f} Zeitschritte")


if __name__ == "__main__":
    demo()