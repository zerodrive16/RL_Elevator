"""
Aufzugssteuerung als MDP - Phase 2b: realistische Kapazitaet/Mehrfachwarten
============================================================================

Aufbauend auf elevator_environment.py (Phase 1/2), aber mit zwei Aenderungen
an der Umgebungsdynamik, die zusammen die noch fehlende Realitaetsluecke aus
Phase 1/2 schliessen (mehrere Personen pro Etage, echte Kabinenkapazitaet):

1) MEHRERE Personen koennen gleichzeitig an derselben Etage warten (gedeckelt
   durch max_wait_per_floor, siehe Herleitung der Zustandsraumgroesse in der
   Session-Diskussion - ohne diesen Deckel waere der Zustandsraum unendlich).
2) Die Kabinenkapazitaet (`capacity`) wird jetzt WIRKLICH durchgesetzt: passen
   nicht alle Wartenden rein, bleibt der Rest an der Etage zurueck.

WICHTIGE MODELLAENDERUNG ggue. Phase 1/2 ("Variante B" statt "Variante A"):
Das Fahrtziel einer Person wird jetzt NICHT mehr beim Erscheinen gewuerfelt,
sondern erst beim tatsaechlichen Einsteigen (Aktion O). Das ist keine
Informations-Verschleierung, sondern eine ehrlichere Modellierung des in der
Aufgabenstellung beschriebenen Mechanismus: der Rufknopf zeigt nur "hier
wartet jemand" an, das Fahrtziel wird laut Aufgabenstellung erst ueber die
Tasten "im Inneren der Kabine" angegeben - also erst beim Einsteigen bekannt.
Naebenbei loest das auch das Zustandsraum-Explosionsproblem: w_i muss dadurch
nur ein Zaehler sein (wie viele warten), kein Vektor pro Zielrichtung.

Zustand s_t = (f, d, w_1..w_n, z_1..z_n):
    f  : aktuelles Stockwerk, f in {1,...,n_floors}
    d  : Tuerstatus (offen / geschlossen)
    w_i: ANZAHL der an Etage i wartenden Personen, 0..max_wait_per_floor
         (ihr jeweiliges Fahrtziel steht noch nicht fest, siehe oben)
    z_i: ANZAHL der Personen in der Kabine mit Fahrtziel i, mit der echten
         Nebenbedingung sum(z) <= capacity

Kostenfunktion identisch zu Phase 1/2: c(s,a) = sum(w) + sum(z).

Fahrtdauer-Logging (KEIN Teil des MDP-Zustands, reines Auswertungs-
Bookkeeping wie in elevator_environment.py): da jetzt mehrere Personen
gleichzeitig an derselben Etage warten bzw. im selben Kabinenziel-"Topf"
sitzen koennen, wird pro Etage/Ziel eine FIFO-Liste von Start-Zeitpunkten
gefuehrt statt eines einzelnen Werts - beim Aus-/Einsteigen wird jeweils der
AELTESTE Eintrag entnommen (wer zuerst da war, steigt zuerst ein/aus).
"""

import random


class ElevatorEnvV2:
    def __init__(self, n_floors=4, p=0.3, capacity=5, max_wait_per_floor=2, seed=None):
        self.n_floors = n_floors
        self.p = p
        self.capacity = capacity                    # jetzt WIRKLICH durchgesetzt (siehe step())
        self.max_wait_per_floor = max_wait_per_floor  # Deckel K, noetig fuer endlichen Zustandsraum
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.floor = 1
        self.door_open = False
        self.waiting_count = [0] * self.n_floors          # w_i: Anzahl Wartender an Etage i
        self.waiting_starts = [[] for _ in range(self.n_floors)]  # FIFO-Ankunftszeiten je Etage (nur Logging)
        self.cabin_count = [0] * self.n_floors             # z_i: Anzahl Kabinenpassagiere mit Ziel i
        self.cabin_starts = [[] for _ in range(self.n_floors)]     # FIFO-Startzeiten je Kabinenziel (nur Logging)
        self.t = 0
        self.completed_trips = []
        return self.get_state()

    def get_state(self):
        # das ist bereits der VOLLSTAENDIGE, wahre Zustand (Phase 2b = "volle Info",
        # nichts wird dem Agenten hier verschwiegen - das kommt erst in Phase 3)
        return (self.floor, self.door_open, tuple(self.waiting_count), tuple(self.cabin_count))

    def available_actions(self):
        # identisch zu elevator_environment.py - Aktionsraum aendert sich durch
        # die Kapazitaets-/Mehrfachwarten-Erweiterung nicht
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
        self.t += 1

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
            idx = f - 1

            # Aussteigen: ALLE Personen mit Kabinenziel f verlassen die Kabine
            # (nicht nur eine wie in Phase 1/2 - jetzt koennen es mehrere sein)
            n_exit = self.cabin_count[idx]
            for _ in range(n_exit):
                start = self.cabin_starts[idx].pop(0)          # aeltester Eintrag zuerst
                self.completed_trips.append(self.t - start)
            self.cabin_count[idx] = 0
            self.cabin_starts[idx] = []

            # Einsteigen: so viele wie noch Platz ist (echte Kapazitaetspruefung!)
            # Ziel wird JETZT gewuerfelt (Variante B, siehe Docstring oben), nicht vorher
            capacity_remaining = self.capacity - sum(self.cabin_count)
            n_board = min(self.waiting_count[idx], capacity_remaining)
            ziel_kandidaten = [e for e in range(1, self.n_floors + 1) if e != f]
            for _ in range(n_board):
                start = self.waiting_starts[idx].pop(0)         # wer zuerst wartete, steigt zuerst ein
                ziel = self.rng.choice(ziel_kandidaten)          # jetzt erst bekannt, unabhaengig je Person
                ziel_idx = ziel - 1
                self.cabin_count[ziel_idx] += 1
                self.cabin_starts[ziel_idx].append(start)
            self.waiting_count[idx] -= n_board
            # verbleibende (waiting_count[idx] > 0) Personen bleiben stehen und warten weiter -
            # genau das ist der neue, in Phase 1/2 unmoegliche Fall "Kapazitaet reicht nicht"

        # stochastischer Ankunftsprozess: pro Etage unabhaengig, NUR falls noch Platz
        # in der Warteschlange dieser Etage ist (max_wait_per_floor-Deckel)
        for idx in range(self.n_floors):
            if self.waiting_count[idx] < self.max_wait_per_floor and self.rng.random() < self.p:
                self.waiting_count[idx] += 1
                self.waiting_starts[idx].append(self.t)

        return self.get_state(), cost

    def _cost(self):
        # c(s,a): identisch zu Phase 1/2, jetzt aber echte Summen ueber mehrere
        # Personen statt Boolean-Zaehlung
        wartend = sum(self.waiting_count)
        in_kabine = sum(self.cabin_count)
        return wartend + in_kabine


def demo(steps=100, p=0.2, max_wait_per_floor=2, seed=42):
    """Random-Baseline auf der Phase-2b-Umgebung, analog zu demo() in
    elevator_environment.py - dient als schneller Korrektheits-/Plausibilitaets-
    Check der neuen Umgebungsmechanik (Kapazitaet, Mehrfachwarten, Variante B),
    bevor darauf eine eigene Value Iteration aufgebaut wird."""
    env = ElevatorEnvV2(p=p, max_wait_per_floor=max_wait_per_floor, seed=seed)
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