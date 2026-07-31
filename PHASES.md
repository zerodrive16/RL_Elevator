# Phasen des Aufzug-Projekts

Uebersicht der bisher umgesetzten Phasen (Projektvorschlag 12), jeweils
aufgeschluesselt in Umgebung / Wissen des Agenten / moegliche Aktionen /
Strategie / Zweck, sowie einer formalen MDP-Definition (Zustandsraum,
Aktionsraum, Uebergangsfunktion, Kostenfunktion) je Umgebung fuer Teil (a)
der Aufgabenstellung. Phasen 1, 2, 2b UND 3 sind implementiert; am Ende:
weitere offene Experimente (v.a. Teil (c), p-Variation).

---

## Vorlesungsbezug: Umgebungszustand, Agentenzustand, Beobachtbarkeit

(Quelle: Prof. Dr. Thomas Gabel, Vorlesung "Grundlagen adaptiver
Wissenssysteme", Folien "Der Zustand der Umgebung" / "Der Zustand eines
Agenten" / "Voll beobachtbare Umgebungen" — Foliensatz-Screenshots im Repo
unter `img.png`/`img_1.png`.) Diese Begriffe sind die formale Grundlage
dafuer, warum Phase 1/2/2b MDPs sind und Phase 3 (streng genommen) ein POMDP:

- **Umgebungszustand `s_t^e`**: die (moeglicherweise nicht oeffentlich
  verfuegbare) Repraesentation der Umwelt — enthaelt alle Informationen, die
  die Umgebung braucht, um die Verteilung der naechsten Beobachtungen/
  Bewertungen zu bestimmen. In der Praxis oft nicht vollstaendig beobachtbar
  fuer den Agenten; kann aus Agentensicht ueberfluessige/irrelevante
  Informationen enthalten.
- **Agentenzustand `s_t^a`**: die interne Repraesentation, die der Agent von
  der ihn umgebenden Welt hat — enthaelt alle Informationen, die der Agent
  fuer die Wahl seiner naechsten Aktion braucht; das ist die Information, mit
  der RL-Algorithmen tatsaechlich arbeiten. Kann eine **beliebige Funktion der
  Historie** sein: `s_t^a = f(h_t)`.
- **Volle Beobachtbarkeit**: `o_t = s_t^a = s_t^e =: s_t` — der Agent kann
  den Umgebungszustand vollstaendig wahrnehmen (Beobachtung = Agentenzustand
  = Umgebungszustand). Fuehrt formal zu **Markov'schen Entscheidungsprozessen
  (MDP)** — im Gegensatz zu partiell beobachtbaren Umgebungen (**POMDP**).

**Einordnung unserer Phasen in dieses Schema:**

| Phase | `s_t^e` (wahrer Umgebungszustand) | `o_t` (Beobachtung) | Gilt `o_t = s_t^e`? | Formal |
|---|---|---|---|---|
| 1, 2 | `(f, d, w, z)` aus `ElevatorEnv` | identisch zu `s_t^e` | Ja | MDP |
| 2b | `(f, d, w, z)` aus `ElevatorEnvV2` (echte Zaehler) | identisch zu `s_t^e` | Ja | MDP |
| 3 | `(f, d, w, z)` aus `ElevatorEnvV2` (unveraendert!) | `(f, d, ω, ζ)`, Boolean-Projektion | **Nein** | **POMDP** |

In Phase 1/2/2b gilt `o_t = s_t^a = s_t^e` — der Agent sieht den kompletten
wahren Zustand, nichts wird ihm vorenthalten. Damit sind diese drei Phasen
per Definition MDPs, exakt loesbar durch Value Iteration.

In Phase 3 gilt das nicht mehr: der Umgebungszustand `s_t^e` bleibt
unveraendert derselbe wie in Phase 2b (die Welt "kennt" weiterhin die echten
Zaehler), aber die Beobachtung `o_t` ist eine **nicht umkehrbare** Funktion
davon (`o_t = O(s_t^e)`, echte Information geht verloren — siehe unser
`w=1/2/3`-Beispiel oben). Da `o_t ≠ s_t^e`, ist das Problem laut obiger
Definition **formal ein POMDP**. Unser Agentenzustand ist dabei bewusst die
denkbar einfachste Funktion der Historie, `s_t^a = f(h_t) := o_t` (nur die
aktuelle Beobachtung, keine Akkumulation vergangener Information — z.B.
"seit wann ist der Rufknopf schon gedrueckt?" flieszt NICHT mit ein). Wir
loesen also nicht das eigentliche POMDP (das wuerde ein Belief-State ueber
`s_t^e`, geschaetzt aus der gesamten Beobachtungshistorie, erfordern), sondern
behandeln `o_t` direkt so, als waere es bereits der volle Markov-Zustand —
exakt die in "Naechste Schritte"/"Nachtrag" beschriebene, bewusste Naeherung.

---

## Phase 1 — Random-Baseline

**Code:** `phase1_2/elevator_environment.py::demo()`, ausgewertet ueber
`common/evaluation.py::make_random_policy` (Label `phase1_random` in `results/runs.csv`)

| Komponente | Beschreibung |
|---|---|
| **Umgebung** | `ElevatorEnv`, 4 Etagen, Ankunftswahrscheinlichkeit `p` pro Etage und Zeitschritt, Kapazitaet 5 (bindet in diesem Modell praktisch nie, siehe `phase1_2/elevator_environment.py:40`). |
| **Was der Agent "weiss"** | Praktisch nichts — der Agent liest nur `available_actions()` (welche Aktionen in diesem Zeitschritt *ueberhaupt gueltig* sind, abhaengig von Etage/Tuerstatus), ignoriert aber vollstaendig, wer wo wartet oder welches Kabinenziel gerade offen ist. Kein Zustand fliesst in die Entscheidung ein. |
| **Was moeglich ist (Aktionsraum)** | Wie im Aufgabenblatt: `H` (hoch), `R` (runter), `O` (Tuer oeffnen), `S` (Tuer schliessen) — eingeschraenkt auf das, was im aktuellen Zustand zulaessig ist (bei offener Tuer nur `S`, am Rand kein `H`/`R`). |
| **Strategie** | Gleichverteilt zufaellige Wahl unter den gueltigen Aktionen (`make_random_policy`, `common/evaluation.py:22-30`). Kein Lernen, keine Optimierung. |
| **Zweck** | Referenzpunkt/untere Guete-Schranke: zeigt, wie schlecht es ohne jede Steuerungslogik laeuft — der Massstab, an dem sich jede "intelligente" Strategie messen lassen muss. |

---

## Phase 2 — Value Iteration mit voller Information

**Code:** `phase1_2/value_iteration.py` (komplett), ausgewertet ueber
`common/evaluation.py::make_table_policy` (Label `phase2_vi_full_info`)

| Komponente | Beschreibung |
|---|---|
| **Umgebung** | Dieselbe `ElevatorEnv` wie Phase 1, gleiche Dynamik/Kostenfunktion. Value Iteration baut sich zusaetzlich ein *vollstaendiges* Modell der Uebergangswahrscheinlichkeiten `P(s'|s,a)` nach (`arrival_distribution()`), das `ElevatorEnv` selbst nur stichprobenartig simuliert. |
| **Was der Agent "weiss"** | **Volle Information.** Sobald eine Person an Etage `i` erscheint, ist ihr echtes Zielstockwerk sofort Teil des beobachtbaren Zustands `w_i` (nicht erst beim Einsteigen) — siehe `phase1_2/elevator_environment.py:98-99`. Der Agent kennt also nicht nur *dass* jemand wartet, sondern *wohin* diese Person will, bevor sie ueberhaupt eingestiegen ist. Das ist mehr Information, als im Aufgabenblatt fuer die reale Steuerung vorgesehen ist (dort: nur Rufknopf gedrueckt ja/nein) — bewusste Vereinfachung fuer diese Phase. |
| **Was moeglich ist (Aktionsraum)** | Identisch zu Phase 1 (`H`, `R`, `O`, `S`, zustandsabhaengig eingeschraenkt). |
| **Strategie** | Optimale (kostenminimale) Strategie `pi*`, berechnet per Value Iteration (Bellman-Optimalitaetsgleichung, `phase1_2/value_iteration.py:135-184`). Zur Laufzeit reines Tabellen-Lookup `pi*(s)` — die eigentliche "Intelligenz" steckt vollstaendig in der vorab berechneten Tabelle, nicht in einer Laufzeit-Entscheidung. |
| **Zweck** | Obere Guete-Schranke / Best Case: zeigt, was maximal erreichbar ist, *wenn* der Agent alles ueber die Zukunft wissen duerfte, was in der Welt bereits feststeht. |

### Formale MDP-Definition (Phase 1 & 2, `ElevatorEnv`)

- **Zustandsraum** `S`: `s = (f, d, w, z)` mit
  - `f ∈ {1,...,n_floors}` — aktuelles Stockwerk
  - `d ∈ {offen, geschlossen}` — Tuerstatus
  - `w = (w_1,...,w_n)`, `w_i ∈ {0} ∪ ({1,...,n_floors}\{i})` — Zieletage der (hoechstens einen) an Etage `i` wartenden Person, `0` = niemand wartet
  - `z = (z_1,...,z_n)`, `z_i ∈ {0,1}` — "mind. eine Person in der Kabine hat Ziel `i`"
  - `|S| = n_floors · 2 · n_floors^{n_floors} · 2^{n_floors}` = 32.768 bei `n_floors=4`
- **Aktionsraum** `A(s)`: `{H,R,O,S}`, zustandsabhaengig eingeschraenkt — bei `d=offen` nur `{S}`; sonst `{O} ∪ ({H}` falls `f<n_floors) ∪ ({R}` falls `f>1)`
- **Uebergangsfunktion** `P(s'|s,a)`: deterministischer Teil (Bewegung/Tuer/Ein-Aussteigen, `apply_action()`) gefolgt vom stochastischen Ankunftsprozess — je Etage unabhaengig Bernoulli(`p`), Ziel bei Ankunft sofort gleichverteilt unter den `n_floors-1` anderen Etagen gezogen (`arrival_distribution()`)
- **Kostenfunktion** `c(s,a) = Σ 1[w_i≠0] + Σ z_i` (unabhaengig von `a`)
- **Diskontierung** `gamma = 0.9`

---

## Phase 2b — Value Iteration mit realistischer Kapazitaet/Mehrfachwarten

**Code:** `phase2b/elevator_environment_2.py` (Umgebung), `phase2b/value_iteration_2.py`
(Value Iteration), ausgewertet ueber dieselbe `common/evaluation.py` wie Phase 1/2
(Label `phase2b_random` / `phase2b_vi_realistic`)

| Komponente | Beschreibung |
|---|---|
| **Umgebung** | Neue Klasse `ElevatorEnvV2`. Zwei Realitaetsluecken aus Phase 1/2 geschlossen: (1) **mehrere Personen** koennen gleichzeitig an derselben Etage warten (gedeckelt durch `max_wait_per_floor`, siehe unten), (2) die **Kabinenkapazitaet (5)** wird jetzt wirklich durchgesetzt — passen nicht alle rein, bleibt der Rest wartend zurueck (in Phase 1/2 konnte das wegen der 1-Person-Vereinfachung nie passieren). |
| **Was der Agent "weiss"** | Weiterhin **volle Information** — nichts wird versteckt, `w`/`z` sind exakte Zaehler. Aber: das Fahrtziel einer wartenden Person wird jetzt erst **beim Einsteigen** gewuerfelt statt beim Erscheinen ("Variante B" statt "Variante A" aus Phase 2) — passend zur Aufgabenstellung, die Rufknopf (nur Anwesenheit) und Zieltasten "im Inneren der Kabine" (Zielwahl) explizit trennt. Dadurch bleibt `w_i` ein einfacher Zaehler statt eines Vektors pro Zielrichtung — sonst waere der Zustandsraum nicht mehr handhabbar (siehe Diskussion unten). |
| **Was moeglich ist (Aktionsraum)** | Identisch zu Phase 1/2. |
| **Strategie** | Value Iteration wie in Phase 2, aber mit neuer Uebergangsfunktion: die Aktion `O` ist jetzt selbst teilweise stochastisch (welches `z_j` beim gleichzeitigen Einsteigen mehrerer Personen erhoeht wird, folgt einer Multinomialverteilung, siehe `phase2b/value_iteration_2.py::boarding_distribution()`), zusaetzlich zum bekannten Ankunftsprozess danach. |
| **Zweck** | Isoliert die "echte Kapazitaet"-Realitaetsluecke von der "Beobachtungs"-Realitaetsluecke (Phase 3) — zeigt, wie viel allein die Mehrfachwarten-/Kapazitaetsdynamik kostet, bevor zusaetzlich Information reduziert wird. |

**Zustandsraumgroesse & Deckel `max_wait_per_floor` (K):** Die reale Warteschlange
ist unbeschraenkt, fuer eine tabellarische Value Iteration muss sie gedeckelt
werden. Formel: `Zustandsraum(K) = 4 · 2 · (K+1)⁴ · 126` (126 = Anzahl gueltiger
Kabinenbelegungen mit `sum(z) <= 5`, siehe Herleitung in der Session-Diskussion).

| K | Zustaende | Laufzeit (gemessen) | `V*(Start, p=0.1)` |
|---|---|---|---|
| 1 | 16.128 | 15 s | 16.53 |
| 2 (frueherer Referenzwert, siehe Nachtrag unten) | 81.648 | 2:21 min | 21.23 |
| **3 (aktueller Referenzwert)** | **258.048** | **~6-9 min** | **22.29** |

**Warum `K=3` statt `K=2`:** `K=2` fuehrte bei hoeherer Last (`p≥0.3`) zu
einem Ueberlastungs-Kollaps (optimale Strategie gibt aktives Bedienen
komplett auf), der sich als groesstenteils **Artefakt des kleinen Deckels**
herausstellte, nicht als Eigenschaft der Aufgabe selbst — siehe Abschnitt
"Systemueberlastung" weiter unten fuer die vollstaendige Herleitung und
Verifikation. `K=3` ist der aktuelle Kompromiss aus Robustheit (kollabiert
bei `p=0.3` nicht mehr) und Rechenzeit; `K=4`/`K=5` sind bewusst noch nicht
getestet, bis `K=3` durchgehend nachvollzogen ist. Die `K=2`-Ergebnisse
bleiben in `results/runs.csv` als dokumentiertes Vergleichsbeispiel erhalten.

### Formale MDP-Definition (Phase 2b, `ElevatorEnvV2`)

- **Zustandsraum** `S`: `s = (f, d, w, z)` mit
  - `f ∈ {1,...,n_floors}`, `d ∈ {offen, geschlossen}` — wie oben
  - `w = (w_1,...,w_n)`, `w_i ∈ {0,...,K}` — ANZAHL der an Etage `i` wartenden Personen (`K = max_wait_per_floor`, kuenstlicher Deckel fuer Endlichkeit)
  - `z = (z_1,...,z_n)`, `z_i ≥ 0`, mit Nebenbedingung `Σ z_i ≤ capacity` (=5) — ANZAHL Kabinenpassagiere je Zieletage
  - `|S| = n_floors · 2 · (K+1)^{n_floors} · \binom{capacity+n_floors}{n_floors}` = 1008·(K+1)⁴ bei `n_floors=4, capacity=5`
- **Aktionsraum** `A(s)`: identisch zu Phase 1/2
- **Uebergangsfunktion** `P(s'|s,a)`: zweistufig —
  1. Bei `a=O`: Aussteigen (deterministisch, `z_f→0`), danach Einsteigen: `n_board = min(w_f, capacity - Σz)` Personen steigen ein, jede zieht **unabhaengig, gleichverteilt** ihr Ziel unter den `n_floors-1` anderen Etagen ("Variante B") → Multinomialverteilung ueber moegliche `z`-Konfigurationen (`boarding_distribution()`). Bei `a∈{H,R,S}` deterministisch wie in Phase 1/2.
  2. Danach: Ankunftsprozess wie in Phase 1/2, aber gedeckelt — je Etage `i`: `w_i<K` ⇒ mit Wahrscheinlichkeit `p` auf `w_i+1`, sonst bleibt `w_i` unveraendert (Warteschlange voll)
- **Kostenfunktion** `c(s,a) = Σ w_i + Σ z_i` (echte Kopfzahlen)
- **Diskontierung** `gamma = 0.9`

**`K=3` ist der aktuelle Referenzwert fuer Phase 2b/3** (siehe Begruendung
oben). `K=1`/`K=2` liefen zusaetzlich als Vergleichspunkte, `K=2` zeigt
zudem das Ueberlastungs-Kollaps-Artefakt bei hoher Last (siehe unten).

---

## Phase 3 — Value Iteration mit reduzierter (Boolean-)Beobachtung

**Code:** `phase3/value_iteration_3.py` (Naeherungs-Value-Iteration), ausgefuehrt auf
der **echten** `ElevatorEnvV2` aus Phase 2b (Label `phase3_boolean_obs`)

Das ist die im Aufgabenblatt tatsaechlich beschriebene Variante: "nicht
erkennbar, wie viele Personen warten/in der Kabine sind, nur ob". Zentral ist
hier die Trennung zwischen **wahrem Zustand** (was in der Welt real passiert)
und **Beobachtung** (was der Agent davon sieht) — vorher (Phase 1/2/2b) waren
beide identisch ("volle Information"), ab Phase 3 nicht mehr.

| Komponente | Beschreibung |
|---|---|
| **Umgebung (wahrer Zustand)** | Unveraendert `ElevatorEnvV2` aus Phase 2b — dieselbe Klasse, dieselbe Instanz-Logik, echte Kapazitaet, echte Kopfzahlen, echte Kostenfunktion. An der WELT selbst aendert sich durch Phase 3 nichts, auch die Kapazitaetspruefung (`n_board = min(waiting_count, capacity - Σcabin_count)`) sitzt weiterhin in `ElevatorEnvV2.step()` und wird unabhaengig von der jeweiligen Politik durchgesetzt — es koennen also **nie mehr als 5 Personen gleichzeitig in der Kabine sein**, egal wie "uninformiert" die Politik entscheidet (empirisch mit 2000 Simulationsschritten unter Volllast bestaetigt: maximale Belegung exakt 5). Die Politik entscheidet nur ueber `H`/`R`/`O`/`S`, nie darueber, *wie viele* einsteigen. |
| **Was der Agent "weiss" (Beobachtung)** | **Nur noch Boolean.** Der Agent beobachtet nicht `w_i`/`z_i` (echte Zaehler), sondern nur `ω_i = 1[w_i>0]` ("wartet ueberhaupt jemand an Etage i?") und `ζ_i = 1[z_i>0]` ("will ueberhaupt jemand aus der Kabine zur Etage i?"). Die Abbildung `s=(f,d,w,z) ↦ o=(f,d,ω,ζ)` ist NICHT umkehrbar — z.B. `w_i=1` und `w_i=2` erzeugen dieselbe Beobachtung `ω_i=1`. Der Agent kann diese Zustaende also nicht unterscheiden, obwohl sie sich in der Realitaet unterschiedlich weiterentwickeln (echtes POMDP: die Beobachtung ist fuer sich genommen nicht Markov). |
| **Was moeglich ist (Aktionsraum)** | Identisch zu allen bisherigen Phasen — haengt nur von `f`/`d` ab, die vollstaendig beobachtbar bleiben. |
| **Strategie** | Da eine echte POMDP-Loesung (Belief-State-Tracking) nicht Teil der Vorlesung ist, wird ein eigenes, **kleineres Naeherungs-MDP** direkt ueber dem Beobachtungsraum geloest — die Beobachtung wird bewusst so behandelt, ALS WAERE sie der volle Markov-Zustand (siehe Formale Definition unten). Die resultierende Strategie `pi*_approx(o)` wird danach auf der echten `ElevatorEnvV2` ausgefuehrt: der wahre Zustand wird bei jedem Schritt auf die Beobachtung projiziert (`project_to_observation()`), damit nachgeschlagen — aber die tatsaechlich anfallenden (echten) Kosten werden gemessen. |
| **Zweck** | Zeigt, was die im Aufgabenblatt tatsaechlich vorgesehene Informationsbeschraenkung real kostet — der direkte, faire Vergleich zu Phase 2b (identische wahre Umgebung, identisches K, nur weniger Beobachtung). |

### Formale MDP-Definition (Phase 3, Naeherungsmodell)

**Wahres System** (unveraendert Phase 2b): Zustandsraum/Uebergaenge/Kosten wie
dort. Die Beobachtungsfunktion `O: S → Ω` bildet ab via
`(f,d,w,z) ↦ (f,d, (1[w_i>0])_i, (1[z_i>0])_i )`.

**Naeherungs-MDP fuer die Planung** (eigener, in sich konsistenter, aber
approximativer Zustandsraum `Ω`, NICHT identisch mit einer korrekten
POMDP-Loesung):
- **Zustandsraum** `Ω`: `o = (f, d, ω, ζ)`, `ω_i, ζ_i ∈ {0,1}`. `|Ω| = n_floors · 2 · 2^{n_floors} · 2^{n_floors}` = 2.048 bei `n_floors=4`
- **Aktionsraum**: identisch zu allen bisherigen Phasen
- **Uebergangsfunktion (Naeherung)**: `ω_i=1` wird beim Einsteigen so behandelt, als waere es GENAU eine Person (dieselbe Vereinfachung wie Phase 1/2 — im Boolean-Modell ist "mehr als 1" nicht repraesentierbar), die ein gleichverteiltes Ziel unter den `n_floors-1` anderen Etagen zieht ("Variante B" wie Phase 2b). Ankunftsprozess: `ω_i=0` ⇒ mit Wahrscheinlichkeit `p` auf `1`; `ω_i=1` bleibt `1` (kein "mehr" beobachtbar)
- **Kostenfunktion (Naeherung, NUR fuer die Planung)**: `c_approx(o,a) = Σ ω_i + Σ ζ_i` — bewusst NICHT die echte Kostenfunktion, da der Agent echte Kopfzahlen beim Planen gar nicht kennen kann. Diese Naeherung ist der Kern der "wir loesen die Beobachtung als MDP, obwohl es eigentlich ein POMDP ist"-Vereinfachung.
- **Diskontierung** `gamma = 0.9`

**Wichtig fuer den Vergleich:** `V*_approx(Startzustand)` aus diesem Modell
ist NICHT mit `V*` aus Phase 1/2/2b vergleichbar (andere Kostenbasis). Fuer
den Phasenvergleich zaehlen ausschliesslich die *echten*, auf der wahren
`ElevatorEnvV2` gemessenen Metriken (`avg_cost_per_step`,
`avg_discounted_cost`, `avg_trip_duration` aus `evaluate()`), niemals
`c_approx`.

---

## Ergebnisse im Vergleich (p=0.1, gamma=0.9, siehe `results/runs.csv`)

| | avg_cost_per_step | avg_discounted_cost | avg_trip_duration | `V*(Start)` |
|---|---|---|---|---|
| Phase 1 (Random) | 4.31 | 20.26 | 27.72 Schritte | — |
| Phase 2 (VI, volle Info, Orakel) | 2.61 | 16.16 | 10.66 Schritte | 15.97 |
| Phase 2b Random (K=3) | 11.63 | 31.83 | 48.32 Schritte | — |
| Phase 2b VI (K=1, kein Orakel) | 2.84 | 15.74 | 12.89 Schritte | 16.53 |
| Phase 2b VI (K=2, frueherer Referenzwert) | 4.22 | 20.69 | 12.61 Schritte | 21.23 |
| **Phase 2b VI (K=3, aktueller Referenzwert)** | **4.67** | **21.94** | **12.53 Schritte** | **22.29** |
| **Phase 3 (Boolean-Beobachtung, K=3)** | **5.71** | **21.39** | **13.39 Schritte** | *(nicht vergleichbar, siehe oben)* |

Die Monte-Carlo-Schaetzung (`avg_discounted_cost`) liegt bei jeder VI-Zeile
nah am jeweiligen `V*(Startzustand)` — bestaetigt die Value-Iteration-Rechnung
unabhaengig, auch fuer das neue Modell.

**Wichtige Erkenntnis aus dem Phase-2/2b-Vergleich:** Der Sprung von Phase 2
(15.97) auf Phase 2b K=1 (16.53) — also NUR das Orakel-Vorwissen ueber das
Fahrtziel verlieren, bei sonst identischer Ein-Personen-Vereinfachung — kostet
kaum etwas (+3,5%). Der grosse Sprung kommt von K=1 auf K=2/K=3 (16.53 →
21.23 → 22.29) durch echte Kapazitaet/Mehrfachwarten. Das legt nahe: die
Kapazitaetsrealitaet treibt die Kosten deutlich staerker als der reine
Informationsverlust durchs spaetere Wuerfeln des Ziels.

**Phase 2b vs. Phase 3 bei p=0.1:** Phase 3 ist bei K=3 spuerbar schlechter
als Phase 2b (5.71 vs. 4.67 Kosten/Schritt, +22%) — anders als bei K=2, wo
beide praktisch identisch waren (4.22 vs. 4.29). Der Unterschied zeigt sich
also doch schon bei moderater Last, wurde beim kleineren Deckel (K=2) nur
verdeckt. **Bei hoeherer Last wird der Abstand noch deutlicher, siehe naechster
Abschnitt.**

---

## Teil (c): p-Variation — Ergebnisse ueber alle Phasen

**Code:** `p_variation.py` — fuehrt Phase 1, 2, 2b und 3 fuer eine Liste von
`p`-Werten durch und haengt alles an `results/runs.csv` an. Getestet:
`p ∈ {0.1, 0.15, 0.2, 0.25, 0.3, 0.5}`, Phase 2b/3 bei `K=3`
(aktueller Referenzwert, siehe oben).

| p | Phase 1 (Random) | Phase 2 (Orakel) | Phase 2b Random (K=3) | **Phase 2b VI (K=3)** | **Phase 3 (Boolean, K=3)** |
|---|---|---|---|---|---|
| 0.10 | 4.31 / 27.7 / 196 | 2.61 / 10.7 / 530 | 11.63 / 48.3 / 462 | 4.67 / 12.5 / 1059 | 5.71 / 13.4 / 967 |
| 0.15 | 4.80 / 25.3 / 216 | 3.27 / 12.0 / 547 | 12.97 / 49.9 / 494 | 6.97 / 14.0 / 1418 | 7.54 / 15.7 / 1336 |
| 0.20 | 5.15 / 25.3 / 229 | 3.71 / 12.0 / 501 | 13.87 / 50.4 / 527 | 9.08 / 15.0 / 1693 | **11.17 / — / 0** |
| 0.25 | 5.25 / 23.9 / 239 | **3.90 / — / 0** | 14.57 / 51.8 / 518 | 10.40 / 16.1 / 1818 | 11.36 / — / 0 |
| 0.30 | 5.48 / 23.4 / 259 | 3.92 / — / 0 | 14.88 / 53.6 / 542 | 11.46 / 16.3 / 1968 | 11.49 / — / 0 |
| 0.50 | 5.78 / 21.1 / 263 | 3.95 / — / 0 | 15.74 / 55.3 / 563 | **11.69 / — / 0** | 11.69 / — / 0 |

Format je Zelle weiterhin: `avg_cost_per_step / avg_trip_duration / n_completed_trips_total`.
Fett markiert wieder die jeweils erste Kollaps-Schwelle je Strategie.

Format je Zelle: `avg_cost_per_step / avg_trip_duration / n_completed_trips_total`.
Fett markiert: die jeweils erste `p`-Stufe, ab der eine Strategie **komplett
aufhoert, Fahrten abzuschliessen** (`n_completed_trips_total=0`,
`avg_trip_duration` nicht mehr definiert). Das ist kein Darstellungsfehler,
sondern ein eigenstaendiges Phaenomen — siehe naechster Abschnitt.

---

## Systemueberlastung bei hoher Last (Ueberlastungs-Kollaps)

Ab einer bestimmten `p`-Schwelle schliesst die jeweils optimale (VI-)Strategie
**gar keine Fahrten mehr ab** — nicht weil sie schlechter wird, sondern weil
sie es aufgibt, aktiv zu bedienen. Das ist ein eigenstaendiges, ueberprueftes
Ergebnis (kein Bug), das im Bericht explizit erklaert werden sollte.

### Konkretes Beispiel (p=0.5, K=2, `phase2b/value_iteration_2.py`-Politik)

Simulation ab einem gesaettigten Zustand (`w=(2,2,2,2)`, jede Etage am
Warteschlangen-Deckel, Kabine leer):

```
t=5   Aktion=H  Zustand=(4, False, (2,2,2,2), (0,0,0,0))  Kosten=8
t=6   Aktion=R  Zustand=(3, False, (2,2,2,2), (0,0,0,0))  Kosten=8
t=7   Aktion=H  Zustand=(4, False, (2,2,2,2), (0,0,0,0))  Kosten=8
t=8   Aktion=R  Zustand=(3, False, (2,2,2,2), (0,0,0,0))  Kosten=8
...   (Muster wiederholt sich unveraendert bis Simulationsende)
t=40  Aktion=R  Zustand=(3, False, (2,2,2,2), (0,0,0,0))  Kosten=8
```

Ab `t=6` pendelt der Aufzug nur noch zwischen Etage 3 und 4, **ohne die Tuer
je wieder zu oeffnen** — obwohl an allen vier Etagen Personen warten. Die
Politik waehlt an `(4, False, (2,2,2,2), (0,0,0,0))` explizit `R` statt `O`.

**Verifikation, dass das kein Bug ist:**
- Direkter Wertevergleich in der Bellman-Gleichung: `O` liefert an diesem
  Zustand einen HOEHEREN (schlechteren) Wert (93.9) als `R` (80.0) — die
  Value Iteration hat also konsistent gerechnet.
- Ein klassischer SCAN-Algorithmus (Aufzug faehrt geordnet rauf/runter, oeffnet
  immer wenn jemand da ist) wurde bei `p=0.5, K=2` ueber 500 Schritte
  gegengetestet: `avg_cost_per_step=12.87`, 0 abgeschlossene Fahrten — deutlich
  schlechter als die VI-Politik (`7.96`, ebenfalls 0 Fahrten). "Aktiv bedienen"
  ist hier also nachweislich NICHT die bessere Strategie.
- **Schrittzahl-Test:** dieselbe VI-Politik ueber 150 vs. 500 vs. 1500 Schritte
  bei `p=0.3` und `p=0.5` — Ergebnis bleibt bei 0 abgeschlossenen Fahrten,
  `avg_cost_per_step` konvergiert nur noch enger (z.B. p=0.5: 7.86 → 7.96 →
  7.99). **150 Schritte sind also nicht zu wenig** — das System erholt sich
  nie, es handelt sich um dauerhafte, nicht um voruebergehende Ueberlastung.

### Warum das inhaltlich Sinn ergibt

Die Kostenfunktion ist `wartend + in_kabine`. Jemanden einsteigen zu lassen
**reduziert die Kosten nicht sofort** — es verschiebt sie nur von "wartend"
zu "in Kabine" (beide zaehlen gleich in `c`). Bei hoher Last fuellt sich die
soeben geleerte Etage fast sofort wieder (Warteschlangen-Deckel `K` wird
staendig neu erreicht), waehrend aufgenommene Passagiere erst ueber mehrere
Schritte zu ihrem Ziel gebracht werden muessen — in der Zwischenzeit traegt
man ihre Kosten zusaetzlich mit, ohne dass sich an der Gesamtlast etwas
bessert. Sobald die Ankunftsrate (`n_floors · p`) dauerhaft ueber das hinausgeht,
was ein einziger Aufzug mit Kapazitaet 5 abarbeiten kann, ist "aktiv bedienen"
im diskontierten Erwartungswert schlechter als Schadensbegrenzung.

### Welche Experimente ab wann betroffen sind

Die Kollaps-Schwelle ist **nicht fuer alle Strategien gleich** — das ist der
eigentlich interessante Befund:

- **Random-Politiken (Phase 1 & Phase 2b Random)**: kollabieren im getesteten
  Bereich (bis p=0.5) NIE — sie "versuchen" nichts gezielt und geraten daher
  nicht in die selbstverstaerkende "nie oeffnen"-Falle; sie oeffnen per Zufall
  trotzdem ab und zu und schleusen so langsam Personen durch (wenn auch zu
  hohen Kosten).
- **Phase 2 (Orakel, `ElevatorEnv`, Ein-Personen-Deckel)**: stabil bis `p=0.20`,
  kollabiert bei `p=0.25`.
- **Phase 2b VI (K=3, aktueller Referenzwert)**: stabil bis `p=0.30`,
  kollabiert erst bei `p=0.50` (bei K=2 war die Schwelle noch `p=0.30`, siehe
  Nachtrag).
- **Phase 3 (Boolean-Beobachtung)**: stabil bis `p=0.15`, kollabiert bereits
  bei `p=0.20` — **UNVERAENDERT, egal ob die reale Umgebung mit K=2 oder K=3
  simuliert wird** (siehe Nachtrag: die Naeherungs-Politik selbst haengt gar
  nicht von K ab).

**Das ist die eigentliche Kernaussage von Phase 3:** der Informationsverlust
zeigt sich nicht (nur) als sanft steigende Durchschnittskosten, sondern als
eine **niedrigere Belastungsgrenze, ab der die Strategie komplett aufgibt**.
Mit weniger Information (nur Boolean statt echter Zaehler) trifft der Agent
schon bei geringerer Last die (aus seiner Naeherungs-Kostenfunktion `c_approx`
korrekte, aber pessimistischere) Entscheidung, dass sich Bedienen nicht mehr
lohnt — er kann eben nicht unterscheiden, ob an einer Etage 1 oder 2 Personen
warten, und bewertet die Lage dadurch fruehzeitiger als aussichtslos.

**Konsequenz fuer den weiteren Vergleich:** `avg_trip_duration` ist ab der
jeweiligen Kollaps-Schwelle nicht mehr aussagekraeftig (undefiniert). Fuer
Kurven/Plots im Bericht sollten diese Punkte explizit als "System ueberlastet,
keine Fahrten abgeschlossen" markiert werden statt als fehlende Datenpunkte
zu erscheinen.

### Nachtrag: Ist das die Kostenfunktion oder der `K`-Deckel? (`K=3` bei p=0.3 getestet)

Berechtigte Frage, die sich in der Diskussion ergab: liegt der Kollaps an der
Kostenfunktion selbst (`c = wartend + in_kabine`), oder ist er ein Artefakt
der kuenstlichen Warteschlangen-Deckelung `K`? Die Kostenfunktion ist
mathematisch aequivalent zu "minimiere Gesamt-Warte-+Fahrzeit aller Personen"
— ein Standardziel, daran liegt es vermutlich nicht.

**Testergebnis (`K=3` statt `K=2`, sonst gleiche Bedingungen, `p=0.3`):**
`n_completed_trips_total` springt von **0** (K=2) auf **1968** (K=3, ueber 20
Episoden x 150 Schritte), `avg_trip_duration=16.26`. Das System kollabiert bei
gleicher Last mit mehr Puffer **nicht mehr** — meine fruehere Einschaetzung
"`K=3` wuerde die Kollaps-Schwelle vermutlich nicht verschieben" war also
**falsch**. Grund: In der echten (unbeschraenkten) Welt wuerde eine nie
bediente Warteschlange unbegrenzt wachsen (Kosten → ∞), wodurch "nie
bedienen" niemals optimal waere. Der `K`-Deckel begrenzt diese Kosten aber
kuenstlich auf `4K+capacity` — bei kleinem `K` macht das "Aufgeben" billiger,
als es in der Realitaet waere. Ein groesseres `K` hebt diese Obergrenze an
und macht "Aufgeben" dadurch unattraktiver, ohne die Bedienkapazitaet
(`capacity=5`) selbst zu aendern.

**Wichtige methodische Konsequenz:** `avg_cost_per_step` ist **zwischen
verschiedenen `K`-Werten nicht direkt vergleichbar** — bei `K=3, p=0.3` liegt
er sogar HOEHER (11.46) als beim "aufgegebenen" `K=2`-Plateau (7.76), obwohl
das System bei `K=3` tatsaechlich funktioniert (throughput statt Stillstand).
Der Deckel bestimmt selbst mit, wie hoch Kosten ueberhaupt werden koennen.
Fuer `K`-uebergreifende Vergleiche sind `n_completed_trips_total` (>0 vs. =0)
und `avg_trip_duration` die verlaesslicheren Kennzahlen.

**Bemerkenswert:** Auch bei `K=3` waehlt die Politik am absolut vollen
Zustand (`w=(3,3,3,3)`, leere Kabine) weiterhin `R` statt `O` — das lokale
"hier lohnt sich Bedienen gerade nicht"-Verhalten existiert grundsaetzlich
weiterhin. Nur bleibt das System mit mehr Puffer nicht mehr dauerhaft in
dieser Ecke gefangen, sondern kann sie umgehen. Der Kollaps war also ein
**gradueller Effekt der Deckelgroesse**, kein Alles-oder-Nichts-Fehler in
der Modellierung.

**Bestaetigung durch den vollstaendigen `p`-Sweep bei `K=3`:** Referenzwert
wurde daraufhin von `K=2` auf `K=3` umgestellt, kompletter Sweep erneut
gerechnet (siehe Tabelle "Teil (c)" oben). Ergebnis:

- **Phase 2b VI profitiert deutlich**: Kollaps-Schwelle verschiebt sich von
  `p=0.30` (K=2) auf `p=0.50` (K=3) — bei `p=0.30` liefert K=3 jetzt 1968
  abgeschlossene Fahrten statt 0.
- **Phase 3 profitiert NICHT** — die Kollaps-Schwelle bleibt exakt bei
  `p=0.20`, egal ob die reale Umgebung mit `K=2` oder `K=3` simuliert wird.

**Warum das so ist — und warum das die Kernaussage von Phase 3 sogar
staerkt:** Die Naeherungs-Value-Iteration in `phase3/value_iteration_3.py` rechnet
auf dem Boolean-Beobachtungsraum, der `K` gar nicht kennt (kein Parameter in
`value_iteration()` dort) — die berechnete Politik ist also fuer JEDES `K`
exakt dieselbe Tabelle, nur die reale Ausfuehrungsumgebung (`ElevatorEnvV2`)
aendert sich. Phase 2b's Value Iteration dagegen bezieht `K` explizit in ihre
Uebergangsfunktion ein und kann ihre Bedien-Schwelle daher an einen groesseren
Puffer anpassen. Das heisst: **die Kollaps-Schwelle von Phase 3 ist nicht durch
mehr Kabinen-/Warteschlangenpuffer zu verschieben — nur durch mehr Information.**
Das ist ein staerkerer, robusterer Beleg fuer den Wert der Information als der
urspruengliche p=0.1-Vergleich, der durch das K=2-Artefakt noch verdeckt war.

**Einordnung fuer `K=2` als aelteren Referenzwert:** bleibt in `results/runs.csv`
und in den Tabellen oben zum Vergleich erhalten, wird aber fuer die
Interpretation/den Bericht durch `K=3` ersetzt.

### Praezisierung: was genau ist K-unabhaengig — die Entscheidung, nicht die Welt

Berechtigter Einwand aus der Diskussion: die wahre Umgebung UNTERSCHEIDET sich
zwischen `K=2` und `K=3` durchaus (real koennen 2 bzw. 3 Personen an einer
Etage warten) — wie kann das Ergebnis dann trotzdem identisch sein? Antwort:
**K-unabhaengig ist die Politik-*Entscheidung*, nicht die reale Welt.**

Grund: `project_to_observation()` bildet jeden wahren Zaehlerwert `w_i > 0`
auf dieselbe Beobachtung `ω_i = 1` ab — die Politik kann `w_i=1`, `w_i=2` und
`w_i=3` also gar nicht unterscheiden. Konkretes Beispiel (Aufzug an Etage 1,
`p=0.2`):

```
wahrer Zustand w=(0,0,1,0) -> Beobachtung (…,(0,0,1,0),…) -> Aktion: O
wahrer Zustand w=(0,0,2,0) -> Beobachtung (…,(0,0,1,0),…) -> Aktion: O   (dieselbe!)
wahrer Zustand w=(0,0,3,0) -> Beobachtung (…,(0,0,1,0),…) -> Aktion: O   (dieselbe!)
```

Alle drei wahren Zustaende sind fuer die Politik ununterscheidbar und fuehren
zwangslaeufig zur selben Aktion — daher auch: sobald die Politik bei einer
bestimmten Beobachtung "nicht mehr bedienen" entscheidet, bleibt die Tuer in
JEDER `K`-Welt zu, unabhaengig davon, ob sich dahinter 2 oder 3 Personen
stapeln. Das erklaert, warum das **binaere** Ergebnis "wird ueberhaupt jemals
jemand angeliefert?" (`n_completed_trips_total = 0` vs. `>0`) K-unabhaengig ist.

**Was sich TROTZDEM unterscheidet: die tatsaechliche Schwere des Kollapses.**
Im Hintergrund (unbeobachtet) stauen sich real trotzdem unterschiedlich viele
Personen auf, je nachdem wie hoch `K` ist — das schlaegt sich in den echten
Kosten nieder, auch wenn in beiden Faellen niemand ankommt:

| K | `avg_cost_per_step` bei p=0.2 | `n_completed_trips_total` |
|---|---|---|
| 2 | 7.58 | 0 |
| 3 | 11.17 | 0 |

Beide Male kollabiert das System vollstaendig (0 Fahrten) — aber bei `K=3`
ist es dabei real teurer, weil sich dort mehr Personen unbemerkt aufstauen
koennen. **Kurz:** K-unabhaengig ist die JA/NEIN-Frage "funktioniert das
System noch", NICHT die absolute Kostenhoehe waehrend des Kollapses.

### Ist Phase 3s Naeherung "zu schlecht", oder genau die erwartete Version?

Bei genauerer Analyse (siehe konkretes Beispiel unten) verweigert Phase 3s
Strategie bei `p=0.20` sogar das Einsteigenlassen einer **einzelnen**, direkt
vor der Tuer wartenden Person. Berechtigte Nachfrage dazu: ist das ein
Mangel unserer Modellierung, oder genau die im Aufgabenblatt vorgesehene
"schlechte Version"? Wichtig ist hier, zwei Dinge sauber zu trennen:

1. **Was der Agent beobachten darf** — nur Boolean (Rufknopf/Zieltaste
   gedrueckt, ja/nein). Das ist exakt die Aufgabenstellung, daran gibt es
   nichts zu verbessern.
2. **Wie das Planungsmodell intern mit dieser Beobachtung umgeht** — die
   Annahme "besetzt = genau 1 Person" (siehe `action_effect_distribution()`
   in `phase3/value_iteration_3.py`). Das ist eine zusaetzliche Modellierungsentscheidung
   *auf* der Beobachtungsbeschraenkung, nicht direkt durch sie vorgegeben.

Punkt 2 so einfach wie moeglich zu halten (keine Wahrscheinlichkeitsschaetzung
"wie viele stecken *wahrscheinlich* hinter 'besetzt', gegeben `p`?") ist
bewusst die naheliegendste, unaufwendigste Art, ein MDP direkt auf der
Beobachtung zu loesen, OHNE eine echte Belief-State-Schaetzung (= eine
tatsaechliche POMDP-Loesung) vorzunehmen — und genau das soll laut Vorlesung
nicht behandelt werden. Eine "schlauere" Naeherung waere hier also kein
sauberer Zwischenschritt, sondern bereits der erste Schritt in Richtung einer
echten POMDP-Loesung.

**Fazit: Ja, das ist genau die erwartete "schlechte Version".** Beide
Zutaten sind korrekt und im vorgesehenen Rahmen — die Beobachtungsbeschraenkung
entspricht der Aufgabenstellung, die Planungsvereinfachung ist die einfachste
vorlesungskonforme Wahl. Dass diese Kombination schon bei moderater Last so
drastisch versagt (bis hin zur Weigerung, eine einzelne wartende Person
mitzunehmen, siehe Beispiel unten), ist **kein Konstruktionsfehler**, sondern
die geforderte Demonstration: selbst diese elementarste Form von
Informationsverlust hat drastische, real messbare Konsequenzen.

**Konkretes Beispiel (Aufzug an Etage 4, genau 1 Person wartet dort, Kabine leer):**

```
p=0.10:  policy = O   (Wert: O=21.05 < R=21.46 — Tuer oeffnen ist besser)
p=0.20:  policy = R   (Wert: R=29.29 < O=30.03 — wegfahren ist besser!)
```

Beide Werte sind Bellman-konsistent berechnet (kein Rechenfehler). Der Grund:
aus Sicht des Naeherungsmodells verhaelt sich "besetzt" IMMER wie genau 1
Person — das ist strukturell aequivalent zu einem dauerhaften `K=1`-System
(egal wie gross `K` in der echten Umgebung tatsaechlich ist, siehe oben).
Und `K=1`-artige Systeme kollabieren nachweislich bei niedrigerem `p` als
`K=2`/`K=3`-Systeme (vgl. Phase 2, das ebenfalls effektiv nur 1 Person/Etage
kennt und schon bei `p=0.25` kollabiert).

---

## Naechste Schritte

1. **`K=4`/`K=5` testen** (optional) — `K=3` ist jetzt durchgehend verifiziert
   (kompletter `p`-Sweep gelaufen, Kollaps-Schwellen dokumentiert). Ein
   weiterer Anstieg wuerde Phase 2b's Schwelle vermutlich nochmal verschieben,
   Phase 3's aber nicht (siehe Nachtrag) — geringe Prioritaet, da die
   Kernaussage (Phase 3 kollabiert K-unabhaengig frueher) bereits robust belegt ist.
2. **`gamma` variieren** — z.B. 0.7 vs. 0.9 vs. 0.99, zeigt kurzsichtige vs.
   vorausschauende Strategie (und vermutlich auch Einfluss auf die
   Kollaps-Schwelle: kurzsichtigere Strategien koennten laenger "versuchen").
3. **Stabilitaet der Monte-Carlo-Schaetzung pruefen** — `EVAL_EPISODES`/
   `EVAL_STEPS` erhoehen, beobachten wie nah `avg_discounted_cost` an
   `V*(Startzustand)` heranrueckt (Bezug zur "T-Range"-Herleitung) — im
   stabilen `p`-Bereich.
4. **Visualisierung** (Teil (c)) — Plots aus `results/runs.csv`: Kosten/Schritt
   und Fahrtdauer je Phase ueber `p`, mit expliziter Kennzeichnung der
   Kollaps-Schwellen aus dem Abschnitt oben.