"""
Gemeinsame Auswertung fuer alle Phasen des Aufzug-Projekts (Random-Baseline,
Value Iteration mit voller Information Phase 1/2, realistische Kapazitaet
Phase 2b, spaeter reduzierte Beobachtung Phase 3).

Bewusst ENVIRONMENT-AGNOSTISCH gehalten (kein Import von ElevatorEnv/
ElevatorEnvV2 hier): evaluate() bekommt eine env_factory(seed) uebergeben,
die eine frische Umgebungsinstanz liefert - egal welchen Typs. So kann
dieselbe Auswertung fuer Phase 1/2 (elevator_environment.py) UND Phase 2b
(elevator_environment_2.py) verwendet werden, ohne Code zu duplizieren.

Jede Phase ruft evaluate() mit denselben (gamma, steps, n_episodes) auf und
haengt das Ergebnis per append_result() an results/runs.csv UND an eine
phasenspezifische results/{label}.csv an - so bleiben alle bisherigen
Phasen-Ergebnisse erhalten und lassen sich sowohl phasenweise als auch
gemeinsam vergleichen/plotten (Teil (c) der Aufgabenstellung).
"""

import csv
import os
import random
import statistics

RESULTS_DIR = "results"
RUNS_CSV = os.path.join(RESULTS_DIR, "runs.csv")

# Spalten, die NIE Teil des Vergleichs-Schluessels sind (das sind die berechneten
# Ergebnisse selbst) - alle anderen Spalten einer Zeile (label, gamma, steps,
# n_episodes, und je nach Environment z.B. n_floors/p oder zusaetzlich
# capacity/max_wait_per_floor) identifizieren zusammen eine Konfiguration.
METRIC_COLUMNS = [
    "V", "avg_total_cost", "avg_cost_per_step", "avg_discounted_cost",
    "avg_trip_duration", "n_completed_trips_total", "n_total_arrivals", "completion_rate",
]


def make_random_policy(seed):
    """policy_fn(env) -> Aktion, zieht gleichverteilt aus den verfuegbaren
    Aktionen. Eigener RNG, unabhaengig vom Ankunfts-RNG der Umgebung."""
    rng = random.Random(seed)

    def policy_fn(env):
        return rng.choice(env.available_actions())

    return policy_fn


def make_table_policy(policy):
    """policy_fn(env) -> Aktion, Tabellen-Lookup einer (z.B. per Value
    Iteration) vorberechneten Strategie."""

    def policy_fn(env):
        state = (env.floor, env.door_open, tuple(env.waiting_dest), tuple(env.cabin_dest))
        return policy[state]

    return policy_fn


def write_convergence_log(path, history):
    """Schreibt die Konvergenz-Historie einer Value Iteration (siehe
    return_history=True bei value_iteration() in den Phase-Modulen) als CSV:
    iteration, delta (max|V_neu-V_alt| ueber alle Zustaende - das eigentliche
    Abbruchkriterium), V_start (Wert des Startzustands in dieser Iteration -
    die Zahl, die am Ende als V*(Startzustand) berichtet wird)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["iteration", "delta", "V_start"])
        for iteration, delta, v_start in history:
            writer.writerow([iteration, f"{delta:.6f}", round(v_start, 4)])
    print(f"Konvergenz-Log ({len(history)} Iterationen) geschrieben nach: {path}")


def run_episode(policy_fn, env_factory, steps, seed):
    """Fuehrt eine policy_fn ueber `steps` Schritte auf einer frisch von
    env_factory(seed) erzeugten Umgebung aus. Gibt die Kosten pro Schritt, die
    abgeschlossenen Fahrtdauern UND die Gesamtzahl aller in dieser Episode
    aufgetretenen Personen zurueck (env.total_arrivals - fuer die
    Erledigungsquote, siehe evaluate()). env_factory ist bewusst
    Environment-Typ-agnostisch (ElevatorEnv, ElevatorEnvV2, ...)."""
    env = env_factory(seed)  # eigene, frische Umgebung pro Episode
    costs = []
    for _ in range(steps):
        action = policy_fn(env)          # naechste Aktion von der uebergebenen Strategie erfragen
        _, cost = env.step(action)
        costs.append(cost)               # Kosten Schritt fuer Schritt sammeln (fuer Summe/Diskontierung)
    return costs, list(env.completed_trips), env.total_arrivals


def discounted_sum(costs, gamma):
    # Sigma_t gamma^t * cost_t, t=0,1,2,... - diskontierte Gesamtkosten EINER simulierten
    # Episode. Gemittelt ueber mehrere Episoden (siehe evaluate()) ergibt das einen
    # Vergleichswert zu V*, das in value_iteration.py exakt (nicht simuliert) berechnet
    # wurde - dient nur als zusaetzlicher Korrektheits-Check, nicht zur Berechnung von V*
    # selbst. Bei gamma=0.9 ist der Fehler durch die Begrenzung auf `steps` Schritte statt
    # unendlich vieler vernachlaessigbar klein (siehe Herleitung zur "T-Range" im Bericht).
    return sum(c * gamma**t for t, c in enumerate(costs))


def evaluate(label, policy_fn_factory, env_factory, gamma, steps, n_episodes, seed_start=0, **meta):
    """Mittelt eine Strategie ueber n_episodes unabhaengige Episoden (verschiedene
    Seeds), damit Random- und VI-Strategie fair (gleiche Bedingungen, gemittelt
    statt eine einzelne Stichprobe) vergleichbar sind.

    policy_fn_factory(seed) -> policy_fn(env) -> Aktion.
    env_factory(seed) -> frische, mit diesem seed initialisierte Umgebungsinstanz
    (Environment-Typ-agnostisch, siehe Moduldocstring).
    **meta: zusaetzliche Parameter der jeweiligen Umgebung (z.B. n_floors, p,
    oder bei Phase 2b zusaetzlich capacity, max_wait_per_floor) - werden 1:1 in
    die Ergebniszeile uebernommen, damit results/runs.csv nachvollziehbar bleibt.
    """
    total_costs, discounted_costs, trip_durations = [], [], []
    total_arrivals_sum = 0
    # Seeds 0,1,2,...,n_episodes-1 (bzw. ab seed_start): jede Episode bekommt einen
    # eigenen, aber reproduzierbaren Zufallsverlauf des Ankunftsprozesses
    for i in range(n_episodes):
        seed = seed_start + i
        policy_fn = policy_fn_factory(seed)
        costs, trips, arrivals = run_episode(policy_fn, env_factory, steps, seed)
        total_costs.append(sum(costs))                       # undiskontierte Gesamtkosten dieser Episode
        discounted_costs.append(discounted_sum(costs, gamma))  # diskontierte Gesamtkosten dieser Episode
        trip_durations.extend(trips)                          # alle abgeschlossenen Fahrten sammeln
        total_arrivals_sum += arrivals                        # alle aufgetretenen Personen ueber alle Episoden

    n_completed = len(trip_durations)
    result = {"label": label, **meta, "gamma": gamma, "steps": steps, "n_episodes": n_episodes}
    result.update({
        "avg_total_cost": round(statistics.mean(total_costs), 4),
        "avg_cost_per_step": round(statistics.mean(total_costs) / steps, 4),
        "avg_discounted_cost": round(statistics.mean(discounted_costs), 4),
        "avg_trip_duration": round(statistics.mean(trip_durations), 4) if trip_durations else None,
        "n_completed_trips_total": n_completed,
        "n_total_arrivals": total_arrivals_sum,
        # Erledigungsquote: Anteil der aufgetretenen Personen, die in `steps` Schritten
        # auch tatsaechlich ankamen (nicht 1:1 aus n_completed/n_total_arrivals ableitbar,
        # da manche der aufgetretenen Personen am Episodenende noch unterwegs sind - genau
        # DAS macht die Quote aussagekraeftig, siehe PHASES.md/PRAESENTATION.md)
        "completion_rate": round(n_completed / total_arrivals_sum, 4) if total_arrivals_sum else None,
    })
    return result


def _upsert_row(result, path):
    """Fuegt eine Ergebniszeile in path ein (Upsert): existiert bereits eine Zeile mit
    denselben Nicht-Metrik-Spalten (label, gamma, steps, n_episodes, sowie alle in
    **meta uebergebenen Environment-Parameter), wird sie ersetzt statt dupliziert
    (z.B. beim erneuten Ausfuehren des Skripts). Neue Phasen/Parameter-Kombinationen
    kommen als zusaetzliche Zeile dazu -> die Historie bleibt erhalten, nur echte
    Wiederholungen derselben Konfiguration werden nicht angehaeuft.

    Spaltenmenge wird dynamisch als Vereinigung aller bisher gesehenen Felder
    gefuehrt, damit verschiedene Environment-Typen (mit unterschiedlichen
    Meta-Feldern, z.B. Phase 1/2 ohne 'capacity' vs. Phase 2b mit 'capacity')
    nebeneinander in derselben Datei stehen koennen."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    existing_rows = []
    existing_fields = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)

    # Spaltenreihenfolge: bisherige Felder zuerst (stabil), neue Felder hinten an
    fieldnames = list(existing_fields)
    for k in result.keys():
        if k not in fieldnames:
            fieldnames.append(k)

    key_cols = [c for c in fieldnames if c not in METRIC_COLUMNS]
    new_key = {c: str(result.get(c, "")) for c in key_cols}
    # Zeilen eines ANDEREN Environment-Typs haben fehlende Felder ("") an den
    # Stellen, die beim neuen Ergebnis gesetzt sind -> matchen automatisch nicht
    rows = [r for r in existing_rows if {c: r.get(c, "") for c in key_cols} != new_key]
    rows.append({k: str(v) for k, v in result.items()})

    with open(path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ergebnis '{result['label']}' in {path} aktualisiert ({len(rows)} Zeilen insgesamt).")


# Exakt diese Spalten (und nur diese) soll die Pro-Phase-Datei results/{label}.csv
# enthalten - bewusst schlanker als runs.csv, das zusaetzlich avg_total_cost/
# avg_cost_per_step/avg_discounted_cost/capacity fuehrt (fuer Cross-Phase-Vergleiche).
PER_PHASE_COLUMNS = [
    "label", "n_floors", "p", "gamma", "steps", "n_episodes",
    "V", "n_completed_trips_total", "n_total_arrivals", "completion_rate",
    "avg_trip_duration", "max_wait_per_floor",
]


def append_result(result, path=RUNS_CSV, also_per_phase=True):
    """Schreibt eine Ergebniszeile per Upsert (siehe _upsert_row()) - standardmaessig
    in ZWEI Dateien: (1) die zentrale `path` (default: results/runs.csv, mit ALLEN
    Feldern, fuer Phasen-uebergreifende Vergleiche) UND (2) eine phasenspezifische
    Datei `results/{label}.csv` (also_per_phase=True) mit NUR den Spalten aus
    PER_PHASE_COLUMNS, damit jede Phase auch fuer sich genommen eine eigene,
    schlanke Ergebnistabelle hat."""
    _upsert_row(result, path)
    if also_per_phase:
        per_phase_path = os.path.join(RESULTS_DIR, f"{result['label']}.csv")
        if per_phase_path != path:
            slim_result = {k: result.get(k, "") for k in PER_PHASE_COLUMNS}
            _upsert_row(slim_result, per_phase_path)