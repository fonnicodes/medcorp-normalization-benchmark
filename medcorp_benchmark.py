"""Benchmark MedCorp's flat legacy data against a BCNF-style decomposition.

Uses only the Python standard library. Run with:
    python medcorp_benchmark.py medcorp_legacy_dump.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


def deep_size(obj: object) -> int:
    """Return recursive object size while counting shared objects only once."""
    seen: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            size += sum(visit(k) + visit(v) for k, v in item.items())
        elif isinstance(item, (list, tuple, set, frozenset)):
            size += sum(visit(v) for v in item)
        return size

    return visit(obj)


def load_flat(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def canonical(store: dict[str, str], value: str) -> str:
    """Reuse one identifier object across the normalized structures."""
    return store.setdefault(value, value)


def normalize(rows: list[dict[str, str]]) -> dict[str, object]:
    ids: dict[str, str] = {}
    patients: dict[str, tuple[str, str, str]] = {}
    doctors: dict[str, list[str]] = {}
    wards: dict[str, str] = {}
    treatments: dict[str, tuple[str, float]] = {}
    blood_types: set[str] = set()
    specialties: set[str] = set()
    appointments: list[tuple[str, str, str, str]] = []
    appointment_treatments: dict[int, list[str]] = {}
    appointment_index: dict[tuple[str, str, str, str], int] = {}

    for row in rows:
        patient_id = canonical(ids, row["Patient_ID"])
        doctor_id = canonical(ids, row["Doctor_ID"])
        ward_id = canonical(ids, row["Ward_ID"])
        treatment_code = canonical(ids, row["Treatment_Code"])
        blood_type = canonical(ids, row["Patient_Blood_Type"])
        specialty = canonical(ids, row["Doctor_Specialty"])
        appointment_date = canonical(ids, row["Appointment_Date"])

        blood_types.add(blood_type)
        specialties.add(specialty)
        patients.setdefault(
            patient_id,
            (row["Patient_Name"], row["Patient_DOB"], blood_type),
        )
        doctors.setdefault(
            doctor_id,
            [row["Doctor_Name"], specialty, row["Doctor_Phone"]],
        )
        wards.setdefault(ward_id, row["Ward_Location"])
        treatments.setdefault(
            treatment_code,
            (row["Treatment_Description"], float(row["Treatment_Cost"])),
        )

        natural_key = (patient_id, appointment_date, doctor_id, ward_id)
        appointment_id = appointment_index.get(natural_key)
        if appointment_id is None:
            appointment_id = len(appointments)
            appointment_index[natural_key] = appointment_id
            appointments.append(natural_key)
            appointment_treatments[appointment_id] = []
        appointment_treatments[appointment_id].append(treatment_code)

    return {
        "blood_types": blood_types,
        "specialties": specialties,
        "patients": patients,
        "doctors": doctors,
        "wards": wards,
        "treatments": treatments,
        "appointments": appointments,
        "appointment_treatments": appointment_treatments,
    }


def median_ms(action, trials: int = 25, inner: int = 1) -> float:
    samples = []
    for index in range(trials + 5):
        start = time.perf_counter_ns()
        for repeat in range(inner):
            action(index * inner + repeat)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000 / inner
        if index >= 5:
            samples.append(elapsed)
    return statistics.median(samples)


def benchmark(path: Path) -> dict[str, object]:
    flat = load_flat(path)
    normalized = normalize(flat)
    doctors = normalized["doctors"]
    appointments = normalized["appointments"]
    appointment_treatments = normalized["appointment_treatments"]

    target_doctor = Counter(r["Doctor_ID"] for r in flat).most_common(1)[0][0]
    target_rows = sum(r["Doctor_ID"] == target_doctor for r in flat)
    alternate_phones = ("+1-555-010-1000", "+1-555-010-2000")

    def flat_update(index: int) -> None:
        new_phone = alternate_phones[index % 2]
        for row in flat:
            if row["Doctor_ID"] == target_doctor:
                row["Doctor_Phone"] = new_phone

    def normalized_update(index: int) -> None:
        doctors[target_doctor][2] = alternate_phones[index % 2]

    selected: list[tuple[int, int]] = []
    selected_patients: set[str] = set()
    for appointment_id, appointment in enumerate(appointments):
        patient_id = appointment[0]
        if patient_id not in selected_patients:
            selected_patients.add(patient_id)
            selected.append((appointment_id, appointment_id))
            if len(selected) == 1000:
                break

    # Map each normalized appointment key to the first matching flat row once.
    flat_key_to_index: dict[tuple[str, str, str, str], int] = {}
    for index, row in enumerate(flat):
        key = (
            row["Patient_ID"],
            row["Appointment_Date"],
            row["Doctor_ID"],
            row["Ward_ID"],
        )
        flat_key_to_index.setdefault(key, index)
    selected_pairs = [
        (flat_key_to_index[appointments[appointment_id]], appointment_id)
        for _, appointment_id in selected
    ]

    def flat_read(_: int) -> None:
        output = []
        for row_index, _appointment_id in selected_pairs:
            # The flat row is already a complete human-readable record.
            output.append(flat[row_index])
        if len(output) != 1000:
            raise AssertionError("Flat reconstruction did not return 1,000 records")

    def normalized_read(_: int) -> None:
        output = []
        patients = normalized["patients"]
        wards = normalized["wards"]
        treatments = normalized["treatments"]
        for _row_index, appointment_id in selected_pairs:
            patient_id, date, doctor_id, ward_id = appointments[appointment_id]
            patient_name, dob, blood_type = patients[patient_id]
            doctor_name, specialty, phone = doctors[doctor_id]
            ward_location = wards[ward_id]
            treatment_code = appointment_treatments[appointment_id][0]
            description, cost = treatments[treatment_code]
            output.append(
                (
                    patient_id, patient_name, dob, blood_type, date,
                    doctor_id, doctor_name, specialty, phone,
                    ward_id, ward_location, treatment_code, description, cost,
                )
            )
        if len(output) != 1000:
            raise AssertionError("Normalized reconstruction did not return 1,000 records")

    metrics = {
        "rows": len(flat),
        "entity_counts": {
            "blood_types": len(normalized["blood_types"]),
            "specialties": len(normalized["specialties"]),
            "patients": len(normalized["patients"]),
            "doctors": len(normalized["doctors"]),
            "wards": len(normalized["wards"]),
            "treatments": len(normalized["treatments"]),
            "appointments": len(normalized["appointments"]),
            "appointment_treatments": sum(
                len(v) for v in normalized["appointment_treatments"].values()
            ),
        },
        "target_doctor": target_doctor,
        "target_rows": target_rows,
        "memory_mb": {
            "flat": deep_size(flat) / (1024 * 1024),
            "normalized": deep_size(normalized) / (1024 * 1024),
        },
        "mutation_ms": {
            "flat": median_ms(flat_update),
            "normalized": median_ms(normalized_update, inner=1000),
        },
        "read_1000_ms": {
            "flat": median_ms(flat_read, inner=20),
            "normalized": median_ms(normalized_read, inner=20),
        },
        "trials": 25,
        "python": sys.version.split()[0],
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = benchmark(args.csv_path)
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
