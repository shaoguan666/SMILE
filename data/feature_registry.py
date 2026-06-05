"""Canonical variable ordering and physiological groups for SMART datasets.

The registry follows the column order written by each preprocessing pipeline.
All consumers should request indices from here rather than keeping local
copies of feature or system definitions.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict


REGISTRY_VERSION = "bibm_audit_fixed_v1"


_FEATURE_NAMES = {
    "c12": (
        "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
        "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
        "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP", "MechVent",
        "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH",
        "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
        "TroponinT", "Urine", "WBC", "Weight",
    ),
    "c19": (
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine",
        "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
        "WBC", "Fibrinogen", "Platelets",
    ),
    "mimic": (
        "Capillary refill rate", "Diastolic blood pressure",
        "Fraction inspired oxygen", "Glascow coma scale eye opening",
        "Glascow coma scale motor response", "Glascow coma scale total",
        "Glascow coma scale verbal response", "Glucose", "Heart Rate",
        "Height", "Mean blood pressure", "Oxygen saturation",
        "Respiratory rate", "Systolic blood pressure", "Temperature",
        "Weight", "pH",
    ),
}


_SYSTEM_NAMES = {
    "c12": OrderedDict((
        ("Vital_Invasive", ("DiasABP", "MAP", "SysABP")),
        ("Vital_NonInvasive", ("NIDiasABP", "NIMAP", "NISysABP")),
        ("Vital_Basic", ("HR", "RespRate", "Temp")),
        ("Blood_Gas", ("FiO2", "PaCO2", "PaO2", "pH")),
        ("Electrolyte", ("K", "Mg", "Na", "HCO3")),
        ("Liver_Enzyme", ("Albumin", "ALP", "ALT", "AST", "Bilirubin")),
        ("Renal", ("BUN", "Creatinine", "Urine")),
        ("CBC", ("HCT", "Platelets", "WBC")),
        ("Cardiac_Biomarker", ("TroponinI", "TroponinT")),
    )),
    "c19": OrderedDict((
        ("Vital_Basic", ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp")),
        ("Resp_Gas", ("EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2")),
        ("Liver_Enzyme", ("AST", "Alkalinephos")),
        ("Bilirubin", ("Bilirubin_direct", "Bilirubin_total")),
        ("Renal", ("BUN", "Creatinine")),
        ("Electrolyte", ("Calcium", "Chloride", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium")),
        ("CBC", ("Hct", "Hgb", "WBC", "Platelets")),
        ("Coagulation", ("PTT", "Fibrinogen")),
    )),
    "mimic": OrderedDict((
        ("Vital_BP", ("Diastolic blood pressure", "Mean blood pressure", "Systolic blood pressure")),
        ("Vital_Basic", ("Capillary refill rate", "Heart Rate", "Oxygen saturation", "Respiratory rate", "Temperature")),
        ("GCS", ("Glascow coma scale eye opening", "Glascow coma scale motor response", "Glascow coma scale total", "Glascow coma scale verbal response")),
        ("Resp_Gas", ("Fraction inspired oxygen", "pH")),
        ("Anthropometric", ("Height", "Weight")),
    )),
}


def registry_key(dataset: str) -> str:
    """Return the cohort-level registry key for a task or cohort alias."""
    key = dataset.lower()
    if key == "c12" or "c12" in key:
        return "c12"
    if key == "c19" or "c19" in key:
        return "c19"
    if key == "mimic" or key.startswith("mimic_") or "mimic3" in key:
        return "mimic"
    raise ValueError("Unknown dataset for feature registry: %s" % dataset)


def get_feature_names(dataset: str) -> list[str]:
    """Return canonical feature names in stored pickle column order."""
    return list(_FEATURE_NAMES[registry_key(dataset)])


def get_candidate_system_names(dataset: str) -> OrderedDict[str, list[str]]:
    """Return candidate physiological systems as variable names."""
    systems = _SYSTEM_NAMES[registry_key(dataset)]
    return OrderedDict((name, list(features)) for name, features in systems.items())


def get_candidate_systems(dataset: str) -> OrderedDict[str, list[int]]:
    """Return candidate systems translated into canonical feature indices."""
    features = get_feature_names(dataset)
    index = {feature: i for i, feature in enumerate(features)}
    return OrderedDict(
        (system, [index[feature] for feature in group])
        for system, group in get_candidate_system_names(dataset).items()
    )


def validate_registry(dataset: str, feature_count: int | None = None) -> None:
    """Validate feature dimensions and disjoint candidate-system membership."""
    features = get_feature_names(dataset)
    if feature_count is not None and feature_count != len(features):
        raise ValueError(
            "%s has %d stored features but registry defines %d"
            % (dataset, feature_count, len(features))
        )
    groups = get_candidate_systems(dataset)
    all_indices = [idx for group in groups.values() for idx in group]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Candidate systems overlap in registry for %s" % dataset)
    if any(idx < 0 or idx >= len(features) for idx in all_indices):
        raise ValueError("Candidate systems include invalid indices for %s" % dataset)


def registry_fingerprint(dataset: str) -> str:
    """Return a stable short fingerprint for audit/training compatibility checks."""
    key = registry_key(dataset)
    payload = {
        "version": REGISTRY_VERSION,
        "key": key,
        "features": get_feature_names(key),
        "systems": get_candidate_system_names(key),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


for _dataset_key in _FEATURE_NAMES:
    validate_registry(_dataset_key)
