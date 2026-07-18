import ipaddress

import pandas as pd
import numpy as np


def join_labels(requests: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join incident labels to requests by IP, CIDR range, and TLS fingerprint with temporal bounds."""
    requests = requests.copy()

    if "timestamp" not in requests.columns or labels.empty:
        requests["is_malicious"] = False
        requests["attack_class"] = np.nan
        requests["confidence"] = np.nan
        return requests

    requests["timestamp"] = pd.to_datetime(requests["timestamp"], utc=True)
    for col in ["active_from", "active_until", "labeled_at"]:
        if col in labels.columns:
            labels[col] = pd.to_datetime(labels[col], utc=True)

    ip_labels = labels[labels.identifier_type == "ip"]
    range_labels = labels[labels.identifier_type == "ip_range"]
    tls_labels = labels[labels.identifier_type == "tls_fingerprint"]

    match_cols = ["request_id", "incident_id", "attack_class", "confidence"]
    all_matches = []

    if not ip_labels.empty:
        merged = requests.merge(
            ip_labels[
                [
                    "source_identifier",
                    "attack_class",
                    "confidence",
                    "incident_id",
                    "active_from",
                    "active_until",
                ]
            ],
            left_on="source_ip",
            right_on="source_identifier",
            how="inner",
        )
        merged = merged[
            (merged.timestamp >= merged.active_from)
            & (merged.timestamp <= merged.active_until)
        ]
        if not merged.empty:
            all_matches.append(merged[match_cols])

    for _, row in range_labels.iterrows():
        net = ipaddress.ip_network(row.source_identifier, strict=False)
        mask = requests.source_ip.apply(lambda x: ipaddress.ip_address(x) in net)
        temporal = (requests.timestamp >= row.active_from) & (
            requests.timestamp <= row.active_until
        )
        matched = requests[mask & temporal].copy()
        if not matched.empty:
            matched["incident_id"] = row.incident_id
            matched["attack_class"] = row.attack_class
            matched["confidence"] = row.confidence
            all_matches.append(matched[match_cols])

    if not tls_labels.empty:
        merged = requests.merge(
            tls_labels[
                [
                    "source_identifier",
                    "attack_class",
                    "confidence",
                    "incident_id",
                    "active_from",
                    "active_until",
                ]
            ],
            left_on="tls_fingerprint",
            right_on="source_identifier",
            how="inner",
        )
        merged = merged[
            (merged.timestamp >= merged.active_from)
            & (merged.timestamp <= merged.active_until)
        ]
        if not merged.empty:
            all_matches.append(merged[match_cols])

    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        deduped = combined.drop_duplicates(subset="request_id", keep="first")
        malicious_ids = set(deduped.request_id)
        requests["is_malicious"] = requests.request_id.isin(malicious_ids)
        requests = requests.merge(
            deduped[["request_id", "attack_class", "confidence"]],
            on="request_id",
            how="left",
        )
    else:
        requests["is_malicious"] = False
        requests["attack_class"] = np.nan
        requests["confidence"] = np.nan

    return requests
