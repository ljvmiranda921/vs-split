"""Patterns that you can use for entity-switch.v1

Some were based from the work, 'Entity-Switched Datasets - An Approach to
Auditing the In-Domain Robustness of Named Entity Recognition Models' by Agarwal
et al., (ACL 2021).
"""
from typing import Dict, List

# Indonesian and Filipino Names
ID_NAMES = [
    "Dwi",
    "Siti",
    "Tri",
    "Dian",
    "Ade",
    "Endang",
    "Yudi",
    "Henny",
    "Dewi",
    "Agus",
    "Budi",
    "Bambang",
    "Hendra",
    "Indra",
    "Iwan",
    "Eko",
    "Agung",
    "Achmad",
    "Wahyu",
]

PH_NAMES = [
    "Jay",
    "Jun",
    "Alvin",
    "Jayson",
    "Mj",
    "Carlo",
    "Noel",
    "Rey",
    "Jerome",
    "Arnel" "Lyn",
    "Jocelyn",
    "Maricel",
    "Rowena",
    "Mae",
    "Arlene",
    "Jen",
    "Mary Ann",
    "Cherry",
    "Kristine",
]


def get_id_ph_names_pattern() -> Dict[str, List[str]]:
    return {"PER": ID_NAMES + PH_NAMES}
