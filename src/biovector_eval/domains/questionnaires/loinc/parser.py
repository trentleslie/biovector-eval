"""LOINC CSV parser for questionnaire harmonization.

Parses LOINC Table Core CSV files into BaseEntity objects suitable
for embedding and vector search.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from biovector_eval.base.entity import BaseEntity


class LOINCParser:
    """Parser for LOINC Table Core CSV files.

    Extracts LOINC codes with their associated names and metadata,
    converting them to BaseEntity format for vector embedding.

    The LOINC 6-part structure (COMPONENT:PROPERTY:TIME:SYSTEM:SCALE:METHOD)
    provides rich semantic information that can be used for multi-vector
    search strategies.

    Attributes:
        loinc_csv_path: Path to LoincTableCore.csv
        consumer_names_path: Optional path to ConsumerName.csv for
            additional consumer-friendly names
    """

    # Key columns to extract from LOINC CSV
    REQUIRED_COLUMNS = {
        "LOINC_NUM",
        "LONG_COMMON_NAME",
        "COMPONENT",
        "STATUS",
    }

    METADATA_COLUMNS = {
        "PROPERTY",
        "TIME_ASPCT",
        "SYSTEM",
        "SCALE_TYP",
        "METHOD_TYP",
        "CLASS",
        "CLASSTYPE",
        "SHORTNAME",
        "STATUS",
        "VersionFirstReleased",
        "VersionLastChanged",
    }

    def __init__(
        self,
        loinc_csv_path: Path | str,
        consumer_names_path: Path | str | None = None,
    ):
        """Initialize LOINC parser.

        Args:
            loinc_csv_path: Path to LoincTableCore.csv
            consumer_names_path: Optional path to ConsumerName.csv
        """
        self.loinc_csv_path = Path(loinc_csv_path)
        self.consumer_names_path = (
            Path(consumer_names_path) if consumer_names_path else None
        )
        self._consumer_names: dict[str, str] | None = None

    def _load_consumer_names(self) -> dict[str, str]:
        """Load consumer-friendly names from ConsumerName.csv.

        Returns:
            Dict mapping LOINC_NUM to consumer-friendly name.
        """
        if self._consumer_names is not None:
            return self._consumer_names

        self._consumer_names = {}
        if self.consumer_names_path and self.consumer_names_path.exists():
            with open(self.consumer_names_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loinc_num = row.get("LoincNumber", "").strip()
                    consumer_name = row.get("ConsumerName", "").strip()
                    if loinc_num and consumer_name:
                        self._consumer_names[loinc_num] = consumer_name

        return self._consumer_names

    def parse(
        self,
        active_only: bool = True,
        include_classes: list[str] | None = None,
        exclude_classes: list[str] | None = None,
    ) -> list[BaseEntity]:
        """Parse LOINC CSV into BaseEntity objects.

        Args:
            active_only: If True, only include ACTIVE status codes.
            include_classes: If provided, only include codes from these classes.
                            Example: ["SURVEY.GENERAL", "SURVEY.PROMIS"]
            exclude_classes: If provided, exclude codes from these classes.

        Returns:
            List of BaseEntity objects representing LOINC codes.
        """
        consumer_names = self._load_consumer_names()
        entities: list[BaseEntity] = []

        with open(self.loinc_csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate required columns exist
            if reader.fieldnames is None:
                raise ValueError("CSV file has no headers")
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            for row in reader:
                # Filter by status
                status = row.get("STATUS", "").strip()
                if active_only and status != "ACTIVE":
                    continue

                # Filter by class
                loinc_class = row.get("CLASS", "").strip()
                if include_classes and loinc_class not in include_classes:
                    continue
                if exclude_classes and loinc_class in exclude_classes:
                    continue

                # Extract core fields
                loinc_num = row.get("LOINC_NUM", "").strip()
                long_common_name = row.get("LONG_COMMON_NAME", "").strip()
                component = row.get("COMPONENT", "").strip()
                short_name = row.get("SHORTNAME", "").strip()

                if not loinc_num or not long_common_name:
                    continue

                # Build synonyms list (unique, non-empty values)
                synonyms_set: set[str] = set()
                if short_name and short_name != long_common_name:
                    synonyms_set.add(short_name)
                if component and component != long_common_name:
                    synonyms_set.add(component)
                if loinc_num in consumer_names:
                    consumer_name = consumer_names[loinc_num]
                    if consumer_name != long_common_name:
                        synonyms_set.add(consumer_name)

                # Build metadata
                metadata = {
                    "loinc_class": loinc_class,
                    "component": component,
                    "property": row.get("PROPERTY", "").strip(),
                    "time_aspect": row.get("TIME_ASPCT", "").strip(),
                    "system": row.get("SYSTEM", "").strip(),
                    "scale_type": row.get("SCALE_TYP", "").strip(),
                    "method_type": row.get("METHOD_TYP", "").strip(),
                    "class_type": row.get("CLASSTYPE", "").strip(),
                    "status": status,
                }

                # Create entity
                entity = BaseEntity(
                    id=loinc_num,
                    name=long_common_name,
                    synonyms=sorted(synonyms_set),
                    metadata=metadata,
                )
                entities.append(entity)

        return entities

    def parse_survey_codes(self) -> list[BaseEntity]:
        """Parse questionnaire-relevant LOINC codes.

        This method returns LOINC codes covering:
        - Diet / Food Security (SDOH, SAMHSA, NURSE.OMAHA, PHENX)
        - Stress / Anxiety (PROMIS, MTLHLTH, NIH.EMO, NEUROQ, PHQ)
        - Happiness / Well-being (PROMIS, NIH.EMO, R-OUTCOMES)
        - Lifestyle (IPAQ, NIDA, COVID sleep/behavior items)
        - Health History (MDS, USSGFHT, OASIS, CMS)
        - Digestion (GNHLTH GI symptoms, NURSE.OMAHA bowel function)
        - Personality (PHENX personality instruments)

        Approximately 10,000 codes from 31 LOINC classes (30 SURVEY.* + PHENX).

        Excluded classes (specialized clinical, not general survey content):
        - SURVEY.OPTIMAL (rehab equipment)
        - SURVEY.RFC (disability assessment)
        - SURVEY.ESRD (renal staffing)
        - PANEL.* (panel definitions, not items)
        - H&P.* (clinical history documentation)

        Returns:
            List of BaseEntity objects for questionnaire-relevant LOINC codes.
        """
        return self.parse(
            active_only=True,
            include_classes=[
                # Core survey classes
                "SURVEY.PROMIS",  # ~2452 - anxiety, emotional distress, well-being
                "SURVEY.GNHLTH",  # ~1614 - general health, GI symptoms
                "SURVEY.MTLHLTH",  # ~394 - mental health surveys
                "SURVEY.CMS",  # ~950 - medical history items
                "SURVEY.CDC",  # ~varies - CDC health surveys
                "SURVEY.AHRQ",  # ~varies - healthcare research
                "SURVEY.IPAQ",  # ~53 - Physical Activity Questionnaire
                "SURVEY.SAMHSA",  # ~35 - diet items (fast food, fruits, sodas)
                "SURVEY.HAQ",  # Health Assessment Questionnaire
                "SURVEY.GDS",  # Geriatric Depression Scale
                "SURVEY.EPDS",  # Edinburgh Postnatal Depression
                "SURVEY.PNDS",  # Nursing Data Set
                "SURVEY.MISC",  # Miscellaneous surveys
                "SURVEY.USSGFHT",  # ~21 - Family Health History
                # Stress/anxiety/well-being classes
                "SURVEY.NIH.EMO",  # ~62 - fear, anxiety, positive affect, satisfaction
                "SURVEY.NEUROQ",  # ~382 - worry, stress, well-being
                "SURVEY.R-OUTCOMES",  # ~79 - user/life satisfaction
                "SURVEY.OASIS",  # ~128 - prior status assessments
                "SURVEY.MDS",  # ~804 - prior functioning, history
                # Diet/lifestyle additions
                "SURVEY.SDOH",  # ~96 - Social Determinants with 39 food security items
                "SURVEY.PHQ",  # ~80 - PHQ/GAD anxiety, appetite items
                "SURVEY.NIDA",  # ~75 - substance use
                "SURVEY.NURSE.OMAHA",  # ~402 - nutrition, physical activity, bowel
                "SURVEY.NURSE.HHCC",  # Home Health Care Classification
                "SURVEY.NMMDS",  # ~161 - psychological stress
                "SURVEY.COVID",  # ~186 - sleep, behavior changes
                "SURVEY.BPI",  # Brief Pain Inventory
                # PhenX - comprehensive instrument library
                "PHENX",  # ~1359 - diet, personality, lifestyle instruments
            ],
        )

    def save_entities(
        self,
        entities: list[BaseEntity],
        output_path: Path | str,
    ) -> None:
        """Save parsed entities to JSON file.

        Args:
            entities: List of BaseEntity objects to save.
            output_path: Path to output JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [e.to_dict() for e in entities]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def parse_loinc_csv(
    loinc_csv_path: Path | str,
    consumer_names_path: Path | str | None = None,
    active_only: bool = True,
    survey_only: bool = False,
) -> list[BaseEntity]:
    """Convenience function to parse LOINC CSV.

    Args:
        loinc_csv_path: Path to LoincTableCore.csv
        consumer_names_path: Optional path to ConsumerName.csv
        active_only: If True, only include ACTIVE status codes.
        survey_only: If True, only include survey-related codes.

    Returns:
        List of BaseEntity objects.
    """
    parser = LOINCParser(loinc_csv_path, consumer_names_path)
    if survey_only:
        return parser.parse_survey_codes()
    return parser.parse(active_only=active_only)
