"""Demographics domain scaffold.

This module will provide support for demographic variable evaluation
once data is available. Demographics data typically includes:
- Variable codes/identifiers
- Variable labels/descriptions
- Category values and labels

Expected data format: TSV with columns like:
- variable_code: Unique identifier
- variable_label: Human-readable name
- description: Detailed description (used as synonym)
- category: Variable category

TODO: Implement DemographicsDomain class when data is ready.
"""

# Domain will be registered when implementation is complete
# from biovector_eval.domains import register_domain
# from biovector_eval.domains.demographics.domain import DemographicsDomain
# register_domain("demographics", DemographicsDomain)

__all__: list[str] = []
