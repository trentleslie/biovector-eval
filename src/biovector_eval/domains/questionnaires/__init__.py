"""Questionnaires domain scaffold.

This module will provide support for questionnaire item evaluation
once data is available. Questionnaire data typically includes:
- Question codes/identifiers
- Question text
- Response options
- Scale labels

Expected data format: TSV with columns like:
- question_id: Unique identifier
- question_text: Full question text
- short_label: Brief label (used as name)
- response_options: Semicolon-separated response choices

TODO: Implement QuestionnairesDomain class when data is ready.
"""

# Domain will be registered when implementation is complete
# from biovector_eval.domains import register_domain
# from biovector_eval.domains.questionnaires.domain import QuestionnairesDomain
# register_domain("questionnaires", QuestionnairesDomain)

__all__: list[str] = []
