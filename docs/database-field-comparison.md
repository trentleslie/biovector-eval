# Database Field Comparison: HMDB, UMLS, LOINC

## Purpose
Compare available fields from direct database ingestion to prioritize Kraken integration work vs. what KG2 already provides.

---

## HMDB (Human Metabolome Database)

### Currently Used in biovector-eval (4 fields)
| Field | Example |
|-------|---------|
| `accession` | `HMDB0000001` |
| `name` | "1-Methylhistidine" |
| `synonyms` | ["L-1-Methylhistidine", "1-MHis"] |
| `iupac_name` | "(2S)-2-amino-3-(1-methyl-1H-imidazol-4-yl)propanoic acid" |

### Full HMDB Schema (~130 fields per metabolite)

#### Identifiers
| Field | Example |
|-------|---------|
| `accession` | `HMDB0000001` |
| `secondary_accessions` | `["HMDB00001", "HMDB0006262"]` |
| `cas_registry_number` | `332-80-9` |
| `pubchem_compound_id` | `92105` |
| `chebi_id` | `50599` |
| `kegg_id` | `C01152` |
| `chemspider_id` | `83160` |
| `drugbank_id` | `DB04151` |
| `foodb_id` | `FDB012119` |
| `pdb_id` | `1MH` |
| `wikipedia_id` | `1-Methylhistidine` |

#### Names & Descriptions
| Field | Example |
|-------|---------|
| `name` | "1-Methylhistidine" |
| `synonyms` | ["L-1-Methylhistidine", "1-MHis", "π-Methylhistidine"] |
| `iupac_name` | "(2S)-2-amino-3-(1-methyl-1H-imidazol-4-yl)propanoic acid" |
| `traditional_iupac` | "1-methylhistidine" |
| `description` | "1-Methylhistidine is a methylated amino acid..." |

#### Chemical Properties
| Field | Example |
|-------|---------|
| `chemical_formula` | `C7H11N3O2` |
| `average_molecular_weight` | `169.1824` |
| `monisotopic_molecular_weight` | `169.085126611` |
| `smiles` | `CN1C=NC(C[C@H](N)C(O)=O)=C1` |
| `inchi` | `InChI=1S/C7H11N3O2/c1-10-4-9-5(3-10)2-6(8)7(11)12/h3-4,6H,2,8H2,1H3,(H,11,12)/t6-/m0/s1` |
| `inchikey` | `BRMWTNUJHUMWMS-LURJTMIESA-N` |
| `state` | "Solid" |

#### Physical Properties
| Field | Example |
|-------|---------|
| `water_solubility` | `67.2 g/L` |
| `logp` | `-3.5` |
| `pka_strongest_acidic` | `1.69` |
| `pka_strongest_basic` | `8.85` |
| `physiological_charge` | `0` |
| `polar_surface_area` | `81.14 Å²` |
| `rotatable_bond_count` | `3` |
| `refractivity` | `43.87` |
| `polarizability` | `17.08` |

#### Taxonomy & Classification
| Field | Example |
|-------|---------|
| `kingdom` | "Organic compounds" |
| `super_class` | "Organic acids and derivatives" |
| `class` | "Carboxylic acids and derivatives" |
| `sub_class` | "Amino acids, peptides, and analogues" |
| `direct_parent` | "Histidine and derivatives" |
| `alternative_parents` | ["Imidazolyl carboxylic acids", "Alpha amino acids"] |

#### Biological Context
| Field | Example |
|-------|---------|
| `biospecimen_locations` | ["Blood", "Urine", "Cerebrospinal Fluid"] |
| `tissue_locations` | ["Skeletal Muscle", "Liver"] |
| `cellular_locations` | ["Cytoplasm", "Mitochondria"] |
| `pathways` | [{"name": "Histidine Metabolism", "kegg_id": "map00340"}] |
| `protein_associations` | [{"uniprot_id": "P42898", "name": "Histidine ammonia-lyase"}] |
| `biological_function` | "Energy source" |

#### Clinical Data
| Field | Example |
|-------|---------|
| `normal_concentrations` | [{"biospecimen": "Blood", "value": "5-15 μM"}] |
| `abnormal_concentrations` | [{"biospecimen": "Urine", "condition": "Histidinemia", "value": ">200 μM"}] |
| `diseases` | [{"name": "Histidinemia", "omim_id": "235800"}] |
| `drugbank_metabolite_id` | `DBMET00123` |

#### External Database Cross-References
| Database | Field |
|----------|-------|
| KEGG | `kegg_id` |
| PubChem | `pubchem_compound_id` |
| ChEBI | `chebi_id` |
| UniProt | `uniprot_id` |
| DrugBank | `drugbank_id` |
| FooDB | `foodb_id` |
| Phenol-Explorer | `phenol_explorer_compound_id` |
| KNApSAcK | `knapsack_id` |
| BioCyc | `biocyc_id` |
| METLIN | `metlin_id` |
| VMH | `vmh_id` |

---

## UMLS Metathesaurus

### Overview
- **3.49M concepts**, **17.4M unique names**
- **190 source vocabularies** (ICD-10, SNOMED CT, MeSH, RxNorm, etc.)

### Core Data Files

#### MRCONSO.RRF (Concepts & Names)
| Field | Description | Example |
|-------|-------------|---------|
| `CUI` | Concept Unique Identifier | `C0017725` |
| `LAT` | Language | `ENG` |
| `TS` | Term status | `P` (preferred) |
| `LUI` | Lexical unique identifier | `L0017725` |
| `STT` | String type | `PF` (preferred form) |
| `SUI` | String unique identifier | `S0046854` |
| `ISPREF` | Is preferred atom | `Y` |
| `AUI` | Atom unique identifier | `A0022506` |
| `SAUI` | Source atom identifier | `M0008817` |
| `SCUI` | Source concept identifier | `D005947` |
| `SDUI` | Source descriptor identifier | `D005947` |
| `SAB` | Source abbreviation | `MSH` (MeSH) |
| `TTY` | Term type in source | `MH` (Main Heading) |
| `CODE` | Source vocabulary code | `D005947` |
| `STR` | String (the actual name) | "Glucose" |
| `SRL` | Source restriction level | `0` |
| `SUPPRESS` | Suppressible flag | `N` |
| `CVF` | Content view flag | `4096` |

#### MRREL.RRF (Relationships)
| Field | Description | Example |
|-------|-------------|---------|
| `CUI1` | Concept 1 | `C0017725` |
| `AUI1` | Atom 1 | `A0022506` |
| `STYPE1` | Source type 1 | `AUI` |
| `REL` | Relationship label | `RN` (narrower) |
| `CUI2` | Concept 2 | `C0017726` |
| `AUI2` | Atom 2 | `A0022507` |
| `STYPE2` | Source type 2 | `AUI` |
| `RELA` | Relationship attribute | `is_a` |
| `RUI` | Relationship identifier | `R12345678` |
| `SAB` | Source | `SNOMEDCT_US` |
| `SL` | Source of relationship label | `SNOMEDCT_US` |
| `RG` | Relationship group | `0` |

**Common REL values:**
- `PAR`/`CHD` - Parent/Child
- `RB`/`RN` - Broader/Narrower
- `SY` - Synonym
- `RO` - Other related
- `AQ` - Allowed qualifier

**Common RELA values (~25% of relationships):**
- `is_a`, `part_of`, `has_part`
- `branch_of`, `tributary_of`
- `component_of`, `has_component`
- `has_finding_site`, `classifies`
- `mapped_from`, `mapped_to`

#### MRSAT.RRF (Attributes)
| Field | Description | Example |
|-------|-------------|---------|
| `CUI` | Concept identifier | `C0017725` |
| `LUI` | Lexical identifier | `L0017725` |
| `SUI` | String identifier | `S0046854` |
| `METAUI` | Meta unique identifier | `A0022506` |
| `STYPE` | Source type | `AUI` |
| `CODE` | Source code | `D005947` |
| `ATUI` | Attribute identifier | `AT12345` |
| `SATUI` | Source attribute identifier | `AN0012345` |
| `ATN` | Attribute name | `SEMANTIC_TYPE` |
| `SAB` | Source | `MTH` |
| `ATV` | Attribute value | `Carbohydrate` |
| `SUPPRESS` | Suppressible flag | `N` |

#### MRSTY.RRF (Semantic Types)
| Field | Description | Example |
|-------|-------------|---------|
| `CUI` | Concept identifier | `C0017725` |
| `TUI` | Semantic type identifier | `T109` |
| `STN` | Semantic type tree number | `A1.4.1.2.1` |
| `STY` | Semantic type name | `Organic Chemical` |
| `ATUI` | Attribute identifier | `AT12345678` |
| `CVF` | Content view flag | `4096` |

**Key Semantic Types:**
- `T047` - Disease or Syndrome
- `T109` - Organic Chemical
- `T116` - Amino Acid, Peptide, or Protein
- `T121` - Pharmacologic Substance
- `T123` - Biologically Active Substance
- `T059` - Laboratory Procedure
- `T034` - Laboratory or Test Result

#### MRDEF.RRF (Definitions)
| Field | Description | Example |
|-------|-------------|---------|
| `CUI` | Concept identifier | `C0017725` |
| `AUI` | Atom identifier | `A0022506` |
| `ATUI` | Attribute identifier | `AT38139342` |
| `SATUI` | Source attribute identifier | - |
| `SAB` | Source | `MSH` |
| `DEF` | Definition text | "A primary source of energy for living organisms..." |
| `SUPPRESS` | Suppressible flag | `N` |

---

## LOINC (Logical Observation Identifiers Names and Codes)

### Overview
- **~100,000+ observation codes**
- **6-axis naming system** for precise identification

### Main LOINC Table Fields

#### Core Identifiers
| Field | Description | Example |
|-------|-------------|---------|
| `LOINC_NUM` | Unique code | `2345-7` |
| `LONG_COMMON_NAME` | Readable name | "Glucose [Mass/volume] in Serum or Plasma" |
| `SHORTNAME` | Abbreviated name | "Glucose SerPl-mCnc" |
| `CONSUMER_NAME` | Patient-friendly | "Blood Sugar" |
| `DisplayName` | Clinician-friendly | "Glucose, Serum" |

#### 6-Axis Decomposition
| Axis | Field | Description | Example |
|------|-------|-------------|---------|
| 1 | `COMPONENT` | What's measured | "Glucose" |
| 2 | `PROPERTY` | Measurement characteristic | "MCnc" (mass concentration) |
| 3 | `TIME_ASPCT` | Timing | "Pt" (point in time) |
| 4 | `SYSTEM` | Specimen/body system | "Ser/Plas" (Serum or Plasma) |
| 5 | `SCALE_TYP` | Result scale | "Qn" (quantitative) |
| 6 | `METHOD_TYP` | Methodology (optional) | "Glucometer" |

**Common PROPERTY values:**
- `MCnc` - Mass concentration
- `SCnc` - Substance concentration
- `Imp` - Impression/interpretation
- `Prid` - Presence or identity
- `NFr` - Number fraction
- `ACnc` - Arbitrary concentration

**Common SCALE_TYP values:**
- `Qn` - Quantitative
- `Ord` - Ordinal
- `Nom` - Nominal
- `Nar` - Narrative
- `Doc` - Document

#### Classification
| Field | Description | Example |
|-------|-------------|---------|
| `CLASS` | Observation class | "CHEM" |
| `CLASSTYPE` | Type code | `1` (Laboratory) |
| `ORDER_OBS` | Order vs observation | "Both" |
| `STATUS` | Active/deprecated | "ACTIVE" |
| `STATUS_REASON` | If deprecated, why | "DUPLICATE" |

**CLASS values:**
- `CHEM` - Chemistry
- `HEM/BC` - Hematology/Blood Cell Count
- `UA` - Urinalysis
- `MICRO` - Microbiology
- `SERO` - Serology
- `DRUG/TOX` - Drug Levels & Toxicology
- `PANEL.CHEM` - Chemistry Panel

**CLASSTYPE values:**
- `1` - Laboratory class
- `2` - Clinical class
- `3` - Claims attachments
- `4` - Surveys

#### Additional Metadata
| Field | Description | Example |
|-------|-------------|---------|
| `DefinitionDescription` | Clinical narrative | "Measures glucose concentration..." |
| `RELATEDNAMES2` | Synonyms | "FBS;Fasting glucose;Blood sugar" |
| `FORMULA` | Calculation formula | "HDL/Total Cholesterol" |
| `EXAMPLE_UCUM_UNITS` | Standard units | "mg/dL" |
| `EXMPL_ANSWERS` | Valid answers | "Positive;Negative;Indeterminate" |
| `SURVEY_QUEST_TEXT` | Survey question | "How often do you exercise?" |
| `VersionFirstReleased` | First release | "2.50" |
| `VersionLastChanged` | Last change | "2.73" |
| `CHNG_TYPE` | Change type | "MIN" (minor) |

#### Panel Structure (LoincPartLink)
| Field | Description | Example |
|-------|-------------|---------|
| `LOINC` | Panel code | `24323-8` |
| `PartNumber` | Part code | `LP7057-5` |
| `PartName` | Part name | "Basic Metabolic Panel" |
| `PartTypeName` | Part type | "Component" |
| `LinkTypeName` | Link type | "Primary" |
| `PanelType` | Panel classification | "Panel" |

**PanelType values:**
- `Panel` - True panel/battery
- `Organizer` - Grouping construct
- `Convenience group` - Informal grouping

---

## Summary: Fields to Compare Against KG2

### High-Value Fields Likely Missing from KG2

#### HMDB
- [ ] Physical/chemical properties (logP, solubility, pKa, molecular weight)
- [ ] Normal/abnormal concentration ranges by biospecimen
- [ ] Spectral data (NMR, MS)
- [ ] Detailed taxonomy (kingdom → direct_parent hierarchy)
- [ ] Cellular/tissue localizations
- [ ] Full cross-reference set (15+ databases)

#### UMLS
- [ ] Full MRSAT attribute set (source-specific properties)
- [ ] Detailed RELA relationship qualifiers
- [ ] Semantic type assignments (MRSTY)
- [ ] Multi-source synonym aggregation
- [ ] Source-specific term types (TTY)

#### LOINC
- [ ] 6-axis semantic decomposition
- [ ] Panel/group hierarchies
- [ ] UCUM unit specifications
- [ ] Answer lists for nominal scales
- [ ] Survey question text
- [ ] Clinical definitions

---

## Sources

- [HMDB Downloads](https://hmdb.ca/downloads)
- [HMDB About](https://www.hmdb.ca/about)
- [HMDB 5.0 Paper](https://academic.oup.com/nar/article/50/D1/D622/6431815)
- [UMLS Metathesaurus Reference](https://www.ncbi.nlm.nih.gov/books/NBK9684/)
- [UMLS 2025AB Release Notes](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/notes.html)
- [LOINC Database Structure](https://loinc.org/kb/users-guide/loinc-database-structure/)
- [LOINC Features](https://docs.snomed.org/implementation-guides/loinc-implementation-guide/about-loinc/2.1-loinc-features)

---

*Generated for Kraken integration prioritization - January 2026*
