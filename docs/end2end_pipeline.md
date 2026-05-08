# Scripts for Running End-to-End Pipelines

This module documents the end-to-end classification pipelines available in this repository for Survey Assist (**SA**). The codebase currently contains scripts for both Standard Industry Classification (**SIC**) and Standard Occupational Classification (**SOC**) workflows.

Both pipelines start from respondent-provided survey fields and persist intermediate outputs plus a metadata JSON file so runs can be resumed from checkpoints. The common input file (CSV or parquet) is expected to contain at least `sic2007_employee`, `soc2020_job_title`, `soc2020_job_description`, and `sic2007_self_employed`.

## SIC pipeline

The SIC pipeline supports the full multi-stage flow: initial classification, ambiguity assessment, follow-up question generation, synthetic follow-up answer generation, second semantic search, and final classification.

The intended stage order is:
Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6 (rerun stage 1 with `-s`) -> Stage 7 (rerun stage 2 with `-s`).

| Stage | Pipeline process | Required columns | Columns added |
|--|--|--|--|
| 1 | Create `merged_industry_desc` and perform semantic search | `sic2007_employee`, `soc2020_job_title`, `soc2020_job_description`, `sic2007_self_employed` | `merged_industry_desc`, `semantic_search_results` |
| 2 | Initial classification and ambiguity assessment | `soc2020_job_title`, `soc2020_job_description`, `merged_industry_desc`, `semantic_search_results` | `unambiguously_codable`, `initial_code`, `alt_sic_candidates` |
| 3 | Generate a follow-up question when `unambiguously_codable` is `False` | `unambiguously_codable`, `merged_industry_desc`, `soc2020_job_title`, `soc2020_job_description`, `alt_sic_candidates` | `followup_question` |
| 4 | Generate a follow-up answer | `unambiguously_codable`, `merged_industry_desc`, `soc2020_job_title`, `soc2020_job_description`, `followup_question` | `followup_answer` |
| 5 | Modify `merged_industry_desc` | `merged_industry_desc`, `followup_question`, `followup_answer` | `extended_industry_desc` |
| 6 | Second semantic search using the extended description | `soc2020_job_title`, `soc2020_job_description`, `extended_industry_desc` | `second_semantic_search_results` |
| 7 | Final classification and ambiguity assessment | `soc2020_job_title`, `soc2020_job_description`, `extended_industry_desc`, `second_semantic_search_results` | `unambiguously_codable_final`, `final_code`, `alt_sic_candidates_final` |

### Run the full SIC pipeline

Use the runner in `scripts/sic_pipeline`:

```bash
./scripts/sic_pipeline/run_full_pipeline.sh [-p <1|2>] -i </path/to/input.{csv|parquet}> -o </path/to/output/folder> [-m </path/to/metadata.json>] [-b 20]
```

Where:
- `-p 2` (optional): Select one-prompt (`1`) or two-prompt (`2`) pipeline. Default is `2`.
- `-i`: Input CSV or parquet file.
- `-o`: Output folder.
- `-m` (optional): Existing metadata JSON file.
- `-b` (optional): Batch size. For LLM stages the practical maximum is `10`.

### Run individual SIC stages

```bash
poetry run python scripts/sic_pipeline/<stage_script>.py -i <path/to/input.{csv|parquet}> -o <path/to/output/folder> [-m <path/to/metadata.json>] [-n <output_shortname>] [-b <batch_size>] [-s] [-r]
```

Where:
- `-n` (optional): Output file prefix. Defaults to the stage ID such as `STG1`.
- `-s` (optional): Use the second-run behavior for scripts that support it.
- `-r` (optional): Resume from the last persisted intermediate checkpoint.

## SOC pipeline

The SOC pipeline currently ships as a one-prompt classification flow with follow-up enrichment and a second semantic-search pass. Unlike the SIC pipeline, the full two-prompt / final-classification path is not currently implemented in `scripts/soc_pipeline`.

The currently available stage order is:
Stage 1 -> Stage 2 -> Stage 4 -> Stage 5 -> Stage 6 (rerun stage 1 with `-s`).

| Stage | Available script | Pipeline process | Required columns | Columns added |
|--|--|--|--|--|
| 1 | `stage_1_add_semantic_search.py` | Create `merged_industry_desc` and perform SOC semantic search | `sic2007_employee`, `soc2020_job_title`, `soc2020_job_description`, `sic2007_self_employed` | `merged_industry_desc`, `semantic_search_results` |
| 2 | `stage_2_one_prompt_assign_sic_code.py` | Run one-prompt SOC classification, ambiguity assessment, and follow-up question generation | `soc2020_job_title`, `soc2020_job_description`, `merged_industry_desc`, `semantic_search_results` | `unambiguously_codable`, `initial_code`, `alt_soc_candidates`, `followup_question` |
| 4 | `stage_4_add_synthetic_responses.py` | Generate a follow-up answer | `unambiguously_codable`, `merged_industry_desc`, `soc2020_job_title`, `soc2020_job_description`, `followup_question` | `followup_answer` |
| 5 | `stage_5_modify_job_description.py` | Extend the job description using follow-up question and answer | `soc2020_job_description`, `followup_question`, `followup_answer` | `extended_job_desccription` |
| 6 | `stage_1_add_semantic_search.py -s` | Run a second semantic-search pass using the follow-up answer in the search query | `soc2020_job_title`, `soc2020_job_description`, `merged_industry_desc`, `followup_answer` | `second_semantic_search_results` |

Notes:
- The stage-2 SOC script keeps a legacy filename, `stage_2_one_prompt_assign_sic_code.py`, but it calls the SOC classifier and writes `alt_soc_candidates`.
- The SOC `run_full_pipeline.sh` currently executes a complete runnable path only for `-p 1`. The `-p 2` branch still contains commented-out placeholder stages and should not be treated as a complete two-prompt pipeline.

### Run the full SOC pipeline

Use the runner in `scripts/soc_pipeline`:

```bash
./scripts/soc_pipeline/run_full_pipeline.sh -p 1 -i </path/to/input.{csv|parquet}> -o </path/to/output/folder> [-m </path/to/metadata.json>] [-b 20]
```

Where:
- `-p 1`: Required for the currently complete SOC runner path.
- `-i`: Input CSV or parquet file.
- `-o`: Output folder.
- `-m` (optional): Existing metadata JSON file.
- `-b` (optional): Batch size.

### Run individual SOC stages

```bash
poetry run python scripts/soc_pipeline/<stage_script>.py -i <path/to/input.{csv|parquet}> -o <path/to/output/folder> [-m <path/to/metadata.json>] [-n <output_shortname>] [-b <batch_size>] [-s] [-r]
```

All SOC stage scripts use the shared pipeline CLI. In practice:
- `-s` is used by `stage_1_add_semantic_search.py` for the second semantic-search pass.
- `-r` resumes from the last persisted checkpoint when intermediate outputs exist.

## Prerequisites

- Python 3.12
- Poetry
- Google Cloud authentication for LLM-backed stages: `gcloud auth application-default login`

## Metadata

The metadata JSON stores shared configuration values plus per-stage checkpoint state. If no metadata file is supplied, defaults from the pipeline package are used.

| Field | Default value | Description |
|--|--|--|
| `original_dataset_name` | `<path/to/input.{csv|parquet}>` | Path to the original dataset |
| `embedding_model_name` | `all-MiniLM-L6-v2` | Embedding model used for semantic search |
| `embedding_db_dir` | `data/vector_store` | Directory for the vector store |
| `embedding_k_matches` | `20` | Number of semantic-search matches returned |
| `llm_model_name` | `gemini-2.5-flash` | LLM model used for classification or synthetic responses |
| `llm_model_location` | `europe-west2` | LLM location |
| `llm_candidates_limit` | `10` | Maximum number of candidates returned by configurable pipeline stages |
| `sic_code_digits` | `5` | Number of SIC digits |
| `sic_index_file` | `industrial_classification_utils.data.sic_index / uksic2007indexeswithaddendumdecember2022.xlsx` | SIC index lookup file |
| `sic_structure_file` | `industrial_classification_utils.data.sic_index / publisheduksicsummaryofstructureworksheet.xlsx` | SIC structure lookup file |
| `soc_index_file` | `occupational_classification_utils.data.soc_index / soc2020volume2thecodingindexexcel16102024.xlsx` | SOC index lookup file |
| `soc_structure_file` | `occupational_classification_utils.data.soc_index / soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx` | SOC structure lookup file |
| `batch_size` | `100` | Checkpoint batch size |
| `batch_size_async` | `10` | Async batch size for LLM stages |
