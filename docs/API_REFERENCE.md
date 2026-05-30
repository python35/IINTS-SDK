# API Reference

This page is generated from the Python source tree by `tools/docs/generate_api_reference.py`.
Do not edit it by hand; regenerate it after public module changes.

Documented modules: **178**

## Package Index

| Package | Modules |
| --- | ---: |
| `ai` | 11 |
| `analysis` | 30 |
| `api` | 4 |
| `cli` | 3 |
| `core` | 32 |
| `data` | 25 |
| `demo_assets` | 1 |
| `emulation` | 5 |
| `highlevel` | 1 |
| `jetson` | 3 |
| `learning` | 3 |
| `live_patient` | 11 |
| `mdmp` | 3 |
| `metrics` | 1 |
| `population` | 3 |
| `presets` | 1 |
| `research` | 16 |
| `root` | 1 |
| `scenarios` | 3 |
| `templates` | 6 |
| `utils` | 5 |
| `validation` | 7 |
| `visualization` | 3 |

## Module Index

### `ai`

- [`iints.ai`](#iintsai)
- [`iints.ai.assistant`](#iintsaiassistant)
- [`iints.ai.backends`](#iintsaibackends)
- [`iints.ai.backends.base`](#iintsaibackendsbase)
- [`iints.ai.backends.mistral_api`](#iintsaibackendsmistral_api)
- [`iints.ai.backends.ollama`](#iintsaibackendsollama)
- [`iints.ai.cli`](#iintsaicli)
- [`iints.ai.mdmp_guard`](#iintsaimdmp_guard)
- [`iints.ai.model_catalog`](#iintsaimodel_catalog)
- [`iints.ai.prepare`](#iintsaiprepare)
- [`iints.ai.prompts`](#iintsaiprompts)

### `analysis`

- [`iints.analysis`](#iintsanalysis)
- [`iints.analysis.algorithm_xray`](#iintsanalysisalgorithm_xray)
- [`iints.analysis.baseline`](#iintsanalysisbaseline)
- [`iints.analysis.booth_demo`](#iintsanalysisbooth_demo)
- [`iints.analysis.carelink_workbench`](#iintsanalysiscarelink_workbench)
- [`iints.analysis.clinical_benchmark`](#iintsanalysisclinical_benchmark)
- [`iints.analysis.clinical_metrics`](#iintsanalysisclinical_metrics)
- [`iints.analysis.clinical_tir_analyzer`](#iintsanalysisclinical_tir_analyzer)
- [`iints.analysis.diabetes_metrics`](#iintsanalysisdiabetes_metrics)
- [`iints.analysis.edge_efficiency`](#iintsanalysisedge_efficiency)
- [`iints.analysis.edge_performance_monitor`](#iintsanalysisedge_performance_monitor)
- [`iints.analysis.eucys_results`](#iintsanalysiseucys_results)
- [`iints.analysis.evidence_bundle`](#iintsanalysisevidence_bundle)
- [`iints.analysis.explainability`](#iintsanalysisexplainability)
- [`iints.analysis.explainable_ai`](#iintsanalysisexplainable_ai)
- [`iints.analysis.hardware_benchmark`](#iintsanalysishardware_benchmark)
- [`iints.analysis.metrics`](#iintsanalysismetrics)
- [`iints.analysis.population_report`](#iintsanalysispopulation_report)
- [`iints.analysis.poster`](#iintsanalysisposter)
- [`iints.analysis.reporting`](#iintsanalysisreporting)
- [`iints.analysis.run_quality`](#iintsanalysisrun_quality)
- [`iints.analysis.safety_index`](#iintsanalysissafety_index)
- [`iints.analysis.safety_visualizer`](#iintsanalysissafety_visualizer)
- [`iints.analysis.sensor_filtering`](#iintsanalysissensor_filtering)
- [`iints.analysis.study_analysis`](#iintsanalysisstudy_analysis)
- [`iints.analysis.study_engine`](#iintsanalysisstudy_engine)
- [`iints.analysis.study_experiment`](#iintsanalysisstudy_experiment)
- [`iints.analysis.study_poster`](#iintsanalysisstudy_poster)
- [`iints.analysis.study_protocol`](#iintsanalysisstudy_protocol)
- [`iints.analysis.validator`](#iintsanalysisvalidator)

### `api`

- [`iints.api`](#iintsapi)
- [`iints.api.base_algorithm`](#iintsapibase_algorithm)
- [`iints.api.registry`](#iintsapiregistry)
- [`iints.api.template_algorithm`](#iintsapitemplate_algorithm)

### `cli`

- [`iints.cli`](#iintscli)
- [`iints.cli.cli`](#iintsclicli)
- [`iints.cli.patient_cli`](#iintsclipatient_cli)

### `core`

- [`iints.core`](#iintscore)
- [`iints.core.algorithms`](#iintscorealgorithms)
- [`iints.core.algorithms.battle_runner`](#iintscorealgorithmsbattle_runner)
- [`iints.core.algorithms.clinical_baseline`](#iintscorealgorithmsclinical_baseline)
- [`iints.core.algorithms.correction_bolus`](#iintscorealgorithmscorrection_bolus)
- [`iints.core.algorithms.discovery`](#iintscorealgorithmsdiscovery)
- [`iints.core.algorithms.fixed_basal_bolus`](#iintscorealgorithmsfixed_basal_bolus)
- [`iints.core.algorithms.hybrid_algorithm`](#iintscorealgorithmshybrid_algorithm)
- [`iints.core.algorithms.imitation_controller`](#iintscorealgorithmsimitation_controller)
- [`iints.core.algorithms.lstm_algorithm`](#iintscorealgorithmslstm_algorithm)
- [`iints.core.algorithms.mock_algorithms`](#iintscorealgorithmsmock_algorithms)
- [`iints.core.algorithms.neural_controller`](#iintscorealgorithmsneural_controller)
- [`iints.core.algorithms.pid_controller`](#iintscorealgorithmspid_controller)
- [`iints.core.algorithms.standard_pump_algo`](#iintscorealgorithmsstandard_pump_algo)
- [`iints.core.device`](#iintscoredevice)
- [`iints.core.device_manager`](#iintscoredevice_manager)
- [`iints.core.devices`](#iintscoredevices)
- [`iints.core.devices.models`](#iintscoredevicesmodels)
- [`iints.core.patient`](#iintscorepatient)
- [`iints.core.patient.bergman_model`](#iintscorepatientbergman_model)
- [`iints.core.patient.models`](#iintscorepatientmodels)
- [`iints.core.patient.patient_factory`](#iintscorepatientpatient_factory)
- [`iints.core.patient.profile`](#iintscorepatientprofile)
- [`iints.core.physiology_variation`](#iintscorephysiology_variation)
- [`iints.core.safety`](#iintscoresafety)
- [`iints.core.safety.config`](#iintscoresafetyconfig)
- [`iints.core.safety.input_validator`](#iintscoresafetyinput_validator)
- [`iints.core.safety.supervisor`](#iintscoresafetysupervisor)
- [`iints.core.simulation`](#iintscoresimulation)
- [`iints.core.simulation.scenario_parser`](#iintscoresimulationscenario_parser)
- [`iints.core.simulator`](#iintscoresimulator)
- [`iints.core.supervisor`](#iintscoresupervisor)

### `data`

- [`iints.data`](#iintsdata)
- [`iints.data.adapter`](#iintsdataadapter)
- [`iints.data.certify`](#iintsdatacertify)
- [`iints.data.column_mapper`](#iintsdatacolumn_mapper)
- [`iints.data.contracts`](#iintsdatacontracts)
- [`iints.data.demo`](#iintsdatademo)
- [`iints.data.evidence`](#iintsdataevidence)
- [`iints.data.guardians`](#iintsdataguardians)
- [`iints.data.importer`](#iintsdataimporter)
- [`iints.data.ingestor`](#iintsdataingestor)
- [`iints.data.mdmp_visualizer`](#iintsdatamdmp_visualizer)
- [`iints.data.medtronic_live`](#iintsdatamedtronic_live)
- [`iints.data.nightscout`](#iintsdatanightscout)
- [`iints.data.quality_checker`](#iintsdataquality_checker)
- [`iints.data.realism_dashboard`](#iintsdatarealism_dashboard)
- [`iints.data.realism_governance`](#iintsdatarealism_governance)
- [`iints.data.realism_reference`](#iintsdatarealism_reference)
- [`iints.data.realism_validator`](#iintsdatarealism_validator)
- [`iints.data.registry`](#iintsdataregistry)
- [`iints.data.research_catalog`](#iintsdataresearch_catalog)
- [`iints.data.runner`](#iintsdatarunner)
- [`iints.data.study_corruption`](#iintsdatastudy_corruption)
- [`iints.data.synthetic_mirror`](#iintsdatasynthetic_mirror)
- [`iints.data.tidepool`](#iintsdatatidepool)
- [`iints.data.universal_parser`](#iintsdatauniversal_parser)

### `demo_assets`

- [`iints.demo_assets`](#iintsdemo_assets)

### `emulation`

- [`iints.emulation`](#iintsemulation)
- [`iints.emulation.legacy_base`](#iintsemulationlegacy_base)
- [`iints.emulation.medtronic_780g`](#iintsemulationmedtronic_780g)
- [`iints.emulation.omnipod_5`](#iintsemulationomnipod_5)
- [`iints.emulation.tandem_controliq`](#iintsemulationtandem_controliq)

### `highlevel`

- [`iints.highlevel`](#iintshighlevel)

### `jetson`

- [`iints.jetson`](#iintsjetson)
- [`iints.jetson.endurance`](#iintsjetsonendurance)
- [`iints.jetson.research_pipeline`](#iintsjetsonresearch_pipeline)

### `learning`

- [`iints.learning`](#iintslearning)
- [`iints.learning.autonomous_optimizer`](#iintslearningautonomous_optimizer)
- [`iints.learning.learning_system`](#iintslearninglearning_system)

### `live_patient`

- [`iints.live_patient`](#iintslive_patient)
- [`iints.live_patient.api`](#iintslive_patientapi)
- [`iints.live_patient.daemon`](#iintslive_patientdaemon)
- [`iints.live_patient.edge_benchmark`](#iintslive_patientedge_benchmark)
- [`iints.live_patient.edge_ops`](#iintslive_patientedge_ops)
- [`iints.live_patient.long_study`](#iintslive_patientlong_study)
- [`iints.live_patient.medtronic_direct`](#iintslive_patientmedtronic_direct)
- [`iints.live_patient.pico_pump`](#iintslive_patientpico_pump)
- [`iints.live_patient.runtime`](#iintslive_patientruntime)
- [`iints.live_patient.service_export`](#iintslive_patientservice_export)
- [`iints.live_patient.uno_q`](#iintslive_patientuno_q)

### `mdmp`

- [`iints.mdmp`](#iintsmdmp)
- [`iints.mdmp.backend`](#iintsmdmpbackend)
- [`iints.mdmp.eu_ai_pact`](#iintsmdmpeu_ai_pact)

### `metrics`

- [`iints.metrics`](#iintsmetrics)

### `population`

- [`iints.population`](#iintspopulation)
- [`iints.population.generator`](#iintspopulationgenerator)
- [`iints.population.runner`](#iintspopulationrunner)

### `presets`

- [`iints.presets`](#iintspresets)

### `research`

- [`iints.research`](#iintsresearch)
- [`iints.research.audit`](#iintsresearchaudit)
- [`iints.research.calibration_gate`](#iintsresearchcalibration_gate)
- [`iints.research.config`](#iintsresearchconfig)
- [`iints.research.control`](#iintsresearchcontrol)
- [`iints.research.control_eval`](#iintsresearchcontrol_eval)
- [`iints.research.data_blend`](#iintsresearchdata_blend)
- [`iints.research.dataset`](#iintsresearchdataset)
- [`iints.research.evaluation`](#iintsresearchevaluation)
- [`iints.research.local_ai`](#iintsresearchlocal_ai)
- [`iints.research.local_ai_gate`](#iintsresearchlocal_ai_gate)
- [`iints.research.losses`](#iintsresearchlosses)
- [`iints.research.metrics`](#iintsresearchmetrics)
- [`iints.research.model_registry`](#iintsresearchmodel_registry)
- [`iints.research.neural_control`](#iintsresearchneural_control)
- [`iints.research.predictor`](#iintsresearchpredictor)

### `root`

- [`iints`](#iints)

### `scenarios`

- [`iints.scenarios`](#iintsscenarios)
- [`iints.scenarios.generator`](#iintsscenariosgenerator)
- [`iints.scenarios.study_pack`](#iintsscenariosstudy_pack)

### `templates`

- [`iints.templates`](#iintstemplates)
- [`iints.templates.default_algorithm`](#iintstemplatesdefault_algorithm)
- [`iints.templates.demos`](#iintstemplatesdemos)
- [`iints.templates.demos.live_stage_demo`](#iintstemplatesdemoslive_stage_demo)
- [`iints.templates.pico_pump.code`](#iintstemplatespico_pumpcode)
- [`iints.templates.scenarios`](#iintstemplatesscenarios)

### `utils`

- [`iints.utils`](#iintsutils)
- [`iints.utils.csv_safety`](#iintsutilscsv_safety)
- [`iints.utils.plotting`](#iintsutilsplotting)
- [`iints.utils.run_io`](#iintsutilsrun_io)
- [`iints.utils.url_safety`](#iintsutilsurl_safety)

### `validation`

- [`iints.validation`](#iintsvalidation)
- [`iints.validation.golden`](#iintsvalidationgolden)
- [`iints.validation.replay`](#iintsvalidationreplay)
- [`iints.validation.run_doctor`](#iintsvalidationrun_doctor)
- [`iints.validation.run_validation`](#iintsvalidationrun_validation)
- [`iints.validation.safety_contract`](#iintsvalidationsafety_contract)
- [`iints.validation.schemas`](#iintsvalidationschemas)

### `visualization`

- [`iints.visualization`](#iintsvisualization)
- [`iints.visualization.cockpit`](#iintsvisualizationcockpit)
- [`iints.visualization.uncertainty_cloud`](#iintsvisualizationuncertainty_cloud)

## `iints`

- Source: `src/iints/__init__.py`
- Summary: No module docstring.
- Explicit exports: `InsulinAlgorithm, AlgorithmInput, AlgorithmResult, AlgorithmMetadata, WhyLogEntry, Simulator, StressEvent, PatientModel, DeviceManager, PatientFactory, PatientProfile, SimulationLimitError, SafetySupervisor, SafetyConfig, SensorModel, PumpModel, SENSOR_PROFILES, create_sensor_model, StandardPumpAlgorithm, ConstantDoseAlgorithm, RandomDoseAlgorithm, RunawayAIAlgorithm, StackingAIAlgorithm, DataIngestor, ImportResult, export_demo_csv, export_standard_csv, guess_column_mapping, import_carelink_csv, import_carelink_timeline, import_cgm_csv, import_cgm_dataframe, load_carelink_event_log, load_demo_dataframe, scenario_from_csv, scenario_from_dataframe, summarize_carelink_csv, NightscoutConfig, import_nightscout, TidepoolClient, TidepoolConfig, import_tidepool, load_openapi_spec, mdmp_gate, MDMPGateError, generate_synthetic_mirror, SyntheticMirrorArtifact, AVAILABLE_STUDY_CORRUPTIONS, apply_study_corruptions, write_corrupted_study_csv, generate_benchmark_metrics, build_booth_demo, build_carelink_workbench, build_study_protocol_payload, render_study_protocol_markdown, write_study_protocol_bundle, ClinicalReportGenerator, EnergyEstimate, estimate_energy_per_decision, AIResponse, IINTSAssistant, MDMPGuard, create_edge_bundle, export_edge_setup, LivePatientDaemon, PatientRuntimeConfig, create_patient_app, export_uno_q_bridge, get_runtime_scenario_profile, list_runtime_scenario_profiles, run_edge_benchmark, summarize_edge_workspace, write_edge_update_script, generate_report, generate_quickstart_report, generate_demo_report, generate_agp_report, generate_agp_assets, generate_results_poster, run_simulation, run_full, run_population, ScenarioGeneratorConfig, generate_random_scenario, PopulationGenerator, PopulationConfig, ParameterDistribution, PopulationRunner, PopulationResult, PatientResult, BergmanPatientModel`

### Public Functions

- `run_simulation(*args: Any, **kwargs: Any) -> Any`
- `run_full(*args: Any, **kwargs: Any) -> Any`
- `run_population(*args: Any, **kwargs: Any) -> Any`
- `generate_report(simulation_results: 'pd.DataFrame', output_path: Optional[str] = None, safety_report: Optional[dict] = None) -> Optional[str]`
- `generate_quickstart_report(simulation_results: 'pd.DataFrame', output_path: Optional[str] = None, safety_report: Optional[dict] = None) -> Optional[str]`
- `generate_demo_report(simulation_results: 'pd.DataFrame', output_path: Optional[str] = None, safety_report: Optional[dict] = None) -> Optional[str]`
- `generate_agp_report(simulation_results: 'pd.DataFrame', output_path: Optional[str] = None, safety_report: Optional[dict] = None, subject_name: str = 'Research simulation', summary_json_path: Optional[str] = None) -> Optional[str]`
- `generate_agp_assets(simulation_results: 'pd.DataFrame', output_dir: Optional[str] = None, subject_name: str = 'Research simulation', summary_json_path: Optional[str] = None) -> Optional[dict]`

## `iints.ai`

- Source: `src/iints/ai/__init__.py`
- Summary: No module docstring.
- Explicit exports: `AIResponse, IINTSAssistant, DEFAULT_MINISTRAL_MODEL, DEFAULT_OLLAMA_HOST, OllamaBackend, GuardResult, MDMPGuard, LocalMistralModelProfile, list_local_mistral_models, prepare_ai_ready_artifacts`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.ai.assistant`

- Source: `src/iints/ai/assistant.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `AIResponse` | `AIResponse` | No module docstring. |
| `IINTSAssistant` | `IINTSAssistant` | Research-only LLM assistant gated by MDMP certification. |

#### `AIResponse` methods

- `to_dict(self) -> dict[str, Any]`

#### `IINTSAssistant` methods

- `explain_decision(self, step: dict[str, Any]) -> AIResponse`
- `analyze_trends(self, glucose_payload: list[Any] | dict[str, Any]) -> AIResponse`
- `detect_anomalies(self, results: dict[str, Any]) -> AIResponse`
- `generate_report(self, run: dict[str, Any]) -> AIResponse`
- `review_realism(self, run: dict[str, Any]) -> AIResponse`

## `iints.ai.backends`

- Source: `src/iints/ai/backends/__init__.py`
- Summary: No module docstring.
- Explicit exports: `CompletionBackend, DEFAULT_MINISTRAL_MODEL, DEFAULT_OLLAMA_HOST, OllamaBackend, MistralAPIBackend`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.ai.backends.base`

- Source: `src/iints/ai/backends/base.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `CompletionBackend` | `CompletionBackend(Protocol)` | No module docstring. |

#### `CompletionBackend` methods

- `available(self) -> bool`
- `complete(self, *, system_prompt: str, user_prompt: str) -> str`

## `iints.ai.backends.mistral_api`

- Source: `src/iints/ai/backends/mistral_api.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `MistralAPIBackend` | `MistralAPIBackend` | No module docstring. |

#### `MistralAPIBackend` methods

- `available(self) -> bool`
- `complete(self, *, system_prompt: str, user_prompt: str) -> str`

## `iints.ai.backends.ollama`

- Source: `src/iints/ai/backends/ollama.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `OllamaBackend` | `OllamaBackend` | No module docstring. |

#### `OllamaBackend` methods

- `available(self) -> bool`
- `server_version(self) -> str | None`
- `version_supported(self) -> tuple[bool | None, str | None]`
- `list_models(self) -> list[str]`
- `resolve_model_name(self) -> str | None`
- `ensure_model_ready(self) -> str`
- `healthcheck(self) -> dict[str, object]`
- `smoke_test(self) -> dict[str, object]`
- `complete(self, *, system_prompt: str, user_prompt: str) -> str`

### Public Constants

- `DEFAULT_MINISTRAL_MODEL`
- `DEFAULT_OLLAMA_HOST`
- `LEGACY_MINISTRAL_MODEL`
- `MINISTRAL_MODEL_ALIASES`
- `MIN_OLLAMA_VERSION_FOR_MINISTRAL_3`

## `iints.ai.cli`

- Source: `src/iints/ai/cli.py`
- Summary: No module docstring.

### Public Functions

- `models() -> None`
- `prepare(run_dir: Annotated[Path, typer.Argument(help='Run output directory containing results.csv and run_metadata.json.')], create_dev_mdmp_cert: Annotated[bool, typer.Option('--create-dev-mdmp-cert/--no-create-dev-mdmp-cert', help='Generate a local development MDMP certificate and keypair for AI commands.')] = True, grade: Annotated[str, typer.Option(help='Grade to embed in the local development MDMP certificate.')] = 'research_grade', expires_days: Annotated[int, typer.Option(help='Certificate expiry window in days for local development certs.')] = 30, key_dir: Annotated[Optional[Path], typer.Option(help='Optional directory to store the generated local MDMP keypair.')] = None) -> None`
- `local_check(model: Annotated[str, typer.Option(help='Ollama model name to validate locally.')] = DEFAULT_MINISTRAL_MODEL, ollama_host: Annotated[Optional[str], typer.Option(help='Override the Ollama base URL.')] = None, timeout_seconds: Annotated[float, typer.Option(help='HTTP timeout for Ollama health checks.')] = 120.0, smoke_test: Annotated[bool, typer.Option('--smoke-test/--no-smoke-test', help='Run a tiny generation request after health checks to prove the model can actually answer.')] = True) -> None`
- `explain(input_json: Annotated[Path, typer.Argument(help='Prepared run directory or JSON file with a single simulation step or decision context.')], mdmp_cert: Annotated[Optional[Path], typer.Option(help='Signed MDMP artifact required before AI analysis can run.')] = None, mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = 'auto', model: Annotated[str, typer.Option(help='Ollama model name to use.')] = DEFAULT_MINISTRAL_MODEL, minimum_grade: Annotated[str, typer.Option(help='Minimum MDMP grade required to allow analysis.')] = 'research_grade', public_key: Annotated[Optional[Path], typer.Option(help='Explicit MDMP public key for verification.')] = None, trust_store: Annotated[Optional[Path], typer.Option(help='MDMP trust store for verification.')] = None, ollama_host: Annotated[Optional[str], typer.Option(help='Override the Ollama base URL.')] = None, timeout_seconds: Annotated[float, typer.Option(help='HTTP timeout for Ollama generation requests.')] = 120.0, output: Annotated[Optional[Path], typer.Option(help='Optional file path to save the explanation.')] = None) -> None`
- `trends(input_json: Annotated[Path, typer.Argument(help='Prepared run directory or JSON file with glucose trace data or a run payload.')], mdmp_cert: Annotated[Optional[Path], typer.Option(help='Signed MDMP artifact required before AI analysis can run.')] = None, mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = 'auto', model: Annotated[str, typer.Option(help='Ollama model name to use.')] = DEFAULT_MINISTRAL_MODEL, minimum_grade: Annotated[str, typer.Option(help='Minimum MDMP grade required to allow analysis.')] = 'research_grade', public_key: Annotated[Optional[Path], typer.Option(help='Explicit MDMP public key for verification.')] = None, trust_store: Annotated[Optional[Path], typer.Option(help='MDMP trust store for verification.')] = None, ollama_host: Annotated[Optional[str], typer.Option(help='Override the Ollama base URL.')] = None, timeout_seconds: Annotated[float, typer.Option(help='HTTP timeout for Ollama generation requests.')] = 120.0, output: Annotated[Optional[Path], typer.Option(help='Optional file path to save the analysis.')] = None) -> None`
- `anomalies(input_json: Annotated[Path, typer.Argument(help='Prepared run directory or JSON file with simulation results or run summary.')], mdmp_cert: Annotated[Optional[Path], typer.Option(help='Signed MDMP artifact required before AI analysis can run.')] = None, mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = 'auto', model: Annotated[str, typer.Option(help='Ollama model name to use.')] = DEFAULT_MINISTRAL_MODEL, minimum_grade: Annotated[str, typer.Option(help='Minimum MDMP grade required to allow analysis.')] = 'research_grade', public_key: Annotated[Optional[Path], typer.Option(help='Explicit MDMP public key for verification.')] = None, trust_store: Annotated[Optional[Path], typer.Option(help='MDMP trust store for verification.')] = None, ollama_host: Annotated[Optional[str], typer.Option(help='Override the Ollama base URL.')] = None, timeout_seconds: Annotated[float, typer.Option(help='HTTP timeout for Ollama generation requests.')] = 120.0, output: Annotated[Optional[Path], typer.Option(help='Optional file path to save the anomaly summary.')] = None) -> None`
- `report(input_json: Annotated[Path, typer.Argument(help='Prepared run directory or JSON file with run-level simulation outputs.')], mdmp_cert: Annotated[Optional[Path], typer.Option(help='Signed MDMP artifact required before AI analysis can run.')] = None, mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = 'auto', model: Annotated[str, typer.Option(help='Ollama model name to use.')] = DEFAULT_MINISTRAL_MODEL, minimum_grade: Annotated[str, typer.Option(help='Minimum MDMP grade required to allow analysis.')] = 'research_grade', public_key: Annotated[Optional[Path], typer.Option(help='Explicit MDMP public key for verification.')] = None, trust_store: Annotated[Optional[Path], typer.Option(help='MDMP trust store for verification.')] = None, ollama_host: Annotated[Optional[str], typer.Option(help='Override the Ollama base URL.')] = None, timeout_seconds: Annotated[float, typer.Option(help='HTTP timeout for Ollama generation requests.')] = 120.0, output: Annotated[Optional[Path], typer.Option(help='Optional file path to save the markdown report.')] = None) -> None`
- `review(input_json: Annotated[Path, typer.Argument(help='Prepared run directory or JSON file with run-level simulation outputs.')], mdmp_cert: Annotated[Optional[Path], typer.Option(help='Signed MDMP artifact required before AI analysis can run.')] = None, mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = 'auto', model: Annotated[str, typer.Option(help='Ollama model name to use.')] = DEFAULT_MINISTRAL_MODEL, minimum_grade: Annotated[str, typer.Option(help='Minimum MDMP grade required to allow analysis.')] = 'research_grade', public_key: Annotated[Optional[Path], typer.Option(help='Explicit MDMP public key for verification.')] = None, trust_store: Annotated[Optional[Path], typer.Option(help='MDMP trust store for verification.')] = None, ollama_host: Annotated[Optional[str], typer.Option(help='Override the Ollama base URL.')] = None, timeout_seconds: Annotated[float, typer.Option(help='HTTP timeout for Ollama generation requests.')] = 120.0, output: Annotated[Optional[Path], typer.Option(help='Optional file path to save the realism review.')] = None) -> None`

## `iints.ai.mdmp_guard`

- Source: `src/iints/ai/mdmp_guard.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `GuardResult` | `GuardResult` | No module docstring. |
| `MDMPGuard` | `MDMPGuard` | Enforce a valid MDMP-signed artifact before AI analysis can run. |

#### `GuardResult` methods

- `to_dict(self) -> dict[str, Any]`

#### `MDMPGuard` methods

- `check(self) -> GuardResult`
- `wrap(self, response: str) -> str`

## `iints.ai.model_catalog`

- Source: `src/iints/ai/model_catalog.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `LocalMistralModelProfile` | `LocalMistralModelProfile` | No module docstring. |

### Public Functions

- `list_local_mistral_models() -> list[LocalMistralModelProfile]`

### Public Constants

- `LOCAL_MISTRAL_MODEL_PROFILES`

## `iints.ai.prepare`

- Source: `src/iints/ai/prepare.py`
- Summary: No module docstring.

### Public Functions

- `prepare_ai_ready_artifacts(run_dir: str | Path, *, create_dev_mdmp_cert: bool = True, grade: str = 'research_grade', expires_days: int = 30, key_dir: str | Path | None = None) -> dict[str, str]`

## `iints.ai.prompts`

- Source: `src/iints/ai/prompts.py`
- Summary: No module docstring.

### Public Functions

- `build_prompt(task: TaskName, payload: Any) -> tuple[str, str]`

### Public Constants

- `MAX_PROMPT_PAYLOAD_CHARS`
- `SYSTEM_PROMPT`
- `TASK_TEMPLATES`

## `iints.analysis`

- Source: `src/iints/analysis/__init__.py`
- Summary: No module docstring.
- Explicit exports: `analyze_run_directory, analyze_study_directory, build_booth_demo, build_carelink_workbench, build_evidence_bundle, ClinicalMetricsCalculator, ClinicalMetricsResult, ClinicalReportGenerator, EVIDENCE_SCOPE, compute_metrics, compare_studies, build_eucys_abstract_draft_markdown, build_eucys_filled_abstract_markdown, build_eucys_jury_qa_markdown, build_eucys_limitations_and_ethics_markdown, build_eucys_poster_outline_markdown, generate_eucys_main_figure, resolve_profile_specs, generate_eucys_results_bundle, generate_results_poster, generate_study_poster, build_algorithm_registry, build_study_design_payload, build_study_protocol_payload, build_study_experiment_template, render_study_protocol_markdown, render_study_experiment_yaml, write_study_protocol_bundle, load_study_experiment_config, load_study_summary, quality_badges_for_metrics, run_baseline_comparison, StudyAlgorithmSpec, StudyArmSpec, StudyComparison, StudyDesignPayload, StudyExperimentConfig, StudyMatrixRow, StudyProfileSpec, StudyRunSummary, StudySummary, write_baseline_comparison`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.analysis.algorithm_xray`

- Source: `src/iints/analysis/algorithm_xray.py`
- Summary: Algorithm X-Ray - IINTS-AF Make invisible medical decisions visible through decision replay and what-if analysis

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `DecisionPoint` | `DecisionPoint` | Single decision point in algorithm timeline |
| `AlgorithmXRay` | `AlgorithmXRay` | X-ray vision into algorithm decision-making process |

#### `DecisionPoint` methods

- `to_dict(self)`

#### `AlgorithmXRay` methods

- `get_quality_report_summary(self) -> Dict[str, Any]`
- `analyze_decision(self, glucose_mgdl: float, glucose_history: List[float], insulin_history: List[float], time_minutes: int) -> DecisionPoint`
- `calculate_personality_profile(self, decision_history: List[DecisionPoint]) -> Dict`
- `generate_decision_replay(self, start_index: int = 0, end_index: Optional[int] = None) -> Dict`
- `compare_what_if_scenarios(self, decision_point: DecisionPoint) -> Dict`
- `export_xray_report(self, filepath: str)`

### Public Functions

- `main()`

## `iints.analysis.baseline`

- Source: `src/iints/analysis/baseline.py`
- Summary: No module docstring.

### Public Functions

- `compute_metrics(results_df: pd.DataFrame) -> Dict[str, float]`
- `run_baseline_comparison(patient_params: Dict[str, Any], stress_event_payloads: List[Dict[str, Any]], duration: int, time_step: int, primary_label: str, primary_results: pd.DataFrame, primary_safety: Dict[str, Any], compare_standard_pump: bool = True, seed: Optional[int] = None) -> Dict[str, Any]`
- `write_baseline_comparison(comparison: Dict[str, Any], output_dir: Path) -> Dict[str, str]`

## `iints.analysis.booth_demo`

- Source: `src/iints/analysis/booth_demo.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `BoothScenarioSpec` | `BoothScenarioSpec` | No module docstring. |
| `ShowcaseRunSpec` | `ShowcaseRunSpec` | No module docstring. |

### Public Functions

- `build_booth_demo(output_dir: str | Path = './results/booth_demo', *, patient_config: str | Path | dict[str, Any] = 'default_patient', duration_minutes: int = 360, time_step: int = 5, seed: int = 42, prepare_ai: bool = True, create_dev_mdmp_cert: bool = True) -> dict[str, str]`

## `iints.analysis.carelink_workbench`

- Source: `src/iints/analysis/carelink_workbench.py`
- Summary: No module docstring.

### Public Functions

- `build_carelink_workbench(input_csv: str | Path, *, output_dir: str | Path = './results/carelink_workbench', scenario_name: str = 'Imported CareLink Scenario', scenario_version: str = '1.0', carb_threshold: float = 0.1, create_dev_mdmp_cert: bool = True, grade: str = 'research_grade', expires_days: int = 30, key_dir: str | Path | None = None) -> dict[str, str]`

## `iints.analysis.clinical_benchmark`

- Source: `src/iints/analysis/clinical_benchmark.py`
- Summary: Clinical Benchmark System Compares AI algorithms against real-world clinical outcomes

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ClinicalBenchmark` | `ClinicalBenchmark` | Clinical benchmark comparison system |

#### `ClinicalBenchmark` methods

- `run_ohio_benchmark(self)`
- `export_benchmark_results(self, results, filename = 'clinical_benchmark.json')`

### Public Functions

- `main()`

## `iints.analysis.clinical_metrics`

- Source: `src/iints/analysis/clinical_metrics.py`
- Summary: Clinical Metrics Calculator - IINTS-AF Standard clinical metrics for diabetes algorithm comparison.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ClinicalMetricsResult` | `ClinicalMetricsResult` | Comprehensive clinical metrics result |
| `ClinicalMetricsCalculator` | `ClinicalMetricsCalculator` | Calculate standard clinical metrics for diabetes management. |

#### `ClinicalMetricsResult` methods

- `to_dict(self) -> Dict`
- `get_summary(self) -> str`
- `get_rating(self) -> str`

#### `ClinicalMetricsCalculator` methods

- `calculate_tir(self, glucose: pd.Series, low: float, high: float) -> float`
- `calculate_all_tir_metrics(self, glucose: pd.Series) -> Dict[str, float]`
- `calculate_gmi(self, glucose: pd.Series) -> float`
- `calculate_cv(self, glucose: pd.Series) -> float`
- `calculate_hypoglycemia_index(self, glucose: pd.Series, timestamp: Optional[pd.Series] = None) -> float`
- `calculate_lbgi(self, glucose: pd.Series) -> float`
- `calculate_hbgi(self, glucose: pd.Series) -> float`
- `calculate_readings_per_day(self, glucose: pd.Series, duration_hours: float) -> float`
- `calculate_data_coverage(self, glucose: pd.Series, expected_interval_minutes: int = 5, duration_hours: float = 24) -> float`
- `calculate(self, glucose: pd.Series, timestamp: Optional[pd.Series] = None, duration_hours: Optional[float] = None) -> ClinicalMetricsResult`
- `compare_metrics(self, metrics1: ClinicalMetricsResult, metrics2: ClinicalMetricsResult) -> Dict[str, Tuple[float, str]]`

### Public Functions

- `demo_clinical_metrics()`

## `iints.analysis.clinical_tir_analyzer`

- Source: `src/iints/analysis/clinical_tir_analyzer.py`
- Summary: Professional 5-Zone TIR Analysis - IINTS-AF Implements Medtronic clinical standard for glucose zone classification

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ClinicalTIRAnalyzer` | `ClinicalTIRAnalyzer` | Professional 5-zone Time in Range analysis following clinical standards |

#### `ClinicalTIRAnalyzer` methods

- `analyze_glucose_zones(self, glucose_values)`

### Public Functions

- `main()`

## `iints.analysis.diabetes_metrics`

- Source: `src/iints/analysis/diabetes_metrics.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `DiabetesMetrics` | `DiabetesMetrics` | Professional diabetes metrics for algorithm evaluation. |

#### `DiabetesMetrics` methods

- `time_in_range(glucose_values, lower = 70, upper = 180)`
- `coefficient_of_variation(glucose_values)`
- `blood_glucose_risk_index(glucose_values, risk_type = 'high')`
- `calculate_all_metrics(df, baseline = 120)`

## `iints.analysis.edge_efficiency`

- Source: `src/iints/analysis/edge_efficiency.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `EnergyEstimate` | `EnergyEstimate` | No module docstring. |

### Public Functions

- `estimate_energy_per_decision(power_watts: float, latency_ms: float, decisions_per_day: int = 288) -> EnergyEstimate`

## `iints.analysis.edge_performance_monitor`

- Source: `src/iints/analysis/edge_performance_monitor.py`
- Summary: Edge AI Performance Monitor - IINTS-AF Jetson Nano performance validation for medical device standards

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `EdgeAIPerformanceMonitor` | `EdgeAIPerformanceMonitor` | Monitor Jetson Nano performance for medical device validation |

#### `EdgeAIPerformanceMonitor` methods

- `start_monitoring(self)`
- `measure_inference_latency(self, inference_function, input_data, iterations = 100)`
- `generate_performance_report(self)`
- `export_performance_data(self, filepath)`

### Public Functions

- `mock_ai_inference(input_data)`
- `main()`

## `iints.analysis.eucys_results`

- Source: `src/iints/analysis/eucys_results.py`
- Summary: No module docstring.

### Public Functions

- `build_eucys_limitations_and_ethics_markdown() -> str`
- `build_eucys_abstract_draft_markdown() -> str`
- `build_eucys_poster_outline_markdown() -> str`
- `build_eucys_jury_qa_markdown() -> str`
- `build_eucys_filled_abstract_markdown(*, root_summary: dict[str, Any], clean_summary: dict[str, Any], study_design: dict[str, Any]) -> str`
- `generate_eucys_main_figure(clean_summary: dict[str, Any], *, output_path: str | Path, csv_output_path: str | Path | None = None) -> dict[str, str]`
- `generate_eucys_results_bundle(study_root: str | Path, *, output_dir: str | Path | None = None) -> dict[str, str]`

### Public Constants

- `EUCYS_ARM_LAYOUT`

## `iints.analysis.evidence_bundle`

- Source: `src/iints/analysis/evidence_bundle.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `EvidenceRun` | `EvidenceRun` | No module docstring. |

#### `EvidenceRun` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `build_evidence_bundle(run_dirs: Iterable[tuple[str, Path]], *, output_dir: Path, title: str = 'IINTS Research Evidence Bundle', local_ai_dir: Optional[Path] = None, pump_bundle_dir: Optional[Path] = None) -> Dict[str, Any]`

### Public Constants

- `EVIDENCE_SCOPE`

## `iints.analysis.explainability`

- Source: `src/iints/analysis/explainability.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ExplainabilityAnalyzer` | `ExplainabilityAnalyzer` | Provides tools for analyzing and explaining the behavior of insulin algorithms. This module implements 'AI as explainability tool' by performing sensitivity analysis. |

#### `ExplainabilityAnalyzer` methods

- `calculate_glucose_variability(self) -> Dict[str, float]`
- `analyze_insulin_response(self, algorithm_type: str = '') -> Dict[str, Any]`
- `perform_sensitivity_analysis(self, algorithm_instance: Any, parameter_name: str, original_value: float, perturbations: List[float], simulation_run_func: Callable[[Any, List[Any], int], pd.DataFrame], fixed_events: Optional[List[Any]] = None, duration_minutes: int = 1440) -> Dict[float, Dict[str, Union[float, str]]]`

## `iints.analysis.explainable_ai`

- Source: `src/iints/analysis/explainable_ai.py`
- Summary: Explainable AI Audit Trail - IINTS-AF Clinical decision transparency system for medical AI validation

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ClinicalAuditTrail` | `ClinicalAuditTrail` | Explainable AI system for clinical decision transparency |

#### `ClinicalAuditTrail` methods

- `log_decision(self, timestamp, glucose_current, glucose_trend, insulin_decision, algorithm_confidence, safety_override = False, context = None)`
- `generate_clinical_summary(self, hours = 24)`
- `export_audit_trail(self, filepath)`

### Public Functions

- `main()`

## `iints.analysis.hardware_benchmark`

- Source: `src/iints/analysis/hardware_benchmark.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PerformanceMetrics` | `PerformanceMetrics` | No module docstring. |
| `HardwareBenchmark` | `HardwareBenchmark` | Hardware performance monitoring for Jetson and other platforms. |

#### `HardwareBenchmark` methods

- `start_monitoring(self)`
- `stop_monitoring(self)`
- `benchmark_algorithm(self, algorithm, test_data, iterations = 100)`
- `get_current_metrics(self) -> Dict`
- `get_metrics_summary(self) -> Dict`
- `export_metrics(self, filename: str)`
- `clear_metrics(self)`

## `iints.analysis.metrics`

- Source: `src/iints/analysis/metrics.py`
- Summary: No module docstring.

### Public Functions

- `calculate_tir(df: pd.DataFrame, lower_bound: float = 70.0, upper_bound: float = 180.0) -> float`
- `calculate_hypoglycemia(df: pd.DataFrame, threshold: float = 70.0) -> float`
- `calculate_hyperglycemia(df: pd.DataFrame, threshold: float = 180.0) -> float`
- `calculate_average_glucose(df: pd.DataFrame) -> float`
- `generate_benchmark_metrics(df: pd.DataFrame) -> Dict[str, float]`

## `iints.analysis.population_report`

- Source: `src/iints/analysis/population_report.py`
- Summary: Population Report Generator — IINTS-AF ======================================== Generates a PDF report with aggregate statistics and visualisations for a Monte Carlo population evaluation run.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PopulationReportGenerator` | `PopulationReportGenerator` | Generate a PDF report for population simulation results. |

#### `PopulationReportGenerator` methods

- `generate_pdf(self, summary_df: pd.DataFrame, aggregate_metrics: Dict[str, Any], aggregate_safety: Dict[str, Any], output_path: str, title: str = 'IINTS-AF Population Evaluation Report') -> str`

## `iints.analysis.poster`

- Source: `src/iints/analysis/poster.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PosterScenario` | `PosterScenario` | No module docstring. |

#### `PosterScenario` methods

- `to_dict(self) -> dict[str, Any]`

### Public Functions

- `generate_results_poster(run_dirs: Sequence[str | Path] | None = None, *, labels: Sequence[str] | None = None, output_path: str | Path = './results/posters/iints_results_poster.png', poster_title: str = '288 Decisions. Every Day. We Test Them All.', subtitle: str = 'Three IINTS-AF scenarios showing control, stress handling, and supervisor protection.', results_root: str | Path = './results', auto_limit: int = 3, summary_output_path: str | Path | None = None) -> dict[str, str]`

## `iints.analysis.reporting`

- Source: `src/iints/analysis/reporting.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ClinicalReportGenerator` | `ClinicalReportGenerator` | Generate a clean, publication-ready PDF report. |

#### `ClinicalReportGenerator` methods

- `generate_agp_pdf(self, simulation_data: pd.DataFrame, output_path: str, *, title: str = 'IINTS Research AGP-Style Report', subject_name: str = 'Research simulation', safety_report: Optional[Dict[str, Any]] = None, target_low: float = 70.0, target_high: float = 180.0, summary_json_path: Optional[str] = None) -> str`
- `export_agp_assets(self, simulation_data: pd.DataFrame, output_dir: str, *, subject_name: str = 'Research simulation', target_low: float = 70.0, target_high: float = 180.0, summary_json_path: Optional[str] = None) -> Dict[str, str]`
- `export_plots(self, simulation_data: pd.DataFrame, output_dir: str) -> Dict[str, str]`
- `generate_pdf(self, simulation_data: pd.DataFrame, safety_report: Dict[str, Any], output_path: str, title: str = 'IINTS-AF Clinical Report') -> str`
- `generate_demo_pdf(self, simulation_data: pd.DataFrame, safety_report: Dict[str, Any], output_path: str, title: str = 'IINTS-AF Demo Report') -> str`

## `iints.analysis.run_quality`

- Source: `src/iints/analysis/run_quality.py`
- Summary: No module docstring.

### Public Functions

- `standardize_simulation_for_realism(results_df: pd.DataFrame) -> pd.DataFrame`
- `write_run_quality_artifacts(results_df: pd.DataFrame, output_dir: str | Path, *, run_label: Optional[str] = None, safety_report: Optional[Dict[str, Any]] = None, realism_reference: str = 'free_living_t1d') -> Dict[str, Any]`

## `iints.analysis.safety_index`

- Source: `src/iints/analysis/safety_index.py`
- Summary: IINTS-AF Safety Index ===================== A single composite metric (0–100) that summarises the clinical safety of an insulin dosing algorithm's output. Higher is safer.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SafetyIndexResult` | `SafetyIndexResult` | Result of the IINTS Safety Index computation. |

#### `SafetyIndexResult` methods

- `to_dict(self) -> dict`

### Public Functions

- `compute_safety_index(results_df: pd.DataFrame, safety_report: Dict, duration_minutes: int, weights: Optional[Dict[str, float]] = None, time_step_minutes: float = 5.0) -> SafetyIndexResult`
- `safety_weights_from_cli(w_below54: Optional[float] = None, w_below70: Optional[float] = None, w_supervisor: Optional[float] = None, w_recovery: Optional[float] = None, w_tail: Optional[float] = None) -> Dict[str, float]`

### Public Constants

- `DEFAULT_WEIGHTS`
- `NORM_SCALES`

## `iints.analysis.safety_visualizer`

- Source: `src/iints/analysis/safety_visualizer.py`
- Summary: No module docstring.

### Public Functions

- `summarize_safety_trace(results_df: pd.DataFrame, safety_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
- `build_safety_visualizer_html(results_df: pd.DataFrame, safety_report: Optional[Dict[str, Any]] = None, *, title: str = 'IINTS Safety Contract Visualizer') -> str`
- `write_safety_visualizer(results_df: pd.DataFrame, output_html: str | Path, *, output_json: str | Path | None = None, safety_report: Optional[Dict[str, Any]] = None, title: str = 'IINTS Safety Contract Visualizer') -> Dict[str, str]`

## `iints.analysis.sensor_filtering`

- Source: `src/iints/analysis/sensor_filtering.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SensorNoiseModel` | `SensorNoiseModel` | Realistic sensor noise with drift and filtering. |
| `KalmanFilter` | `KalmanFilter` | Simple Kalman filter for glucose smoothing. |
| `MovingAverageFilter` | `MovingAverageFilter` | Simple moving average filter. |

#### `SensorNoiseModel` methods

- `add_noise(self, true_glucose, time_step)`

#### `KalmanFilter` methods

- `update(self, measurement)`

#### `MovingAverageFilter` methods

- `update(self, value)`

## `iints.analysis.study_analysis`

- Source: `src/iints/analysis/study_analysis.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `StudyRunSummary` | `StudyRunSummary` | No module docstring. |
| `StudySummary` | `StudySummary` | No module docstring. |
| `StudyComparison` | `StudyComparison` | No module docstring. |

#### `StudyRunSummary` methods

- `to_dict(self) -> dict[str, Any]`

#### `StudySummary` methods

- `to_dict(self) -> dict[str, Any]`

#### `StudyComparison` methods

- `to_dict(self) -> dict[str, Any]`

### Public Functions

- `quality_badges_for_metrics(metrics: dict[str, float], *, certified: bool | None = None) -> list[str]`
- `analyze_run_directory(run_dir: Path) -> StudyRunSummary`
- `analyze_study_directory(study_dir: Path, *, external_reference_metrics: Path | None = None) -> StudySummary`
- `load_study_summary(path: str | Path) -> StudySummary`
- `compare_studies(left: str | Path, right: str | Path, *, left_label: str | None = None, right_label: str | None = None) -> StudyComparison`

### Public Constants

- `CALIBRATION_METRICS`
- `CORE_METRIC_KEYS`
- `PAIRWISE_METRICS`

## `iints.analysis.study_engine`

- Source: `src/iints/analysis/study_engine.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `StudyProfileSpec` | `StudyProfileSpec` | No module docstring. |
| `StudyAlgorithmSpec` | `StudyAlgorithmSpec` | No module docstring. |
| `StudyArmSpec` | `StudyArmSpec` | No module docstring. |
| `StudyMatrixRow` | `StudyMatrixRow` | No module docstring. |
| `StudyDesignPayload` | `StudyDesignPayload` | No module docstring. |

#### `StudyProfileSpec` methods

- `to_dict(self) -> dict[str, str]`

#### `StudyAlgorithmSpec` methods

- `to_dict(self) -> dict[str, str]`

#### `StudyArmSpec` methods

- `to_dict(self) -> dict[str, Any]`

#### `StudyMatrixRow` methods

- `to_dict(self) -> dict[str, Any]`

#### `StudyDesignPayload` methods

- `to_dict(self) -> dict[str, Any]`

### Public Functions

- `slugify_study_token(value: str) -> str`
- `resolve_profile_specs(profile_set: str = DEFAULT_PROFILE_SET) -> list[StudyProfileSpec]`
- `build_algorithm_registry(*, candidate_algorithm: str, candidate_source_type: str = 'path', candidate_source_ref: str | None = None, include_default_baselines: bool = True, extra_algorithms: list[str] | None = None) -> list[StudyAlgorithmSpec]`
- `build_study_design_payload(*, preset: str = 'default', title: str = 'IINTS Scientific Validation Protocol', primary_hypothesis: str | None = None, scenarios: list[str] | None = None, seeds: list[int] | None = None, candidate_algorithm: str = 'your_algorithm', include_default_baselines: bool = True, extra_algorithms: list[str] | None = None, profile_set: str = DEFAULT_PROFILE_SET, external_reference_label: str = 'CareLink personal workbench metrics', candidate_source_type: str = 'path', candidate_source_ref: str | None = None) -> StudyDesignPayload`

### Public Constants

- `DEFAULT_BASELINE_ALGORITHMS`
- `DEFAULT_HYPOTHESES`
- `DEFAULT_METRICS`
- `DEFAULT_PROFILE_SET`
- `DEFAULT_RESEARCH_QUESTION`

## `iints.analysis.study_experiment`

- Source: `src/iints/analysis/study_experiment.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `StudyExperimentConfig` | `StudyExperimentConfig` | No module docstring. |

### Public Functions

- `load_study_experiment_config(path: str | Path) -> StudyExperimentConfig`
- `build_study_experiment_template(*, preset: str, title: str, profile_set: str, seeds: list[int], candidate_algorithm: str, scenarios: list[str], include_default_baselines: bool, extra_algorithms: list[str] | None = None, external_reference_label: str = 'CareLink personal workbench metrics', default_output_dir: str | None = None) -> dict[str, Any]`
- `render_study_experiment_yaml(payload: dict[str, Any]) -> str`

## `iints.analysis.study_poster`

- Source: `src/iints/analysis/study_poster.py`
- Summary: No module docstring.

### Public Functions

- `generate_study_poster(summary_input: str | Path | StudySummary, *, output_path: str | Path = 'results/study_poster.png', title: str = 'IINTS Study Results', subtitle: str = 'Simulation evidence across runs, safety behavior, and certification quality.', summary_output_path: str | Path | None = None) -> dict[str, str]`

## `iints.analysis.study_protocol`

- Source: `src/iints/analysis/study_protocol.py`
- Summary: No module docstring.

### Public Functions

- `build_study_protocol_payload(*, preset: str = 'default', title: str = 'IINTS Scientific Validation Protocol', primary_hypothesis: str | None = None, scenarios: list[str] | None = None, seeds: list[int] | None = None, algorithms: list[str] | None = None, corruption_modes: list[str] | None = None, external_reference_label: str = 'CareLink personal workbench metrics', profile_set: str = DEFAULT_PROFILE_SET, include_default_baselines: bool = True, extra_algorithms: list[str] | None = None) -> dict[str, Any]`
- `render_study_protocol_markdown(payload: dict[str, Any]) -> str`
- `write_study_protocol_bundle(output_dir: str | Path, *, preset: str = 'default', title: str = 'IINTS Scientific Validation Protocol', primary_hypothesis: str | None = None, scenarios: list[str] | None = None, seeds: list[int] | None = None, algorithms: list[str] | None = None, corruption_modes: list[str] | None = None, external_reference_label: str = 'CareLink personal workbench metrics', profile_set: str = DEFAULT_PROFILE_SET, include_default_baselines: bool = True, extra_algorithms: list[str] | None = None) -> dict[str, str]`

### Public Constants

- `DEFAULT_CORRUPTION_MODES`
- `DEFAULT_SCENARIOS`

## `iints.analysis.validator`

- Source: `src/iints/analysis/validator.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ReliabilityLevel` | `ReliabilityLevel(Enum)` | No module docstring. |
| `ValidationResult` | `ValidationResult` | No module docstring. |
| `DataIntegrityValidator` | `DataIntegrityValidator` | Validates data integrity for reverse engineering analysis. |
| `AlgorithmicDriftDetector` | `AlgorithmicDriftDetector` | Detects when AI algorithms drift from safe baseline behavior. |
| `StatisticalReliabilityChecker` | `StatisticalReliabilityChecker` | Checks statistical reliability of Monte Carlo results. |
| `ReverseEngineeringValidator` | `ReverseEngineeringValidator` | Main validator for reverse engineering analysis. |

#### `DataIntegrityValidator` methods

- `validate_glucose_data(self, glucose_values: List[float], timestamps: List[float]) -> ValidationResult`
- `validate_insulin_data(self, insulin_values: List[float]) -> ValidationResult`

#### `AlgorithmicDriftDetector` methods

- `detect_drift(self, ai_outputs: List[float], baseline_outputs: List[float]) -> ValidationResult`

#### `StatisticalReliabilityChecker` methods

- `check_monte_carlo_reliability(self, results: List[List[float]]) -> ValidationResult`

#### `ReverseEngineeringValidator` methods

- `validate_simulation_results(self, simulation_df: pd.DataFrame, baseline_results: Optional[List[float]] = None, monte_carlo_results: Optional[List[List[float]]] = None) -> Dict[str, ValidationResult]`
- `generate_reliability_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]`

## `iints.api`

- Source: `src/iints/api/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.api.base_algorithm`

- Source: `src/iints/api/base_algorithm.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `AlgorithmInput` | `AlgorithmInput` | Dataclass for inputs to the insulin prediction algorithm. |
| `AlgorithmMetadata` | `AlgorithmMetadata` | Metadata for algorithm registration and identification |
| `AlgorithmResult` | `AlgorithmResult` | Result of an insulin prediction with uncertainty |
| `WhyLogEntry` | `WhyLogEntry` | Single entry in the Why Log explaining a decision reason |
| `InsulinAlgorithm` | `InsulinAlgorithm(ABC)` | Abstract base class for insulin delivery algorithms. |

#### `AlgorithmMetadata` methods

- `to_dict(self) -> Dict`

#### `AlgorithmResult` methods

- `to_dict(self) -> Dict`

#### `WhyLogEntry` methods

- `to_dict(self) -> Dict`

#### `InsulinAlgorithm` methods

- `set_isf(self, isf: float)`
- `set_icr(self, icr: float)`
- `get_algorithm_metadata(self) -> AlgorithmMetadata`
- `set_algorithm_metadata(self, metadata: AlgorithmMetadata)`
- `calculate_uncertainty(self, data: AlgorithmInput) -> float`
- `calculate_confidence_interval(self, data: AlgorithmInput, prediction: float, uncertainty: float) -> tuple`
- `explain_prediction(self, data: AlgorithmInput, prediction: Dict[str, Any]) -> str`
- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`
- `get_why_log(self) -> List[WhyLogEntry]`
- `get_why_log_text(self) -> str`
- `reset(self)`
- `get_state(self) -> Dict[str, Any]`
- `set_state(self, state: Dict[str, Any])`

## `iints.api.registry`

- Source: `src/iints/api/registry.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `AlgorithmListing` | `AlgorithmListing` | No module docstring. |
| `LocalPluginRecord` | `LocalPluginRecord` | No module docstring. |

#### `LocalPluginRecord` methods

- `from_dict(cls, payload: Mapping[str, Any]) -> 'LocalPluginRecord'`
- `to_dict(self) -> dict[str, Any]`

### Public Functions

- `get_plugin_home() -> Path`
- `get_plugin_registry_path() -> Path`
- `list_local_plugin_records(kind: str | None = None) -> list[LocalPluginRecord]`
- `install_file_plugin(kind: str, source_path: str | Path, name: str | None = None) -> LocalPluginRecord`
- `install_algorithm_plugin(source_path: str | Path, name: str | None = None) -> LocalPluginRecord`
- `uninstall_local_plugin(name: str, kind: str | None = None, *, remove_file: bool = False) -> bool`
- `list_algorithm_plugins() -> List[AlgorithmListing]`

### Public Constants

- `IINTS_PLUGIN_HOME_ENV`
- `PLUGIN_REGISTRY_SCHEMA_VERSION`
- `SUPPORTED_LOCAL_PLUGIN_KINDS`

## `iints.api.template_algorithm`

- Source: `src/iints/api/template_algorithm.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `TemplateAlgorithm` | `TemplateAlgorithm(InsulinAlgorithm)` | This is a template for a user-defined insulin delivery algorithm. It demonstrates the required structure and provides a basic, safe implementation of a correction and meal bolus logic. |

#### `TemplateAlgorithm` methods

- `get_algorithm_metadata(self) -> AlgorithmMetadata`
- `predict_insulin(self, data: AlgorithmInput) -> Dose`
- `reset(self)`

## `iints.cli`

- Source: `src/iints/cli/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.cli.cli`

- Source: `src/iints/cli/cli.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `BrandedTyperGroup` | `BrandedTyperGroup(TyperGroup)` | No module docstring. |

#### `BrandedTyperGroup` methods

- `get_help(self, ctx)`
- `list_commands(self, ctx)`

### Public Functions

- `app_callback(ctx: typer.Context, version: Annotated[bool, typer.Option('--version', help='Show the installed SDK version and exit.', is_eager=True)] = False)`
- `evaluate(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], population: Annotated[int, typer.Option(help='Number of virtual patients to simulate')] = 100, patient_config_name: Annotated[str, typer.Option('--patient-config', help='Base patient configuration name')] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option('--patient-config-path', help='Path to base patient config YAML')] = None, scenario_path: Annotated[Optional[Path], typer.Option('--scenario', help='Path to scenario JSON')] = None, duration: Annotated[int, typer.Option(help='Simulation duration in minutes')] = 720, time_step: Annotated[int, typer.Option(help='Time step in minutes')] = 5, output_dir: Annotated[Optional[Path], typer.Option(help='Output directory')] = None, max_workers: Annotated[Optional[int], typer.Option(help='Max parallel workers (default: all cores)')] = None, seed: Annotated[Optional[int], typer.Option(help='Random seed for reproducibility')] = None, patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model: auto, bergman, custom, simglucose')] = 'auto')`
- `doctor(smoke_run: Annotated[bool, typer.Option(help='Run a short deterministic smoke simulation')] = False, smoke_duration: Annotated[int, typer.Option(help='Smoke simulation duration in minutes')] = 30, full: Annotated[bool, typer.Option('--full', help='Run extra environment checks that help beginners debug setup issues')] = False, suggest: Annotated[bool, typer.Option('--suggest', help='Print concrete next commands based on what doctor finds')] = False)`
- `update_sdk(source: Annotated[str, typer.Option(help='Update source: pypi or github.')] = 'pypi', install_extras: Annotated[str, typer.Option('--extras', help='Comma-separated extras to install, e.g. full,mdmp,research,edge. Use empty string for none.')] = 'full,mdmp,research,edge', github_ref: Annotated[str, typer.Option('--github-ref', help='Git ref used with --source github.')] = 'main', user: Annotated[bool, typer.Option('--user/--no-user', help='Pass --user to pip for user-site installs.')] = False, pre: Annotated[bool, typer.Option('--pre/--no-pre', help='Allow pre-release versions.')] = False, upgrade_pip: Annotated[bool, typer.Option('--upgrade-pip/--no-upgrade-pip', help='Upgrade pip before installing IINTS.')] = False, dry_run: Annotated[bool, typer.Option('--dry-run', help='Print the pip command without executing it.')] = False, yes: Annotated[bool, typer.Option('--yes', '-y', help='Run without asking for confirmation.')] = False) -> None`
- `run_doctor(algo: Annotated[Optional[Path], typer.Option(help='Algorithm Python file to inspect.')] = None, patient_config_path: Annotated[Optional[Path], typer.Option(help='Patient YAML used for the run.')] = None, scenario_path: Annotated[Optional[Path], typer.Option('--scenario-path', '--scenario', help='Scenario JSON used for the run.')] = None, duration: Annotated[int, typer.Option(help='Requested duration in minutes.')] = 1440, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes.')] = 5, output_dir: Annotated[Optional[Path], typer.Option(help='Output directory to preflight.')] = None, patient_config_name: Annotated[str, typer.Option(help='Packaged patient profile name, if not using a YAML path.')] = 'default_patient', output_json: Annotated[Optional[Path], typer.Option(help='Optional JSON report path.')] = None, fail_on_warning: Annotated[bool, typer.Option(help='Exit with code 1 when any warning is found.')] = False) -> None`
- `evidence_build(run: Annotated[List[str], typer.Option('--run', help='Repeatable run input in label=path form, for example normal=results/normal.')], output_dir: Annotated[Path, typer.Option(help='Output directory for the evidence bundle.')] = Path('results/evidence_bundle'), title: Annotated[str, typer.Option(help='Human-readable bundle title.')] = 'IINTS Research Evidence Bundle', local_ai_dir: Annotated[Optional[Path], typer.Option(help='Optional local AI lab output directory.')] = None, pump_bundle_dir: Annotated[Optional[Path], typer.Option(help='Optional Pico pump bundle directory.')] = None) -> None`
- `validation_profiles(profiles_path: Annotated[Optional[Path], typer.Option(help='Optional custom profiles YAML')] = None)`
- `replay_check(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt)')] = None, patient_config_name: Annotated[str, typer.Option(help='Patient config name')] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option(help='Patient config YAML path')] = None, scenario_path: Annotated[Optional[Path], typer.Option(help='Path to scenario JSON')] = None, duration: Annotated[int, typer.Option(help='Simulation duration in minutes')] = 240, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, seed: Annotated[int, typer.Option(help='Deterministic seed')] = 42, repeats: Annotated[int, typer.Option(help='Replay runs to compare')] = 2, output_json: Annotated[Optional[Path], typer.Option(help='Optional output JSON path')] = None, fail_on_mismatch: Annotated[bool, typer.Option(help='Exit code 1 if replay mismatch is detected')] = True)`
- `golden_benchmark(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt)')] = None, pack_path: Annotated[Optional[Path], typer.Option(help='Optional golden benchmark YAML path')] = None, output_dir: Annotated[Optional[Path], typer.Option(help='Optional output directory for benchmark runs')] = None, seed: Annotated[int, typer.Option(help='Random seed')] = 42, duration_override: Annotated[Optional[int], typer.Option(help='Optional duration override (minutes)')] = None, output_json: Annotated[Optional[Path], typer.Option(help='Write benchmark report JSON')] = None, fail_on_check: Annotated[bool, typer.Option(help='Exit code 1 when any scenario fails')] = True)`
- `validate_run(results_csv: Annotated[Path, typer.Option(help='Path to simulation results CSV')], profile: Annotated[str, typer.Option(help='Validation profile id')] = 'research_default', safety_report_path: Annotated[Optional[Path], typer.Option(help='Optional safety report JSON')] = None, duration_minutes: Annotated[Optional[int], typer.Option(help='Override run duration (minutes)')] = None, profiles_path: Annotated[Optional[Path], typer.Option(help='Optional custom profiles YAML')] = None, output_json: Annotated[Optional[Path], typer.Option(help='Write validation report JSON')] = None, fail_on_check: Annotated[bool, typer.Option(help='Exit with code 1 when required checks fail')] = True)`
- `analyze(study_dir: Annotated[Path, typer.Argument(help='Directory containing one or more run folders.')], output_json: Annotated[Path, typer.Option(help='Write aggregated study summary JSON')] = Path('results/study_summary.json'), output_markdown: Annotated[Optional[Path], typer.Option(help='Optional markdown summary path')] = None, output_csv: Annotated[Optional[Path], typer.Option(help='Optional evidence-table CSV path')] = None, output_evidence_markdown: Annotated[Optional[Path], typer.Option(help='Optional evidence-table markdown path')] = None, carelink_metrics: Annotated[Optional[Path], typer.Option(help='Optional CareLink metrics JSON or workbench directory for external plausibility comparison')] = None) -> None`
- `compare_study(left: Annotated[Path, typer.Argument(help='Left study directory or study summary JSON')], right: Annotated[Path, typer.Argument(help='Right study directory or study summary JSON')], output_json: Annotated[Path, typer.Option(help='Write comparison JSON')] = Path('results/study_comparison.json'), output_markdown: Annotated[Optional[Path], typer.Option(help='Optional markdown comparison path')] = None, left_label: Annotated[Optional[str], typer.Option(help='Optional display label for the left study')] = None, right_label: Annotated[Optional[str], typer.Option(help='Optional display label for the right study')] = None) -> None`
- `poster_study(study_input: Annotated[Path, typer.Argument(help='Study directory or study summary JSON')], output_path: Annotated[Path, typer.Option(help='Output PNG path')] = Path('results/study_poster.png'), title: Annotated[str, typer.Option(help='Poster title')] = 'IINTS Study Results', subtitle: Annotated[str, typer.Option(help='Poster subtitle')] = 'Simulation evidence across runs, safety behavior, and certification quality.') -> None`
- `demo_expo(output_dir: Annotated[Path, typer.Option(help='Directory for the expo demo bundle')] = Path('results/expo_demo'), patient_config: Annotated[str, typer.Option(help='Patient profile name or path-like identifier')] = 'default_patient', duration_minutes: Annotated[int, typer.Option(help='Scenario duration in minutes')] = 360, seed: Annotated[int, typer.Option(help='Random seed for reproducible expo runs')] = 42) -> None`
- `run_study(ctx: typer.Context, algo: Annotated[Optional[Path], typer.Option(help='Path to the algorithm Python file')] = None, experiment: Annotated[Optional[Path], typer.Option(help='Optional study experiment YAML file')] = None, output_dir: Annotated[Path, typer.Option(help='Root directory for the study bundle')] = Path('results/study_bundle'), preset: Annotated[str, typer.Option(help='Study preset: default or eucys')] = 'default', profile_set: Annotated[str, typer.Option(help='Patient profile set to evaluate')] = DEFAULT_PROFILE_SET, scenarios: Annotated[str, typer.Option(help='Optional comma-separated scenario slugs to run')] = '', seeds: Annotated[str, typer.Option(help='Comma-separated seed list')] = '1,2,3,4,5', duration: Annotated[Optional[int], typer.Option(help='Override scenario duration in minutes')] = None, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, carelink_metrics: Annotated[Optional[Path], typer.Option(help='Optional CareLink metrics JSON or workbench directory for plausibility comparison')] = None, prepare_ai: Annotated[bool, typer.Option(help='Generate AI-ready artifacts for non-corrupted study arms')] = True, reference_csv: Annotated[Optional[Path], typer.Option(help='Optional source CSV to corrupt and archive alongside the study protocol')] = None, include_default_baselines: Annotated[bool, typer.Option('--include-default-baselines/--no-include-default-baselines', help='Include the default baseline registry in the study matrix')] = True, extra_algorithms: Annotated[str, typer.Option(help='Comma-separated additional comparison algorithm labels')] = '', external_reference_label: Annotated[str, typer.Option(help='Label for the plausibility reference lane')] = 'CareLink personal workbench metrics', gate_profile: Annotated[Optional[str], typer.Option(help='Optional calibration gate profile id')] = None, gate_profiles_path: Annotated[Optional[Path], typer.Option(help='Optional calibration gate profiles YAML path')] = None, fail_on_gate: Annotated[bool, typer.Option(help='Exit with code 1 when any run fails the calibration gate')] = False) -> None`
- `run_eucys_study(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], patient_config_name: Annotated[str, typer.Option(help='Legacy option kept for compatibility; the EUCYS preset now uses the fixed clinic_safe_core profile set')] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option(help='Legacy option kept for compatibility; the EUCYS preset now uses the fixed clinic_safe_core profile set')] = None, output_dir: Annotated[Path, typer.Option(help='Root directory for the EUCYS study bundle')] = Path('results/eucys_study'), seeds: Annotated[str, typer.Option(help='Comma-separated seed list')] = '1,2,3,4,5,6,7,8,9,10', duration: Annotated[Optional[int], typer.Option(help='Override scenario duration in minutes')] = None, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, carelink_metrics: Annotated[Optional[Path], typer.Option(help='Optional CareLink metrics JSON or workbench directory for plausibility comparison')] = None, prepare_ai: Annotated[bool, typer.Option(help='Generate AI-ready artifacts for certified study arms')] = True, reference_csv: Annotated[Optional[Path], typer.Option(help='Optional source CSV to corrupt and archive alongside the study protocol')] = None, include_default_baselines: Annotated[bool, typer.Option('--include-default-baselines/--no-include-default-baselines', help='Include the default baseline registry in the EUCYS matrix')] = True, gate_profile: Annotated[Optional[str], typer.Option(help='Optional calibration gate profile id')] = None, gate_profiles_path: Annotated[Optional[Path], typer.Option(help='Optional calibration gate profiles YAML path')] = None, fail_on_gate: Annotated[bool, typer.Option(help='Exit with code 1 when any run fails the calibration gate')] = False) -> None`
- `eucys_results(study_dir: Annotated[Path, typer.Argument(help='Root directory of a completed EUCYS or run-study bundle.')], output_dir: Annotated[Optional[Path], typer.Option(help='Optional output directory for the packaged EUCYS results bundle. Defaults to <study_dir>/EUCYS_RESULTS.')] = None) -> None`
- `study_protocol(ctx: typer.Context, output_dir: Annotated[Path, typer.Option(help='Directory where the protocol bundle should be written')] = Path('results/study_protocol'), experiment: Annotated[Optional[Path], typer.Option(help='Optional study experiment YAML file to seed the protocol bundle')] = None, preset: Annotated[str, typer.Option(help='Protocol preset: default or eucys')] = 'default', title: Annotated[str, typer.Option(help='Protocol title')] = 'IINTS Scientific Validation Protocol', primary_hypothesis: Annotated[Optional[str], typer.Option(help='Optional override for the primary H1 hypothesis')] = None, seeds: Annotated[str, typer.Option(help='Comma-separated seed list')] = '1,2,3,4,5', algorithms: Annotated[str, typer.Option(help='Comma-separated candidate+comparison algorithms (legacy; first entry becomes the candidate)')] = 'your_algorithm', scenarios: Annotated[str, typer.Option(help='Comma-separated scenario names')] = 'baseline_day,meal_challenge,exercise_challenge,supervisor_override', profile_set: Annotated[str, typer.Option(help='Patient profile set to encode in the study protocol')] = DEFAULT_PROFILE_SET, include_default_baselines: Annotated[bool, typer.Option('--include-default-baselines/--no-include-default-baselines', help='Include the default baseline registry in the protocol bundle')] = True, extra_algorithms: Annotated[str, typer.Option(help='Comma-separated additional algorithm labels for the protocol bundle')] = '', corruption_modes: Annotated[str, typer.Option(help='Comma-separated corruption modes for the protocol plan')] = ','.join(AVAILABLE_STUDY_CORRUPTIONS), external_reference_label: Annotated[str, typer.Option(help='Label for the real-world plausibility reference')] = 'CareLink personal workbench metrics') -> None`
- `contract_verify(contract_path: Annotated[Optional[Path], typer.Option(help='Optional YAML safety-contract file')] = None, glucose_min: Annotated[float, typer.Option(help='Minimum glucose to test (mg/dL)')] = 50.0, glucose_max: Annotated[float, typer.Option(help='Maximum glucose to test (mg/dL)')] = 160.0, glucose_step: Annotated[float, typer.Option(help='Glucose grid step (mg/dL)')] = 5.0, trend_min: Annotated[float, typer.Option(help='Minimum trend to test (mg/dL/min)')] = -5.0, trend_max: Annotated[float, typer.Option(help='Maximum trend to test (mg/dL/min)')] = 3.0, trend_step: Annotated[float, typer.Option(help='Trend grid step (mg/dL/min)')] = 0.5, proposed_doses: Annotated[str, typer.Option(help='Comma-separated proposed insulin doses (U)')] = '0.0,0.5,1.0,2.0,4.0', iob_values: Annotated[Optional[str], typer.Option(help='Optional comma-separated current IOB values (U)')] = None, output_json: Annotated[Optional[Path], typer.Option(help='Write contract verification report JSON')] = None, fail_on_violation: Annotated[bool, typer.Option(help='Exit with code 1 when any contract violation is found')] = True)`
- `certify_run(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], profile: Annotated[str, typer.Option(help='Validation profile id')] = 'research_default', predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt) for dual-guard forecasting')] = None, patient_config_name: Annotated[str, typer.Option(help='Name of the patient configuration')] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option(help='Path to a patient config YAML')] = None, scenario_path: Annotated[Optional[Path], typer.Option(help='Path to scenario JSON')] = None, duration: Annotated[int, typer.Option(help='Simulation duration in minutes')] = 720, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, output_dir: Annotated[Path, typer.Option(help='Directory to save outputs')] = Path('results/certified_run'), seed: Annotated[Optional[int], typer.Option(help='Random seed')] = None, export_sources: Annotated[bool, typer.Option(help='Write sources_manifest.json with peer-reviewed references.')] = True, source_category: Annotated[Optional[str], typer.Option(help='Optional evidence source category filter (guideline, trial, model, ...).')] = None, write_summary: Annotated[bool, typer.Option(help='Write SUMMARY.md with key artifacts and next steps.')] = True, fail_on_check: Annotated[bool, typer.Option(help='Exit code 1 when validation profile fails')] = True)`
- `study_ready(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], scenario_path: Annotated[Optional[Path], typer.Option(help='Path to scenario JSON')] = None, output_dir: Annotated[Path, typer.Option(help='Directory to save outputs')] = Path('results/study_ready'), duration: Annotated[int, typer.Option(help='Simulation duration in minutes')] = 720, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, seed: Annotated[Optional[int], typer.Option(help='Random seed')] = None, profile: Annotated[str, typer.Option(help='Validation profile id')] = 'research_default', predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt)')] = None, patient_config_name: Annotated[str, typer.Option(help='Patient configuration name')] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option(help='Optional patient config YAML path')] = None, fail_on_check: Annotated[bool, typer.Option(help='Exit code 1 when validation profile fails')] = True)`
- `init(project_name: Annotated[str, typer.Option(help='Name of the project directory')] = 'my_iints_project', template: Annotated[str, typer.Option(help='Project template: research or clinical-trial')] = 'research')`
- `quickstart(project_name: Annotated[str, typer.Option(help='Name of the project directory')] = 'iints_quickstart')`
- `demo(demo_kind: Annotated[Optional[str], typer.Argument(help='Optional story preset: doctor, eucys, booth, live, quick, or full. Example: `iints demo eucys`.')] = None, output_dir: Annotated[Path, typer.Option(help='Directory where demo outputs should be written')] = Path('results/demo'), mode: Annotated[str, typer.Option('--mode', help='Demo mode: live, quick, or full')] = 'live', quick_mode: Annotated[bool, typer.Option('--quick', help='Shortcut for --mode quick')] = False, full_mode: Annotated[bool, typer.Option('--full', help='Shortcut for --mode full')] = False, presentation_mode: Annotated[bool, typer.Option('--presentation/--simulation-only', help='Default starts the full presentation demo; use --simulation-only for the old one-run starter simulation.')] = True, preset: Annotated[Optional[str], typer.Option(help='Optional preset override. Defaults to quickstart_meal for quick and realistic_reference_day for full.')] = None, seed: Annotated[int, typer.Option(help='Deterministic seed for the bundled demo')] = 42, compare_baselines: Annotated[bool, typer.Option(help='Include built-in baselines in the demo output')] = True, audience: Annotated[str, typer.Option(help='Presenter framing for the live demo: mixed, clinical, engineering, or jury.')] = 'jury', prepare_ai: Annotated[bool, typer.Option('--prepare-ai/--skip-ai', help='Prepare optional local-AI artifacts during the live demo.')] = False, build_evidence: Annotated[bool, typer.Option('--evidence/--no-evidence', help='Build a public research evidence bundle after the demo.')] = True, full_code: Annotated[bool, typer.Option('--full-code/--preview-code', help='Print the whole exported live demo script instead of the curated preview.')] = False, overwrite: Annotated[bool, typer.Option('--overwrite/--no-overwrite', help='Allow replacing previously exported live demo files.')] = True, stage_mode: Annotated[bool, typer.Option('--stage/--technical', help='Default keeps the live demo audience-safe; --technical shows more raw execution detail.')] = True, dry_run: Annotated[bool, typer.Option('--dry-run', help='Show the plan without running the demo')] = False)`
- `start(goal: Annotated[str, typer.Option('--goal', help='What you want to do: demo, project, study, edge, or data. Aliases like pi, research, and quickstart also work.')] = 'demo', run_now: Annotated[bool, typer.Option('--run', help='Run the safe starter action for demo, project, or edge instead of only printing the plan.')] = False, output_dir: Annotated[Path, typer.Option(help='Output directory for --goal demo.')] = Path('results/demo'), project_name: Annotated[str, typer.Option(help='Project folder name for --goal project.')] = 'iints_quickstart', edge_output_dir: Annotated[Path, typer.Option(help='Edge project folder for --goal edge.')] = Path('iints_pi_demo'), board: Annotated[str, typer.Option(help='Edge board for --goal edge: raspberry_pi or uno_q.')] = 'raspberry_pi') -> None`
- `onboard(output_dir: Annotated[Path, typer.Option(help='Root directory for the canonical onboarding outputs.')] = Path('results/onboarding'), run_safe_steps: Annotated[bool, typer.Option('--run-safe-steps', help='Run doctor, demo, import-demo, and realism-check.')] = False) -> None`
- `guide()`
- `new_algo(name: Annotated[str, typer.Option(help='Name of the new algorithm')], author: Annotated[str, typer.Option(help='Author of the algorithm')], output_dir: Annotated[Path, typer.Option(help='Directory to save the new algorithm file')] = Path('.'))`
- `presets_list()`
- `presets_show(name: Annotated[str, typer.Option(help='Preset name (e.g., baseline_t1d)')])`
- `presets_run(name: Annotated[str, typer.Option(help='Preset name (e.g., baseline_t1d)')], algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt) for dual-guard forecasting')] = None, output_dir: Annotated[Optional[Path], typer.Option(help='Directory to save outputs')] = None, compare_baselines: Annotated[bool, typer.Option(help='Run PID and standard pump baselines in the background')] = True, seed: Annotated[Optional[int], typer.Option(help='Random seed for deterministic runs')] = None, patient_model_type: Annotated[str, typer.Option('--patient-model', help='Patient model: auto, bergman, custom, simglucose')] = 'auto', sensor_profile: Annotated[Optional[str], typer.Option(help=f"Sensor artifact profile: {', '.join(sorted(SENSOR_PROFILES))}")] = None, sensor_noise_std: Annotated[Optional[float], typer.Option('--sensor-noise-std', help='CGM noise std (mg/dL)')] = None, sensor_lag_minutes: Annotated[Optional[int], typer.Option('--sensor-lag-minutes', help='CGM lag (minutes)')] = None, sensor_dropout_prob: Annotated[Optional[float], typer.Option('--sensor-dropout-prob', help='CGM dropout probability (0-1)')] = None, sensor_bias: Annotated[Optional[float], typer.Option('--sensor-bias', help='CGM bias (mg/dL)')] = None, safety_min_glucose: Annotated[Optional[float], typer.Option('--safety-min-glucose', help='Min plausible glucose (mg/dL)')] = None, safety_max_glucose: Annotated[Optional[float], typer.Option('--safety-max-glucose', help='Max plausible glucose (mg/dL)')] = None, safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option('--safety-max-glucose-delta-per-5-min', help='Max glucose delta per 5 min (mg/dL)')] = None, safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hypo-threshold', help='Hypoglycemia threshold (mg/dL)')] = None, safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-severe-hypo-threshold', help='Severe hypoglycemia threshold (mg/dL)')] = None, safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hyper-threshold', help='Hyperglycemia threshold (mg/dL)')] = None, safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option('--safety-max-bolus', help='Max insulin per bolus (U)')] = None, safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option('--safety-glucose-rate-alarm', help='Glucose rate alarm (mg/dL/min)')] = None, safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option('--safety-max-insulin-per-hour', help='Max insulin per 60 min (U)')] = None, safety_max_iob: Annotated[Optional[float], typer.Option('--safety-max-iob', help='Max insulin on board (U)')] = None, safety_trend_stop: Annotated[Optional[float], typer.Option('--safety-trend-stop', help='Negative trend cutoff (mg/dL/min)')] = None, safety_hypo_cutoff: Annotated[Optional[float], typer.Option('--safety-hypo-cutoff', help='Hard hypo cutoff (mg/dL)')] = None, safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option('--safety-critical-glucose', help='Critical glucose threshold (mg/dL)')] = None, safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option('--safety-critical-duration', help='Critical glucose duration (minutes)')] = None, safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-uncertainty-gate', help='Enable predictor uncertainty gate')] = None, safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option('--safety-predictor-max-std', help='Max allowed predictor std (mg/dL) before fallback')] = None, safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-ood-gate', help='Enable predictor OOD gate')] = None, safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option('--safety-predictor-ood-z', help='Predictor OOD z-score threshold')] = None, safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option('--safety-predictor-ood-feature-fraction', help='Max fraction of features allowed OOD')] = None)`
- `presets_create(name: Annotated[str, typer.Option(help='Preset name (snake_case)')], output_dir: Annotated[Path, typer.Option(help='Output directory for preset files')] = Path('./presets'), initial_glucose: Annotated[float, typer.Option(help='Initial glucose (mg/dL)')] = 140.0, basal_insulin_rate: Annotated[float, typer.Option(help='Basal insulin rate (U/hr)')] = 0.5, insulin_sensitivity: Annotated[float, typer.Option(help='Insulin sensitivity (mg/dL per U)')] = 50.0, carb_factor: Annotated[float, typer.Option(help='Carb factor (g per U)')] = 10.0)`
- `run_wizard()`
- `profiles_presets()`
- `profiles_create(name: Annotated[str, typer.Option(help='Profile name (file stem)')], output_dir: Annotated[Path, typer.Option(help='Output directory for the profile YAML')] = Path('./patient_profiles'), preset: Annotated[Optional[str], typer.Option(help='Optional starter preset: stable-demo, stress-test, or endurance.')] = None, isf: Annotated[float, typer.Option(help='Insulin Sensitivity Factor (mg/dL per unit)')] = 50.0, icr: Annotated[float, typer.Option(help='Insulin-to-carb ratio (grams per unit)')] = 10.0, basal_rate: Annotated[float, typer.Option(help='Basal insulin rate (U/hr)')] = 0.8, initial_glucose: Annotated[float, typer.Option(help='Initial glucose (mg/dL)')] = 120.0, dawn_strength: Annotated[float, typer.Option(help='Dawn phenomenon strength (mg/dL per hour)')] = 0.0, dawn_start: Annotated[float, typer.Option(help='Dawn phenomenon start hour (0-23)')] = 4.0, dawn_end: Annotated[float, typer.Option(help='Dawn phenomenon end hour (0-24)')] = 8.0)`
- `scenarios_generate(name: Annotated[str, typer.Option(help='Scenario name')] = 'Generated Scenario', output_path: Annotated[Path, typer.Option(help='Output JSON path')] = Path('./scenarios/generated_scenario.json'), duration_minutes: Annotated[int, typer.Option(help='Scenario duration in minutes')] = 1440, seed: Annotated[Optional[int], typer.Option(help='Random seed')] = None, meal_count: Annotated[int, typer.Option(help='Number of meal events')] = 3, meal_min_grams: Annotated[float, typer.Option(help='Min meal size (g carbs)')] = 30.0, meal_max_grams: Annotated[float, typer.Option(help='Max meal size (g carbs)')] = 80.0, exercise_count: Annotated[int, typer.Option(help='Number of exercise events')] = 0, sensor_error_count: Annotated[int, typer.Option(help='Number of sensor error events')] = 0)`
- `scenarios_wizard()`
- `scenarios_migrate(input_path: Annotated[Path, typer.Argument(help='Scenario JSON to migrate')], output_path: Annotated[Optional[Path], typer.Option(help='Output path (default: overwrite input)')] = None)`
- `scenarios_export_study_pack(output_dir: Annotated[Path, typer.Option(help='Directory to write the official study pack')] = Path('scenarios/study_pack'), seeds: Annotated[str, typer.Option(help='Comma-separated seed list to include in the pack manifest')] = '1,2,3,4,5,6,7,8,9,10', preset: Annotated[str, typer.Option(help='Study-pack preset: official or eucys')] = 'official')`
- `run(algo: Annotated[Optional[Path], typer.Option(help='Path to the algorithm Python file. Leave blank to use the built-in Clinical Baseline.')] = None, preset: Annotated[Optional[str], typer.Option(help='Optional built-in preset such as baseline_t1d. This fills the patient profile and scenario unless you override them.')] = None, predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt) for dual-guard forecasting')] = None, patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration (for example 'default_patient' or 'clinic_safe_baseline')")] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option(help='Path to a patient config YAML (overrides --patient-config-name)')] = None, scenario_path: Annotated[Optional[Path], typer.Option('--scenario', '--scenario-path', help='Path to the scenario JSON file (for example scenarios/example_scenario.json)')] = None, duration: Annotated[int, typer.Option(help='Simulation duration in minutes')] = 720, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, output_dir: Annotated[Optional[Path], typer.Option(help='Directory to save simulation results')] = None, compare_baselines: Annotated[bool, typer.Option(help='Run built-in baselines in the background for comparison')] = True, seed: Annotated[Optional[int], typer.Option(help='Random seed for deterministic runs')] = None, patient_model_type: Annotated[str, typer.Option('--patient-model', help='Patient model: auto, bergman, custom, simglucose')] = 'auto', sensor_profile: Annotated[Optional[str], typer.Option(help=f"Sensor artifact profile: {', '.join(sorted(SENSOR_PROFILES))}")] = None, sensor_noise_std: Annotated[Optional[float], typer.Option('--sensor-noise-std', help='CGM noise std (mg/dL)')] = None, sensor_lag_minutes: Annotated[Optional[int], typer.Option('--sensor-lag-minutes', help='CGM lag (minutes)')] = None, sensor_dropout_prob: Annotated[Optional[float], typer.Option('--sensor-dropout-prob', help='CGM dropout probability (0-1)')] = None, sensor_bias: Annotated[Optional[float], typer.Option('--sensor-bias', help='CGM bias (mg/dL)')] = None, safety_min_glucose: Annotated[Optional[float], typer.Option('--safety-min-glucose', help='Min plausible glucose (mg/dL)')] = None, safety_max_glucose: Annotated[Optional[float], typer.Option('--safety-max-glucose', help='Max plausible glucose (mg/dL)')] = None, safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option('--safety-max-glucose-delta-per-5-min', help='Max glucose delta per 5 min (mg/dL)')] = None, safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hypo-threshold', help='Hypoglycemia threshold (mg/dL)')] = None, safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-severe-hypo-threshold', help='Severe hypoglycemia threshold (mg/dL)')] = None, safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hyper-threshold', help='Hyperglycemia threshold (mg/dL)')] = None, safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option('--safety-max-bolus', help='Max insulin per bolus (U)')] = None, safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option('--safety-glucose-rate-alarm', help='Glucose rate alarm (mg/dL/min)')] = None, safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option('--safety-max-insulin-per-hour', help='Max insulin per 60 min (U)')] = None, safety_max_iob: Annotated[Optional[float], typer.Option('--safety-max-iob', help='Max insulin on board (U)')] = None, safety_trend_stop: Annotated[Optional[float], typer.Option('--safety-trend-stop', help='Negative trend cutoff (mg/dL/min)')] = None, safety_hypo_cutoff: Annotated[Optional[float], typer.Option('--safety-hypo-cutoff', help='Hard hypo cutoff (mg/dL)')] = None, safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option('--safety-critical-glucose', help='Critical glucose threshold (mg/dL)')] = None, safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option('--safety-critical-duration', help='Critical glucose duration (minutes)')] = None, safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-uncertainty-gate', help='Enable predictor uncertainty gate')] = None, safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option('--safety-predictor-max-std', help='Max allowed predictor std (mg/dL) before fallback')] = None, safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-ood-gate', help='Enable predictor OOD gate')] = None, safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option('--safety-predictor-ood-z', help='Predictor OOD z-score threshold')] = None, safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option('--safety-predictor-ood-feature-fraction', help='Max fraction of features allowed OOD')] = None, wizard: Annotated[bool, typer.Option('--wizard', help='Ask a few questions and build the run interactively')] = False, dry_run: Annotated[bool, typer.Option('--dry-run', help='Validate the setup and print the run plan without executing the simulation')] = False)`
- `run_full(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt) for dual-guard forecasting')] = None, patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration (e.g., 'default_patient')")] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option(help='Path to a patient config YAML (overrides --patient-config-name)')] = None, scenario_path: Annotated[Optional[Path], typer.Option(help='Path to the scenario JSON file')] = None, duration: Annotated[int, typer.Option(help='Simulation duration in minutes')] = 720, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, output_dir: Annotated[Optional[Path], typer.Option(help='Directory to save results + audit + report')] = None, seed: Annotated[Optional[int], typer.Option(help='Random seed for deterministic runs')] = None, safety_min_glucose: Annotated[Optional[float], typer.Option('--safety-min-glucose', help='Min plausible glucose (mg/dL)')] = None, safety_max_glucose: Annotated[Optional[float], typer.Option('--safety-max-glucose', help='Max plausible glucose (mg/dL)')] = None, safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option('--safety-max-glucose-delta-per-5-min', help='Max glucose delta per 5 min (mg/dL)')] = None, safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hypo-threshold', help='Hypoglycemia threshold (mg/dL)')] = None, safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-severe-hypo-threshold', help='Severe hypoglycemia threshold (mg/dL)')] = None, safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hyper-threshold', help='Hyperglycemia threshold (mg/dL)')] = None, safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option('--safety-max-bolus', help='Max insulin per bolus (U)')] = None, safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option('--safety-glucose-rate-alarm', help='Glucose rate alarm (mg/dL/min)')] = None, safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option('--safety-max-insulin-per-hour', help='Max insulin per 60 min (U)')] = None, safety_max_iob: Annotated[Optional[float], typer.Option('--safety-max-iob', help='Max insulin on board (U)')] = None, safety_trend_stop: Annotated[Optional[float], typer.Option('--safety-trend-stop', help='Negative trend cutoff (mg/dL/min)')] = None, safety_hypo_cutoff: Annotated[Optional[float], typer.Option('--safety-hypo-cutoff', help='Hard hypo cutoff (mg/dL)')] = None, safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option('--safety-critical-glucose', help='Critical glucose threshold (mg/dL)')] = None, safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option('--safety-critical-duration', help='Critical glucose duration (minutes)')] = None, safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-uncertainty-gate', help='Enable predictor uncertainty gate')] = None, safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option('--safety-predictor-max-std', help='Max allowed predictor std (mg/dL) before fallback')] = None, safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-ood-gate', help='Enable predictor OOD gate')] = None, safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option('--safety-predictor-ood-z', help='Predictor OOD z-score threshold')] = None, safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option('--safety-predictor-ood-feature-fraction', help='Max fraction of features allowed OOD')] = None)`
- `run_parallel(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt) for dual-guard forecasting')] = None, scenarios_dir: Annotated[Optional[Path], typer.Option(help='Directory with scenario JSON files')] = None, scenario_paths: Annotated[List[Path], typer.Option('--scenario-path', help='Scenario JSON path (repeatable)')] = [], patient_config_name: Annotated[str, typer.Option(help='Patient config name')] = 'default_patient', patient_config_path: Annotated[Optional[Path], typer.Option(help='Patient config YAML path')] = None, patient_configs_dir: Annotated[Optional[Path], typer.Option(help='Directory of patient YAML configs')] = None, duration: Annotated[int, typer.Option(help='Simulation duration in minutes')] = 720, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, output_dir: Annotated[Path, typer.Option(help='Root directory for batch outputs')] = Path('./results/batch'), max_workers: Annotated[Optional[int], typer.Option(help='Max parallel workers')] = None, seed: Annotated[Optional[int], typer.Option(help='Base seed for deterministic runs')] = None, compare_baselines: Annotated[bool, typer.Option(help='Run PID + standard pump baselines')] = False, export_audit: Annotated[bool, typer.Option(help='Export audit trails')] = False, generate_report: Annotated[bool, typer.Option(help='Generate PDF reports')] = False, safety_min_glucose: Annotated[Optional[float], typer.Option('--safety-min-glucose')] = None, safety_max_glucose: Annotated[Optional[float], typer.Option('--safety-max-glucose')] = None, safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option('--safety-max-glucose-delta-per-5-min')] = None, safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hypo-threshold')] = None, safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-severe-hypo-threshold')] = None, safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option('--safety-hyper-threshold')] = None, safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option('--safety-max-bolus')] = None, safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option('--safety-glucose-rate-alarm')] = None, safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option('--safety-max-insulin-per-hour')] = None, safety_max_iob: Annotated[Optional[float], typer.Option('--safety-max-iob')] = None, safety_trend_stop: Annotated[Optional[float], typer.Option('--safety-trend-stop')] = None, safety_hypo_cutoff: Annotated[Optional[float], typer.Option('--safety-hypo-cutoff')] = None, safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option('--safety-critical-glucose')] = None, safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option('--safety-critical-duration')] = None, safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-uncertainty-gate')] = None, safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option('--safety-predictor-max-std')] = None, safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option('--safety-predictor-ood-gate')] = None, safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option('--safety-predictor-ood-z')] = None, safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option('--safety-predictor-ood-feature-fraction')] = None)`
- `scorecard(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file')], profile: Annotated[str, typer.Option(help='Validation profile id')] = 'research_default', predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional predictor checkpoint (.pt)')] = None, presets_csv: Annotated[Optional[str], typer.Option(help='Comma-separated preset names (default: all bundled presets)')] = None, output_dir: Annotated[Path, typer.Option(help='Output directory for scorecard artifacts')] = Path('results/scorecard'), seed: Annotated[Optional[int], typer.Option(help='Base random seed')] = 42)`
- `poster(run_dir: Annotated[List[Path], typer.Option('--run-dir', help='Run bundle directory containing results.csv. Repeat up to three times.')] = [], label: Annotated[List[str], typer.Option('--label', help='Optional poster label aligned to each --run-dir (for example: Normal Run, Meal Stress Test, Supervisor Override).')] = [], output_path: Annotated[Path, typer.Option(help='PNG output path for the poster graphic.')] = Path('./results/posters/iints_results_poster.png'), summary_output_path: Annotated[Optional[Path], typer.Option(help='Optional JSON sidecar summary path.')] = None, title: Annotated[str, typer.Option(help='Main poster headline.')] = '288 Decisions. Every Day. We Test Them All.', subtitle: Annotated[str, typer.Option(help='Supporting poster subtitle.')] = 'Three IINTS-AF scenarios showing control, stress handling, and supervisor protection.', results_root: Annotated[Path, typer.Option(help='Root folder used when no --run-dir values are supplied.')] = Path('./results'))`
- `demo_booth(output_dir: Annotated[Path, typer.Option(help='Directory where the fair-ready demo bundle should be written.')] = Path('./results/booth_demo'), duration: Annotated[int, typer.Option(help='Simulation duration in minutes for each booth scenario.')] = 360, time_step: Annotated[int, typer.Option(help='Simulation step size in minutes.')] = 5, seed: Annotated[int, typer.Option(help='Deterministic random seed.')] = 42, prepare_ai: Annotated[bool, typer.Option('--prepare-ai/--no-prepare-ai', help='Prepare AI-ready artifacts for the Supervisor Override run.')] = True) -> None`
- `demo_export(output_dir: Annotated[Path, typer.Option(help='Directory where the bundled live stage demo files should be written.')] = Path('./iints_demo'), overwrite: Annotated[bool, typer.Option('--overwrite/--no-overwrite', help='Allow overwriting exported demo files.')] = False) -> None`
- `demo_live(output_dir: Annotated[Path, typer.Option(help='Root directory for the exported code and generated live-demo results.')] = Path('./results/live_demo'), run_demo: Annotated[bool, typer.Option('--run/--no-run', help='Run the exported demo after showing the code preview.')] = True, prepare_ai: Annotated[bool, typer.Option('--prepare-ai/--skip-ai', help='Prepare optional local-AI artifacts during the demo run.')] = False, full_code: Annotated[bool, typer.Option('--full-code/--preview-code', help='Print the whole exported script instead of the curated preview.')] = False, overwrite: Annotated[bool, typer.Option('--overwrite/--no-overwrite', help='Allow replacing previously exported demo code files.')] = True, audience: Annotated[str, typer.Option('--audience', help='Presenter framing: mixed, clinical, engineering, or jury. Aliases like doctor and engineer also work.')] = 'mixed', stage_mode: Annotated[bool, typer.Option('--stage/--technical', help='Stage mode hides noisy subprocess logs and prints a clean audience-facing flow.')] = True, build_evidence: Annotated[bool, typer.Option('--evidence/--no-evidence', help='Build a public research evidence bundle after the live run.')] = True, story_mode: Annotated[str, typer.Option('--story', help='Demo story layer: sdk, doctor, eucys, or booth. Positional shortcuts also work via `iints demo doctor`.')] = 'sdk') -> None`
- `report(results_csv: Annotated[Path, typer.Option(help='Path to a simulation results CSV')], output_path: Annotated[Path, typer.Option(help='Output PDF path')] = Path('./results/clinical_report.pdf'), safety_report_path: Annotated[Optional[Path], typer.Option(help='Optional safety report JSON path')] = None, audit_output_dir: Annotated[Optional[Path], typer.Option(help='Optional audit output directory')] = None, bundle_dir: Annotated[Optional[Path], typer.Option(help='If set, write PDF + plots + audit into this folder')] = None, style: Annotated[str, typer.Option(help='Report style: standard or agp')] = 'standard', subject_name: Annotated[str, typer.Option(help='Subject/run label shown on AGP-style reports')] = 'Research simulation', summary_json_path: Annotated[Optional[Path], typer.Option(help='Optional AGP summary JSON output path')] = None, agp_png: Annotated[bool, typer.Option('--png/--no-png', help='For AGP reports, export agp_profile.png and daily_profiles.png.')] = False)`
- `safety_visualize(results_csv: Annotated[Path, typer.Option(help='Path to a simulation results CSV')], output_html: Annotated[Path, typer.Option(help='Output standalone HTML visualizer path')] = Path('./results/safety_visualizer.html'), output_json: Annotated[Optional[Path], typer.Option(help='Optional machine-readable JSON summary path')] = None, safety_report_path: Annotated[Optional[Path], typer.Option(help='Optional safety report JSON path')] = None, title: Annotated[str, typer.Option(help='HTML page title')] = 'IINTS Safety Contract Visualizer') -> None`
- `validate(scenario_path: Annotated[Path, typer.Option(help='Path to a scenario JSON file')], patient_config_path: Annotated[Optional[Path], typer.Option(help='Optional patient config YAML to validate')] = None)`
- `data_list()`
- `data_info(dataset_id: Annotated[str, typer.Argument(help='Dataset id (see `iints data list`)')])`
- `data_cite(dataset_id: Annotated[str, typer.Argument(help='Dataset id (see `iints data list`)')])`
- `data_fetch(dataset_id: Annotated[str, typer.Argument(help='Dataset id (see `iints data list`)')], output_dir: Annotated[Optional[Path], typer.Option(help='Output directory (default: data_packs/official/<id>)')] = None, extract: Annotated[bool, typer.Option(help='Extract zip files if present')] = True, verify: Annotated[bool, typer.Option(help='Verify pinned SHA-256 hashes and emit SHA256SUMS.txt. Public sources without published hashes now require --no-verify.')] = True)`
- `data_research_plan(output_dir: Annotated[Path, typer.Option(help='Output folder for the dataset acquisition plan and source matrix.')] = Path('data_packs/research_dataset_plan'), dataset: Annotated[List[str], typer.Option('--dataset', help='Optional dataset id to include. Repeat for a custom subset; default includes the full curated IINTS research list.')] = []) -> None`
- `data_contract_template(output_path: Annotated[Path, typer.Option(help='Where to write the starter contract YAML')] = Path('data_contract.yaml'))`
- `data_certify_template(output_path: Annotated[Path, typer.Option(help='Where to write the starter certification contract YAML')] = Path('data_contract.yaml'))`
- `data_contract_run(contract_path: Annotated[Path, typer.Argument(help='Path to contract YAML')], input_csv: Annotated[Path, typer.Argument(help='Path to input CSV')], output_json: Annotated[Optional[Path], typer.Option(help='Optional output report JSON path')] = None, apply_builtin_transforms: Annotated[bool, typer.Option(help='Apply built-in unit conversion transforms from the contract')] = True, fail_on_noncompliant: Annotated[bool, typer.Option(help='Exit code 1 when compliance checks fail')] = False, min_mdmp_grade: Annotated[Optional[str], typer.Option(help='Optional MDMP grade gate (draft, research_grade, clinical_grade)')] = None)`
- `data_certify(contract_path: Annotated[Path, typer.Argument(help='Path to certification contract YAML')], input_csv: Annotated[Path, typer.Argument(help='Path to input CSV')], output_json: Annotated[Optional[Path], typer.Option(help='Optional output report JSON path')] = None, apply_builtin_transforms: Annotated[bool, typer.Option(help='Apply built-in unit conversion transforms from the contract')] = True, fail_on_noncompliant: Annotated[bool, typer.Option(help='Exit code 1 when compliance checks fail')] = False, min_mdmp_grade: Annotated[Optional[str], typer.Option(help='Optional certification grade gate (draft, research_grade, clinical_grade, ai_ready)')] = None)`
- `data_eu_ai_pact_review(report_json: Annotated[Path, typer.Argument(help='Path to an MDMP/data certification JSON report')], output_json: Annotated[Optional[Path], typer.Option(help='Optional output governance review JSON path')] = None, strict: Annotated[bool, typer.Option('--strict/--core-only', help='Strict checks include high-risk readiness controls; core-only checks the three AI Pact core themes.')] = True, fail_on_blocked: Annotated[bool, typer.Option(help='Exit code 1 when the governance review is blocked')] = False)`
- `data_realism_check(input_csv: Annotated[Path, typer.Argument(help='Path to CSV in generic/dexcom/libre/carelink format')], output_json: Annotated[Optional[Path], typer.Option(help='Optional output report JSON path')] = None, output_html: Annotated[Optional[Path], typer.Option(help='Optional output dashboard HTML path')] = None, data_format: Annotated[str, typer.Option(help='Data format preset: generic, dexcom, libre, carelink')] = 'generic', time_unit: Annotated[str, typer.Option(help='Timestamp unit for numeric timestamps: minutes or seconds')] = 'minutes', expected_interval_minutes: Annotated[int, typer.Option(help='Expected reading interval in minutes')] = 5, min_meal_grams: Annotated[float, typer.Option(help='Minimum carbs (g) that count as a meal event')] = 10.0, reference: Annotated[Optional[str], typer.Option(help='Optional realism reference profile or dataset id (for example free_living_t1d, azt1d, hupa_ucm).')] = None, min_realism_verdict: Annotated[Optional[str], typer.Option(help='Optional gate: likely_realistic or needs_review. Exit code 1 if the report is worse.')] = None, strict_real_data_gate: Annotated[bool, typer.Option('--strict-real-data-gate/--no-strict-real-data-gate', help='Apply the stricter evidence-readiness gate for real-data/local-AI use.')] = False, mapping: Annotated[List[str], typer.Option('--map', help='Column mapping key=value (e.g., timestamp=Time, glucose=SGV)')] = [])`
- `data_mdmp_visualizer(report_json: Annotated[Path, typer.Argument(help='Path to contract-run JSON report')], output_html: Annotated[Path, typer.Option(help='Output HTML path')] = Path('results/mdmp_dashboard.html'), title: Annotated[str, typer.Option(help='Dashboard title')] = 'IINTS MDMP Certification Dashboard')`
- `data_certify_visualizer(report_json: Annotated[Path, typer.Argument(help='Path to certification report JSON')], output_html: Annotated[Path, typer.Option(help='Output HTML path')] = Path('results/mdmp_dashboard.html'), title: Annotated[str, typer.Option(help='Dashboard title')] = 'IINTS Data Certification Dashboard')`
- `data_corrupt_for_study(input_csv: Annotated[Path, typer.Argument(help='Source CSV that will be deliberately corrupted for ablation studies')], output_csv: Annotated[Path, typer.Option(help='Output CSV path for the corrupted dataset')] = Path('results/corrupted_study.csv'), mode: Annotated[List[str], typer.Option('--mode', help='Corruption mode to apply. Repeat --mode for multiple operators.')] = ['timestamp_shift'], manifest_output: Annotated[Optional[Path], typer.Option(help='Optional corruption manifest JSON path')] = None, seed: Annotated[int, typer.Option(help='Random seed for deterministic corruption')] = 42, timestamp_shift_minutes: Annotated[int, typer.Option(help='Minutes to shift timestamps for the timestamp_shift mode')] = 60, missing_fraction: Annotated[float, typer.Option(help='Fraction of rows to remove for missing_block')] = 0.1, duplicate_fraction: Annotated[float, typer.Option(help='Fraction of rows to duplicate for duplicate_rows')] = 0.05, spike_fraction: Annotated[float, typer.Option(help='Fraction of glucose rows to spike for glucose_spikes')] = 0.03, spike_magnitude_mgdl: Annotated[float, typer.Option(help='Spike magnitude in mg/dL for glucose_spikes')] = 60.0) -> None`
- `data_synthetic_mirror(input_csv: Annotated[Path, typer.Argument(help='Source CSV (validated real dataset)')], contract_path: Annotated[Path, typer.Argument(help='Contract YAML path used as schema/range guard')], output_csv: Annotated[Path, typer.Option(help='Output synthetic CSV path')] = Path('data/synthetic_mirror.csv'), output_json: Annotated[Optional[Path], typer.Option(help='Optional synthetic mirror report JSON')] = Path('results/synthetic_mirror_report.json'), rows: Annotated[Optional[int], typer.Option(help='Optional number of rows to generate')] = None, seed: Annotated[int, typer.Option(help='Random seed for deterministic synthesis')] = 42, noise_scale: Annotated[float, typer.Option(help='Numeric perturbation scale as fraction of source std-dev')] = 0.05, min_mdmp_grade: Annotated[Optional[str], typer.Option(help='Optional MDMP grade gate for generated synthetic dataset')] = 'research_grade', fail_on_noncompliant: Annotated[bool, typer.Option(help='Exit code 1 when generated dataset fails compliance')] = True)`
- `mdmp_template(output_path: Annotated[Path, typer.Option(help='Where to write the MDMP contract YAML')] = Path('mdmp_contract.yaml'))`
- `mdmp_validate(contract_path: Annotated[Path, typer.Argument(help='Path to MDMP contract YAML')], input_csv: Annotated[Path, typer.Argument(help='Path to input CSV')], output_json: Annotated[Optional[Path], typer.Option(help='Optional output report JSON path')] = None, apply_builtin_transforms: Annotated[bool, typer.Option(help='Apply built-in unit conversion transforms (default: off for explicit MDMP operation)')] = False, fail_on_noncompliant: Annotated[bool, typer.Option(help='Exit code 1 when compliance checks fail')] = False, min_mdmp_grade: Annotated[Optional[str], typer.Option(help='Optional MDMP grade gate (draft, research_grade, clinical_grade)')] = None)`
- `mdmp_visualizer(report_json: Annotated[Path, typer.Argument(help='Path to MDMP validation report JSON')], output_html: Annotated[Path, typer.Option(help='Output HTML path')] = Path('results/mdmp_dashboard.html'), title: Annotated[str, typer.Option(help='Dashboard title')] = 'IINTS MDMP Certification Dashboard')`
- `mdmp_synthetic_mirror(input_csv: Annotated[Path, typer.Argument(help='Source CSV')], contract_path: Annotated[Path, typer.Argument(help='MDMP contract YAML')], output_csv: Annotated[Path, typer.Option(help='Output synthetic CSV path')] = Path('data/synthetic_mirror.csv'), output_json: Annotated[Optional[Path], typer.Option(help='Optional synthetic mirror report JSON')] = Path('results/synthetic_mirror_report.json'), rows: Annotated[Optional[int], typer.Option(help='Optional number of rows to generate')] = None, seed: Annotated[int, typer.Option(help='Random seed')] = 42, noise_scale: Annotated[float, typer.Option(help='Numeric perturbation scale')] = 0.05, min_mdmp_grade: Annotated[Optional[str], typer.Option(help='Optional MDMP grade gate')] = 'research_grade', fail_on_noncompliant: Annotated[bool, typer.Option(help='Exit code 1 when generated dataset fails compliance')] = True)`
- `sources(category: Annotated[Optional[str], typer.Option(help='Filter by source category (guideline, trial, model, dataset, ...).')] = None, output_json: Annotated[Optional[Path], typer.Option(help='Optional JSON output path.')] = None)`
- `research_prepare_azt1d(input_dir: Annotated[Path, typer.Option(help='Root directory containing AZT1D Subject folders')] = Path('data_packs/public/azt1d/AZT1D 2025/CGM Records'), output: Annotated[Path, typer.Option(help='Output dataset path (CSV or Parquet)')] = Path('data_packs/public/azt1d/processed/azt1d_merged.csv'), report: Annotated[Path, typer.Option(help='Quality report output path')] = Path('data_packs/public/azt1d/quality_report.json'), time_step: Annotated[int, typer.Option(help='Expected CGM sample interval (minutes)')] = 5, max_gap_multiplier: Annotated[float, typer.Option(help='Segment-break gap multiplier')] = 2.5, dia_minutes: Annotated[float, typer.Option(help='Insulin action duration (minutes)')] = 240.0, peak_minutes: Annotated[float, typer.Option(help='IOB peak time (minutes, OpenAPS bilinear)')] = 75.0, carb_absorb_minutes: Annotated[float, typer.Option(help='Carb absorption duration (minutes)')] = 120.0, max_basal: Annotated[float, typer.Option(help='Clip basal values above this (U/hr)')] = 20.0, max_bolus: Annotated[float, typer.Option(help='Clip bolus values above this (U)')] = 30.0, max_carbs: Annotated[float, typer.Option(help='Clip carb grams above this')] = 200.0, basal_is_rate: Annotated[bool, typer.Option(help='Treat Basal column as U/hr (convert to U/step)')] = True)`
- `research_prepare_ohio(input_dir: Annotated[Path, typer.Option(help='Root directory containing OhioT1DM patient_* folders')] = Path('data_packs/public/ohio_t1dm'), output: Annotated[Path, typer.Option(help='Output dataset path (CSV or Parquet)')] = Path('data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv'), report: Annotated[Path, typer.Option(help='Quality report output path')] = Path('data_packs/public/ohio_t1dm/quality_report.json'), time_step: Annotated[int, typer.Option(help='Expected CGM sample interval (minutes)')] = 5, max_gap_multiplier: Annotated[float, typer.Option(help='Segment-break gap multiplier')] = 2.5, dia_minutes: Annotated[float, typer.Option(help='Insulin action duration (minutes)')] = 240.0, peak_minutes: Annotated[float, typer.Option(help='IOB peak time (minutes, OpenAPS bilinear)')] = 75.0, carb_absorb_minutes: Annotated[float, typer.Option(help='Carb absorption duration (minutes)')] = 120.0, max_insulin: Annotated[float, typer.Option(help='Clip insulin units above this')] = 30.0, max_carbs: Annotated[float, typer.Option(help='Clip carb grams above this')] = 200.0, icr_default: Annotated[float, typer.Option(help='Fallback ICR (g/U)')] = 10.0, isf_default: Annotated[float, typer.Option(help='Fallback ISF (mg/dL per U)')] = 50.0, basal_default: Annotated[float, typer.Option(help='Fallback basal rate (U/hr)')] = 0.0, meal_window_min: Annotated[float, typer.Option(help='Meal→insulin matching window (minutes)')] = 30.0, isf_window_min: Annotated[float, typer.Option(help='ISF estimation window (minutes)')] = 60.0, min_meal_carbs: Annotated[float, typer.Option(help='Minimum carbs to consider a meal (g)')] = 5.0, min_bolus: Annotated[float, typer.Option(help='Minimum insulin to consider a bolus (U)')] = 0.1)`
- `research_prepare_hupa(input_dir: Annotated[Path, typer.Option(help='Root directory containing HUPA-UCM CSV files')] = Path('data_packs/public/hupa_ucm'), output: Annotated[Path, typer.Option(help='Output dataset path (CSV or Parquet)')] = Path('data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv'), report: Annotated[Path, typer.Option(help='Quality report output path')] = Path('data_packs/public/hupa_ucm/quality_report.json'), time_step: Annotated[int, typer.Option(help='Expected CGM sample interval (minutes)')] = 5, max_gap_multiplier: Annotated[float, typer.Option(help='Segment-break gap multiplier')] = 2.5, dia_minutes: Annotated[float, typer.Option(help='Insulin action duration (minutes)')] = 240.0, peak_minutes: Annotated[float, typer.Option(help='IOB peak time (minutes, OpenAPS bilinear)')] = 75.0, carb_absorb_minutes: Annotated[float, typer.Option(help='Carb absorption duration (minutes)')] = 120.0, max_insulin: Annotated[float, typer.Option(help='Clip insulin units above this')] = 30.0, max_carbs: Annotated[float, typer.Option(help='Clip carb grams above this')] = 200.0, carb_serving_grams: Annotated[float, typer.Option(help='Carb serving size (g) for carb_input')] = 10.0, basal_is_rate: Annotated[bool, typer.Option(help='Treat basal_rate as U/hr (convert to U/step)')] = False, icr_default: Annotated[float, typer.Option(help='Fallback ICR (g/U)')] = 10.0, isf_default: Annotated[float, typer.Option(help='Fallback ISF (mg/dL per U)')] = 50.0, basal_default: Annotated[float, typer.Option(help='Fallback basal rate (U/hr)')] = 0.0, meal_window_min: Annotated[float, typer.Option(help='Meal→insulin matching window (minutes)')] = 30.0, isf_window_min: Annotated[float, typer.Option(help='ISF estimation window (minutes)')] = 60.0, min_meal_carbs: Annotated[float, typer.Option(help='Minimum carbs to consider a meal (g)')] = 5.0, min_bolus: Annotated[float, typer.Option(help='Minimum insulin to consider a bolus (U)')] = 0.1)`
- `research_quality(report: Annotated[Path, typer.Option(help='Path to quality_report.json produced by prepare-azt1d')] = Path('data_packs/public/azt1d/quality_report.json'))`
- `research_blend_datasets(source: Annotated[List[str], typer.Option('--source', help='Repeatable dataset source in label=path form, for example azt1d=data/azt1d.csv.')], output: Annotated[Path, typer.Option(help='Output blended predictor dataset path')] = Path('data_packs/processed/predictor_blend.csv'), manifest: Annotated[Path, typer.Option(help='Output blend manifest JSON path')] = Path('data_packs/processed/predictor_blend_manifest.json')) -> None`
- `research_build_control_dataset(run: Annotated[List[str], typer.Option('--run', help='Repeatable run input in label=path form; path may contain raw/steps.csv or results.csv.')], output: Annotated[Path, typer.Option(help='Output controller teacher dataset CSV')] = Path('data_packs/processed/controller_teacher_dataset.csv'), manifest: Annotated[Path, typer.Option(help='Output controller dataset manifest JSON')] = Path('data_packs/processed/controller_teacher_manifest.json')) -> None`
- `research_train_controller(data: Annotated[Path, typer.Option(help='Controller teacher dataset CSV')], output: Annotated[Path, typer.Option(help='Output local controller JSON')] = Path('models/controller_imitation.json'), metrics_output: Annotated[Path, typer.Option(help='Output training metrics JSON')] = Path('models/controller_imitation_metrics.json'), ridge_lambda: Annotated[float, typer.Option(help='Ridge regularization strength')] = 0.001) -> None`
- `research_train_neural_controller(data: Annotated[Path, typer.Option(help='Controller teacher dataset CSV')], output: Annotated[Path, typer.Option(help='Output neural controller checkpoint')] = Path('models/controller_neural.pt'), metrics_output: Annotated[Path, typer.Option(help='Output neural controller metrics JSON')] = Path('models/controller_neural_metrics.json'), epochs: Annotated[int, typer.Option(help='Training epochs')] = 120, hidden_size: Annotated[List[int], typer.Option('--hidden-size', help='Repeatable hidden layer size, for example --hidden-size 64.')] = []) -> None`
- `research_evaluate_controller(model: Annotated[Path, typer.Option(help='Controller model path')], model_kind: Annotated[str, typer.Option(help='Controller type: linear or neural')] = 'linear', output_dir: Annotated[Path, typer.Option(help='Output evaluation directory')] = Path('results/controller_evaluation'), preset: Annotated[List[str], typer.Option('--preset', help='Repeatable held-out preset name.')] = [], seed: Annotated[List[int], typer.Option('--seed', help='Repeatable evaluation seed.')] = [], duration_minutes: Annotated[int, typer.Option(help='Duration per closed-loop run in minutes')] = 1440) -> None`
- `research_local_ai_lab(run: Annotated[List[str], typer.Option('--run', help='Repeatable run input in label=path form; path may contain a Jetson endurance bundle.')], output_dir: Annotated[Path, typer.Option(help='Output directory for datasets, models, and reports')] = Path('results/local_ai_lab'), train_predictor: Annotated[bool, typer.Option('--train-predictor/--skip-predictor', help='Train the local glucose predictor from the generated predictor dataset.')] = True, train_neural: Annotated[bool, typer.Option('--train-neural/--skip-neural', help='Train the PyTorch controller in addition to the auditable linear controller.')] = True, evaluate: Annotated[bool, typer.Option('--evaluate/--skip-evaluation', help='Run held-out closed-loop evaluation after training.')] = True, predictor_config: Annotated[Optional[Path], typer.Option(help='Optional predictor config YAML. Defaults to research/configs/predictor.yaml.')] = None, duration_minutes: Annotated[int, typer.Option(help='Duration per held-out controller-evaluation run.')] = 1440)`
- `research_export_onnx(model: Annotated[Path, typer.Option(help='Predictor checkpoint (.pt)')] = Path('models/hupa_finetuned_v2/predictor.pt'), out: Annotated[Path, typer.Option(help='Output ONNX file path')] = Path('models/predictor.onnx'))`
- `research_audit_split(data: Annotated[Path, typer.Option(help='Prepared dataset path (CSV/Parquet)')], history_steps: Annotated[int, typer.Option(help='History window length')] = 48, horizon_steps: Annotated[int, typer.Option(help='Forecast horizon length')] = 6, feature_columns_csv: Annotated[str, typer.Option(help='Comma-separated feature columns')] = 'glucose_actual_mgdl,patient_iob_units,patient_cob_grams,effective_isf,effective_icr,effective_basal_rate_u_per_hr,glucose_trend_mgdl_min', target_column: Annotated[str, typer.Option(help='Target column')] = 'glucose_actual_mgdl', subject_column: Annotated[str, typer.Option(help='Subject ID column')] = 'subject_id', segment_column: Annotated[Optional[str], typer.Option(help='Segment column (optional)')] = 'segment_id', output_json: Annotated[Optional[Path], typer.Option(help='Write audit report JSON')] = None)`
- `research_evaluate_forecast(input_csv: Annotated[Path, typer.Option(help='CSV with observed/predicted columns')], observed_column: Annotated[str, typer.Option(help='Observed glucose column')] = 'glucose_actual_mgdl', predicted_column: Annotated[str, typer.Option(help='Predicted glucose column')] = 'predicted_glucose_ai_30min', predicted_std_column: Annotated[Optional[str], typer.Option(help='Optional prediction std column')] = 'predictor_uncertainty_std_mgdl', gate_profile: Annotated[Optional[str], typer.Option(help='Optional calibration gate profile id')] = None, gate_profiles_path: Annotated[Optional[Path], typer.Option(help='Optional calibration gate profiles YAML path')] = None, fail_on_gate: Annotated[bool, typer.Option(help='Exit code 1 when calibration gate fails')] = False, output_json: Annotated[Optional[Path], typer.Option(help='Write metrics JSON')] = None)`
- `research_parity_check(model: Annotated[Path, typer.Option(help='Predictor checkpoint (.pt)')], onnx: Annotated[Path, typer.Option(help='Exported ONNX model path')], samples: Annotated[int, typer.Option(help='Random sample count for parity check')] = 64, tolerance: Annotated[float, typer.Option(help='Maximum allowed absolute error')] = 0.001, seed: Annotated[int, typer.Option(help='Random seed')] = 42, output_json: Annotated[Optional[Path], typer.Option(help='Write parity report JSON')] = None)`
- `research_registry_list(registry: Annotated[Path, typer.Option(help='Path to model registry JSON')] = Path('models/registry.json'), stage: Annotated[Optional[str], typer.Option(help='Optional stage filter (candidate/validated/production/archived)')] = None, limit: Annotated[int, typer.Option(help='Max rows to print')] = 30)`
- `research_registry_promote(registry: Annotated[Path, typer.Option(help='Path to model registry JSON')] = Path('models/registry.json'), run_id: Annotated[str, typer.Option(help='Run ID to promote')] = '', stage: Annotated[str, typer.Option(help='Target stage (validated/production/archived/candidate)')] = 'validated', force: Annotated[bool, typer.Option(help='Allow production promotion without validated stage')] = False, output_json: Annotated[Optional[Path], typer.Option(help='Write promotion result JSON')] = None)`
- `import_data(input_csv: Annotated[Path, typer.Option(help='Path to CGM CSV file')], output_dir: Annotated[Path, typer.Option(help='Output directory for scenario + standard CSV')] = Path('./results/imported'), data_format: Annotated[str, typer.Option(help='Data format preset: generic, dexcom, libre, carelink')] = 'generic', scenario_name: Annotated[str, typer.Option(help='Scenario name')] = 'Imported CGM Scenario', scenario_version: Annotated[str, typer.Option(help='Scenario version')] = '1.0', time_unit: Annotated[str, typer.Option(help='Timestamp unit: minutes or seconds')] = 'minutes', carb_threshold: Annotated[float, typer.Option(help='Minimum carbs (g) to create a meal event')] = 0.1, scenario_path: Annotated[Optional[Path], typer.Option(help='Optional output scenario path')] = None, data_path: Annotated[Optional[Path], typer.Option(help='Optional output standard CSV path')] = None, mapping: Annotated[List[str], typer.Option('--map', help='Column mapping key=value (e.g., timestamp=Time, glucose=SGV)')] = [])`
- `import_carelink(input_csv: Annotated[Path, typer.Option(help='Path to a Medtronic CareLink CSV export')], output_dir: Annotated[Path, typer.Option(help='Output directory for imported CareLink artifacts')] = Path('./results/imported_carelink'), scenario_name: Annotated[str, typer.Option(help='Scenario name')] = 'Imported CareLink Scenario', scenario_version: Annotated[str, typer.Option(help='Scenario version')] = '1.0', scenario_path: Annotated[Optional[Path], typer.Option(help='Optional output scenario path')] = None, data_path: Annotated[Optional[Path], typer.Option(help='Optional output standard CSV path')] = None, summary_path: Annotated[Optional[Path], typer.Option(help='Optional output summary JSON path')] = None, carb_threshold: Annotated[float, typer.Option(help='Minimum carbs (g) to create a meal event')] = 0.1)`
- `carelink_workbench(input_csv: Annotated[Path, typer.Option(help='Path to a Medtronic CareLink CSV export')], output_dir: Annotated[Path, typer.Option(help='Output directory for the personal CareLink workspace')] = Path('./results/carelink_workbench'), scenario_name: Annotated[str, typer.Option(help='Scenario name for the generated experiment scenario')] = 'Imported CareLink Scenario', scenario_version: Annotated[str, typer.Option(help='Scenario version')] = '1.0', carb_threshold: Annotated[float, typer.Option(help='Minimum carbs (g) to create a meal event')] = 0.1, create_dev_mdmp_cert: Annotated[bool, typer.Option('--create-dev-mdmp-cert/--no-create-dev-mdmp-cert', help='Generate a local development MDMP certificate and keypair for the AI assistant.')] = True, grade: Annotated[str, typer.Option(help='Grade to embed in the local development MDMP certificate.')] = 'research_grade', expires_days: Annotated[int, typer.Option(help='Certificate expiry window in days for local development certs.')] = 30, key_dir: Annotated[Optional[Path], typer.Option(help='Optional directory for the generated local MDMP keypair.')] = None) -> None`
- `import_wizard()`
- `import_demo(output_dir: Annotated[Path, typer.Option(help='Output directory for scenario + CSV')] = Path('./results/demo_import'), scenario_name: Annotated[str, typer.Option(help='Scenario name')] = 'Demo CGM Scenario', export_raw: Annotated[bool, typer.Option(help='Export the raw demo CSV into output dir')] = True)`
- `import_nightscout_cmd(url: Annotated[str, typer.Option(help='Nightscout base URL. Use https for non-local hosts.')], output_dir: Annotated[Path, typer.Option(help='Output directory for scenario + CSV')] = Path('./results/nightscout_import'), api_secret: Annotated[Optional[str], typer.Option(help='API secret (if required)')] = None, api_secret_env: Annotated[Optional[str], typer.Option(help='Environment variable name containing the Nightscout API secret.')] = None, api_secret_file: Annotated[Optional[Path], typer.Option(help='Path to a file containing the Nightscout API secret.')] = None, token: Annotated[Optional[str], typer.Option(help='API token (if required)')] = None, token_env: Annotated[Optional[str], typer.Option(help='Environment variable name containing the Nightscout API token.')] = None, token_file: Annotated[Optional[Path], typer.Option(help='Path to a file containing the Nightscout API token.')] = None, start: Annotated[Optional[str], typer.Option(help='Start time (ISO string)')] = None, end: Annotated[Optional[str], typer.Option(help='End time (ISO string)')] = None, limit: Annotated[Optional[int], typer.Option(help='Limit number of entries')] = None, scenario_name: Annotated[str, typer.Option(help='Scenario name')] = 'Nightscout Import')`
- `import_tidepool_cmd(base_url: Annotated[str, typer.Option(help='Tidepool API base URL. Use https for non-local hosts.')] = 'https://api.tidepool.org', token: Annotated[Optional[str], typer.Option(help='Tidepool session token')] = None, token_env: Annotated[Optional[str], typer.Option(help='Environment variable name containing the Tidepool session token.')] = None, token_file: Annotated[Optional[Path], typer.Option(help='Path to a file containing the Tidepool session token.')] = None, user_id: Annotated[Optional[str], typer.Option(help='Optional Tidepool user id. Defaults to the current authenticated user.')] = None, start: Annotated[Optional[str], typer.Option(help='Optional ISO-8601 start timestamp.')] = None, end: Annotated[Optional[str], typer.Option(help='Optional ISO-8601 end timestamp.')] = None, output_dir: Annotated[Path, typer.Option(help='Directory for imported scenario and standard CSV.')] = Path('results/tidepool_import'), scenario_name: Annotated[str, typer.Option(help='Scenario name written into scenario.json.')] = 'Tidepool Import')`
- `medtronic_live_cmd(base_url: Annotated[str, typer.Option(help='Authorized Medtronic/CareLink live relay base URL. Use https for non-local hosts.')], endpoint_path: Annotated[str, typer.Option(help='Read-only JSON endpoint path on the authorized relay.')] = '/carelink/live', output_dir: Annotated[Path, typer.Option(help='Directory for live timeline, standard CSV, and latest snapshot JSON.')] = Path('./results/medtronic_live'), token: Annotated[Optional[str], typer.Option(help='Bearer token for the authorized relay.')] = None, token_env: Annotated[Optional[str], typer.Option(help='Environment variable name containing the bearer token.')] = None, token_file: Annotated[Optional[Path], typer.Option(help='Path to a file containing the bearer token.')] = None, device_id: Annotated[Optional[str], typer.Option(help='Optional pump/device id query parameter.')] = None, patient_id: Annotated[Optional[str], typer.Option(help='Optional patient id query parameter.')] = None, since: Annotated[Optional[str], typer.Option(help='Optional ISO-8601 lower-bound query parameter.')] = None, limit: Annotated[Optional[int], typer.Option(help='Optional max records query parameter.')] = None, samples: Annotated[int, typer.Option(help='Number of polling samples. Use 0 to poll until interrupted.')] = 1, poll_seconds: Annotated[float, typer.Option(help='Seconds between polling samples.')] = 30.0, source: Annotated[str, typer.Option(help='Source label written into IINTS outputs.')] = 'medtronic_carelink_live') -> None`
- `medtronic_pump_direct_cmd(transport: Annotated[str, typer.Option(help='Direct pump transport: simulated or official-module.')] = 'simulated', official_factory: Annotated[Optional[str], typer.Option(help='Approved internal factory reference for official-module mode, e.g. package.module:create_transport.')] = None, read_only_confirm: Annotated[Optional[str], typer.Option(help='Required confirmation string for official-module hardware transport.')] = None, output_dir: Annotated[Path, typer.Option(help='Directory for pump_timeline.csv, cgm_standard.csv, and pump_latest.json.')] = Path('./results/medtronic_pump_direct'), samples: Annotated[int, typer.Option(help='Number of snapshots to read. Use 0 to poll until interrupted.')] = 1, poll_seconds: Annotated[float, typer.Option(help='Seconds between snapshots.')] = 30.0, simulated_seed: Annotated[int, typer.Option(help='Seed for simulated bench transport.')] = 42, simulated_start_glucose_mgdl: Annotated[float, typer.Option(help='Initial glucose for simulated bench transport.')] = 118.0, simulated_step_minutes: Annotated[float, typer.Option(help='Minutes advanced per simulated snapshot.')] = 5.0) -> None`
- `check_deps()`
- `algorithms_list()`
- `algorithms_info(name: Annotated[str, typer.Argument(help='Algorithm display name')])`
- `plugin_install(path: Annotated[Path, typer.Argument(help='Path to a local InsulinAlgorithm .py file')], name: Annotated[Optional[str], typer.Option(help='Optional display name override')] = None)`
- `plugin_register_algo(path: Annotated[Path, typer.Argument(help='Path to a local InsulinAlgorithm .py file')], name: Annotated[Optional[str], typer.Option(help='Optional display name override')] = None)`
- `plugin_register_patient_model(path: Annotated[Path, typer.Argument(help='Path to a local patient model .py file')], name: Annotated[Optional[str], typer.Option(help='Optional display name override')] = None)`
- `plugin_register_data_source(path: Annotated[Path, typer.Argument(help='Path to a local data source .py file')], name: Annotated[Optional[str], typer.Option(help='Optional display name override')] = None)`
- `plugin_register_validator(path: Annotated[Path, typer.Argument(help='Path to a local validator .py file')], name: Annotated[Optional[str], typer.Option(help='Optional display name override')] = None)`
- `plugin_list(kind: Annotated[Optional[str], typer.Option(help='Filter by kind: algorithm, patient_model, data_source, validator')] = None)`
- `plugin_uninstall(name: Annotated[str, typer.Argument(help='Local plugin display name')], kind: Annotated[Optional[str], typer.Option(help='Optional kind filter')] = None, remove_file: Annotated[bool, typer.Option(help='Also delete the copied plugin file from the plugin home')] = False)`
- `patientmodel_list()`
- `jetson_doctor()`
- `jetson_endurance_start(algo: Annotated[Path, typer.Option('--algo', help='Path to an InsulinAlgorithm Python file')], predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional LSTM predictor checkpoint')] = None, duration: Annotated[str, typer.Option(help='Duration such as 1h, 24h, 7d, 30d')] = '24h', output_dir: Annotated[Path, typer.Option(help='Output directory for the endurance study')] = Path('results/jetson_endurance'), profile: Annotated[str, typer.Option(help='Endurance profile name')] = 'mixed_adversarial', seed: Annotated[int, typer.Option(help='Deterministic simulation seed')] = 42, patient_model: Annotated[str, typer.Option(help='Patient model name passed to PatientFactory')] = 'bergman', sensor_profile: Annotated[str, typer.Option(help='Sensor profile for the simulated CGM stream')] = 'free_living_cgm', custom_profile: Annotated[Optional[Path], typer.Option(help='YAML file for --profile custom')] = None, time_step: Annotated[int, typer.Option(help='Simulation step size in minutes')] = 5, checkpoint_interval: Annotated[int, typer.Option(help='Checkpoint interval in simulated minutes.')] = 360, hardware_sample_interval: Annotated[int, typer.Option(help='Hardware telemetry interval in simulated minutes.')] = 60, status_interval_steps: Annotated[int, typer.Option(help='How often to persist status and partial CSV data.')] = 25, wall_clock: Annotated[bool, typer.Option('--wall-clock/--accelerated', help='Use real wall-clock pacing so 1d takes an actual 24 hours instead of finishing as fast as possible.')] = False, research_export: Annotated[bool, typer.Option('--research-export/--no-research-export', help='Write a predictor-training dataset and research manifest next to the normal endurance artifacts.')] = True, finalize_research: Annotated[bool, typer.Option('--finalize-research/--no-finalize-research', help='After the endurance run, train local research models and write a held-out evaluation report.')] = False, resume: Annotated[bool, typer.Option(help='Resume from the latest snapshot in the output directory')] = False)`
- `jetson_endurance_finalize_research(output_dir: Annotated[Path, typer.Option(help='Completed endurance output directory')], train_predictor: Annotated[bool, typer.Option('--train-predictor/--skip-predictor', help='Train a glucose predictor from the exported endurance dataset when enough rows are available.')] = True, train_neural: Annotated[bool, typer.Option('--train-neural/--skip-neural', help='Train the PyTorch controller in addition to the auditable linear baseline.')] = True, duration_minutes: Annotated[int, typer.Option(help='Closed-loop evaluation duration per held-out run in minutes.')] = 1440)`
- `jetson_endurance_status(output_dir: Annotated[Path, typer.Option(help='Endurance output directory')])`
- `jetson_endurance_monitor(output_dir: Annotated[Path, typer.Option(help='Endurance output directory')], watch: Annotated[bool, typer.Option(help='Refresh while the run is active')] = False, interval_seconds: Annotated[int, typer.Option(help='Refresh interval for --watch')] = 5)`
- `jetson_endurance_stop(output_dir: Annotated[Path, typer.Option(help='Endurance output directory')], generate_report: Annotated[bool, typer.Option(help='Ask the runner to finalize reports before stopping')] = False)`
- `jetson_endurance_export(output_dir: Annotated[Path, typer.Option(help='Endurance output directory')], output: Annotated[Path, typer.Option(help='Destination zip archive')])`
- `jetson_endurance_install_service(algo: Annotated[Path, typer.Option('--algo', help='Path to an InsulinAlgorithm Python file')], duration: Annotated[str, typer.Option(help='Duration such as 24h, 7d, 30d')] = '7d', output_dir: Annotated[Path, typer.Option(help='Output directory for the endurance study')] = Path('results/jetson_endurance'), predictor_path: Annotated[Optional[Path], typer.Option('--predictor', help='Optional LSTM predictor checkpoint')] = None, profile: Annotated[str, typer.Option(help='Endurance profile name')] = 'mixed_adversarial', seed: Annotated[int, typer.Option(help='Deterministic simulation seed')] = 42, wall_clock: Annotated[bool, typer.Option('--wall-clock/--accelerated', help='Write a service that paces the run in real time instead of accelerated simulation time.')] = False, service_path: Annotated[Optional[Path], typer.Option(help='Where to write the systemd unit file')] = None)`
- `docs_algo(algo_path: Annotated[Path, typer.Option(help='Path to the algorithm Python file to document')])`
- `benchmark(algo_to_benchmark: Annotated[Path, typer.Option(help='Path to the AI algorithm Python file to benchmark')], patient_configs_dir: Annotated[Path, typer.Option(help='Directory containing patient configuration YAML files')] = Path('src/iints/data/virtual_patients'), scenarios_dir: Annotated[Path, typer.Option(help='Directory containing scenario JSON files')] = Path('scenarios'), duration: Annotated[int, typer.Option(help='Simulation duration in minutes for each run')] = 720, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes')] = 5, output_dir: Annotated[Optional[Path], typer.Option(help='Directory to save all benchmark results')] = None, seed: Annotated[Optional[int], typer.Option(help='Base seed for deterministic runs')] = None)`
- `edge_benchmark(algo: Annotated[Path, typer.Option(help='Path to the insulin algorithm Python file used for the edge benchmark.')], output_json: Annotated[Path, typer.Option(help='Output JSON path for the hardware benchmark results.')] = Path('results/edge_benchmark.json'), patient_config: Annotated[str, typer.Option(help='Patient configuration name or YAML path.')] = 'default_patient', patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model type.')] = 'auto', scenario_profile: Annotated[str, typer.Option(help='Digital patient scenario profile.')] = 'normal_day', steps: Annotated[int, typer.Option(help='Number of simulated steps used for throughput measurement.')] = 72, platform_name: Annotated[str, typer.Option('--platform', help="Platform label written into the benchmark report. Use 'auto' to detect locally.")] = 'auto', api_host: Annotated[str, typer.Option(help='Host used for the local dashboard probe.')] = '127.0.0.1', api_port: Annotated[int, typer.Option(help='Port used for the local dashboard probe.')] = 8766, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override.')] = None) -> None`
- `edge_setup(output_dir: Annotated[Path, typer.Option(help='Directory where the edge-ready project scaffold should be written.')] = Path('iints_edge_demo'), board: Annotated[str, typer.Option(help='Edge board target: raspberry_pi or uno_q.')] = 'raspberry_pi', workspace_name: Annotated[str, typer.Option(help='Workspace folder name used for the persistent patient runtime.')] = 'patient_runtime', scenario_profile: Annotated[str, typer.Option(help='Initial live scenario profile.')] = 'normal_day', patient_config: Annotated[str, typer.Option(help='Patient configuration name or YAML path.')] = 'default_patient', patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model type.')] = 'auto', mode: Annotated[str, typer.Option(help='Clock mode for the generated edge project.')] = 'demo-time', speed: Annotated[str, typer.Option(help='Acceleration factor for demo-time mode. Accepts 60 or 60x.')] = '60x', api_host: Annotated[str, typer.Option(help='Dashboard host to bake into the generated runtime config.')] = '127.0.0.1', api_port: Annotated[int, typer.Option(help='Dashboard port to bake into the generated runtime config.')] = 8765, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override.')] = None, service_name: Annotated[str, typer.Option(help='systemd service name without the .service suffix.')] = 'iints-digital-patient', user_name: Annotated[Optional[str], typer.Option(help='Linux user that should own the generated systemd service.')] = None, uno_bridge_port: Annotated[Optional[str], typer.Option(help='Optional UNO Q serial port to bake into a generated bridge systemd service.')] = None, uno_bridge_service_name: Annotated[str, typer.Option(help='UNO Q bridge systemd service name without the .service suffix.')] = 'iints-uno-q-bridge') -> None`
- `edge_quickstart(board: Annotated[str, typer.Option(help='Edge board target: raspberry_pi or uno_q.')] = 'raspberry_pi', output_dir: Annotated[Optional[Path], typer.Option(help='Project directory to create. Defaults to iints_pi_demo or iints_uno_q_demo.')] = None, scenario_profile: Annotated[str, typer.Option(help='Initial live scenario profile.')] = 'expo_hot_start', patient_config: Annotated[str, typer.Option(help='Patient configuration name or YAML path.')] = 'default_patient', patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model type.')] = 'auto', mode: Annotated[str, typer.Option(help='Clock mode for the generated edge project.')] = 'demo-time', speed: Annotated[str, typer.Option(help='Acceleration factor for demo-time mode. Accepts 60 or 60x.')] = '60x', api_host: Annotated[str, typer.Option(help='Dashboard host to bake into the generated runtime config.')] = '127.0.0.1', api_port: Annotated[int, typer.Option(help='Dashboard port to bake into the generated runtime config.')] = 8765, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override.')] = None, start: Annotated[bool, typer.Option('--start/--no-start', help='Start the Linux-side digital patient after creating the project.')] = True, reset: Annotated[bool, typer.Option(help='Reset runtime state when starting.')] = True, foreground: Annotated[bool, typer.Option(help='Run in the foreground instead of spawning the daemon.')] = False, bridge_port: Annotated[str, typer.Option(help='UNO Q bridge port for printed commands and optional testing. Use auto for autodetect.')] = 'auto', test_bridge: Annotated[bool, typer.Option(help='After setup, run one UNO Q bridge test if a board is connected.')] = False, max_steps: Annotated[Optional[int], typer.Option('--max-steps', hidden=True)] = None) -> None`
- `edge_deploy(host: Annotated[str, typer.Option(help='Remote Raspberry Pi hostname or IP address.')], user_name: Annotated[Optional[str], typer.Option('--user', help='Optional remote SSH username.')] = None, ssh_port: Annotated[int, typer.Option(help='SSH port used for the remote Raspberry Pi.')] = 22, remote_dir: Annotated[str, typer.Option(help='Target project directory on the Raspberry Pi.')] = '~/iints_pi_demo', local_output_dir: Annotated[Path, typer.Option(help='Local scaffold directory that will also be synced to the Pi.')] = Path('iints_pi_demo'), board: Annotated[str, typer.Option(help='Edge board target: raspberry_pi or uno_q.')] = 'raspberry_pi', workspace_name: Annotated[str, typer.Option(help='Workspace folder name used for the persistent patient runtime.')] = 'patient_runtime', scenario_profile: Annotated[str, typer.Option(help='Initial live scenario profile deployed to the Pi.')] = 'expo_hot_start', patient_config: Annotated[str, typer.Option(help='Patient configuration name or YAML path.')] = 'default_patient', patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model type.')] = 'auto', mode: Annotated[str, typer.Option(help='Clock mode for the generated edge project.')] = 'demo-time', speed: Annotated[str, typer.Option(help='Acceleration factor for demo-time mode. Accepts 60 or 60x.')] = '60x', api_host: Annotated[str, typer.Option(help='Dashboard host to bake into the generated runtime config. Keep 127.0.0.1 when using Raspberry Pi Connect.')] = '127.0.0.1', api_port: Annotated[int, typer.Option(help='Dashboard port to bake into the generated runtime config.')] = 8765, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override.')] = None, service_name: Annotated[str, typer.Option(help='systemd service name without the .service suffix.')] = 'iints-digital-patient', install_autostart: Annotated[bool, typer.Option(help='Install the generated systemd service, watchdog timer, and desktop autostart on the Pi.')] = True, start_runtime: Annotated[bool, typer.Option(help='Start the Maker Faire runtime after deployment.')] = True, enable_connect_linger: Annotated[bool, typer.Option(help='Run `loginctl enable-linger` on the Pi so Raspberry Pi Connect remote shell keeps working after reboots.')] = True, ssh_timeout_seconds: Annotated[float, typer.Option(help='Timeout per remote SSH step in seconds.')] = 300.0, ssh_retries: Annotated[int, typer.Option(help='How many times to retry a failing SSH step before giving up.')] = 1, uno_bridge_port: Annotated[Optional[str], typer.Option(help='Optional UNO Q serial port on the Pi, for example /dev/ttyACM0. Generates and installs a bridge service.')] = None, uno_bridge_service_name: Annotated[str, typer.Option(help='UNO Q bridge systemd service name without the .service suffix.')] = 'iints-uno-q-bridge', flash_uno_bridge: Annotated[bool, typer.Option(help='Flash the UNO Q bridge sketch remotely after syncing the project. Requires --uno-bridge-port and --uno-fqbn.')] = False, uno_fqbn: Annotated[Optional[str], typer.Option(help='Arduino CLI FQBN used when --flash-uno-bridge is enabled.')] = None, arduino_cli: Annotated[str, typer.Option(help='Arduino CLI executable name or path on the Raspberry Pi.')] = 'arduino-cli', dry_run: Annotated[bool, typer.Option(help='Show the remote deployment plan without executing SSH commands.')] = False, verbose: Annotated[bool, typer.Option(help='Print the raw remote SSH command and deploy stdout for debugging.')] = False) -> None`
- `edge_offline_bundle(output: Annotated[Path, typer.Option(help='Tarball written for offline USB-stick installs.')] = Path('iints_offline.tar.gz'), board: Annotated[str, typer.Option(help='Edge board target baked into the scaffold: raspberry_pi or uno_q.')] = 'raspberry_pi', workspace_name: Annotated[str, typer.Option(help='Workspace folder name used for the persistent patient runtime.')] = 'patient_runtime', scenario_profile: Annotated[str, typer.Option(help='Initial live scenario profile baked into the scaffold.')] = 'expo_hot_start', patient_config: Annotated[str, typer.Option(help='Patient configuration name or YAML path.')] = 'default_patient', patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model type.')] = 'auto', mode: Annotated[str, typer.Option(help='Clock mode baked into the generated edge project.')] = 'demo-time', speed: Annotated[str, typer.Option(help='Acceleration factor for demo-time mode. Accepts 60 or 60x.')] = '60x', api_host: Annotated[str, typer.Option(help='Dashboard host baked into the runtime config. Keep 127.0.0.1 for Pi Connect setups.')] = '127.0.0.1', api_port: Annotated[int, typer.Option(help='Dashboard port baked into the runtime config.')] = 8765, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override.')] = None) -> None`
- `edge_study(algo: Annotated[Path, typer.Option(help='Path to the algorithm Python file.')], output_dir: Annotated[Path, typer.Option(help='Root directory for the Pi-generated study bundle.')] = Path('results/pi_study'), preset: Annotated[str, typer.Option(help='Study preset: default or eucys.')] = 'default', profile_set: Annotated[str, typer.Option(help='Patient profile set to evaluate on the Pi.')] = DEFAULT_PROFILE_SET, scenarios: Annotated[str, typer.Option(help='Optional comma-separated scenario slugs to run.')] = '', seeds: Annotated[str, typer.Option(help='Comma-separated seed list.')] = '1,2,3,4,5', duration: Annotated[Optional[int], typer.Option(help='Override scenario duration in minutes.')] = None, time_step: Annotated[int, typer.Option(help='Simulation time step in minutes.')] = 5, include_default_baselines: Annotated[bool, typer.Option('--include-default-baselines/--no-include-default-baselines', help='Include the default baseline registry in the Pi study matrix.')] = True, extra_algorithms: Annotated[str, typer.Option(help='Comma-separated additional comparison algorithm labels.')] = '', prepare_ai: Annotated[bool, typer.Option(help='Generate AI-ready artifacts for non-corrupted study arms.')] = False) -> None`
- `edge_long_study(config: Annotated[Path, typer.Option(help='YAML config describing the multi-day edge study.')], project_dir: Annotated[Path, typer.Option(help='Edge project directory that contains the algorithms folder and runtime scaffold.')] = Path('.'), resume: Annotated[bool, typer.Option(help='Resume from the next incomplete day by inspecting long_study_index.csv.')] = False) -> None`
- `edge_study_snapshot(project_dir: Annotated[Path, typer.Option(help='Edge project directory used to resolve relative study paths.')] = Path('.'), input_dir: Annotated[Path, typer.Option(help='Long-study directory to snapshot.')] = Path('results/long_study'), output: Annotated[Path, typer.Option(help='Output directory or .tar.gz path for the snapshot archive.')] = Path('snapshots')) -> None`
- `edge_study_export(project_dir: Annotated[Path, typer.Option(help='Edge project directory used to resolve relative study paths.')] = Path('.'), input_dir: Annotated[Path, typer.Option(help='Long-study directory to export.')] = Path('results/long_study'), output: Annotated[Path, typer.Option(help='Zip archive written for transfer to another device.')] = Path('results/long_study_export.zip')) -> None`
- `edge_doctor(board: Annotated[str, typer.Option(help='Board target to validate: raspberry_pi or uno_q.')] = 'raspberry_pi', project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime') -> None`
- `edge_remote_status(host: Annotated[str, typer.Option(help='Remote Raspberry Pi hostname or IP address.')], user_name: Annotated[Optional[str], typer.Option('--user', help='Optional remote SSH username.')] = None, ssh_port: Annotated[int, typer.Option(help='SSH port used for the remote Raspberry Pi.')] = 22, remote_dir: Annotated[str, typer.Option(help='Target project directory on the Raspberry Pi.')] = '~/iints_pi_demo', ssh_timeout_seconds: Annotated[float, typer.Option(help='Timeout per remote SSH step in seconds.')] = 60.0, ssh_retries: Annotated[int, typer.Option(help='How many times to retry a failing SSH step before giving up.')] = 0) -> None`
- `edge_remote_reset(host: Annotated[str, typer.Option(help='Remote Raspberry Pi hostname or IP address.')], user_name: Annotated[Optional[str], typer.Option('--user', help='Optional remote SSH username.')] = None, ssh_port: Annotated[int, typer.Option(help='SSH port used for the remote Raspberry Pi.')] = 22, remote_dir: Annotated[str, typer.Option(help='Target project directory on the Raspberry Pi.')] = '~/iints_pi_demo', scenario_profile: Annotated[Optional[str], typer.Option(help='Optional profile to load after reset.')] = None, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override for the reset profile.')] = None, ssh_timeout_seconds: Annotated[float, typer.Option(help='Timeout per remote SSH step in seconds.')] = 60.0, ssh_retries: Annotated[int, typer.Option(help='How many times to retry a failing SSH step before giving up.')] = 0) -> None`
- `edge_remote_stop(host: Annotated[str, typer.Option(help='Remote Raspberry Pi hostname or IP address.')], user_name: Annotated[Optional[str], typer.Option('--user', help='Optional remote SSH username.')] = None, ssh_port: Annotated[int, typer.Option(help='SSH port used for the remote Raspberry Pi.')] = 22, remote_dir: Annotated[str, typer.Option(help='Target project directory on the Raspberry Pi.')] = '~/iints_pi_demo', ssh_timeout_seconds: Annotated[float, typer.Option(help='Timeout per remote SSH step in seconds.')] = 60.0, ssh_retries: Annotated[int, typer.Option(help='How many times to retry a failing SSH step before giving up.')] = 0) -> None`
- `edge_up(project_dir: Annotated[Path, typer.Option(help='Edge project directory created by `iints edge setup`.')] = Path('.'), workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime', foreground: Annotated[bool, typer.Option(help='Run the digital patient in the foreground instead of spawning the daemon.')] = False, reset: Annotated[bool, typer.Option(help='Reset the runtime state before starting.')] = False, max_steps: Annotated[Optional[int], typer.Option('--max-steps', hidden=True)] = None) -> None`
- `makerfaire_up(project_dir: Annotated[Path, typer.Option(help='Edge project directory created by `iints edge setup`.')] = Path('.'), workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime', scenario_profile: Annotated[str, typer.Option(help='Booth-ready scenario profile. Defaults to expo_hot_start.')] = 'expo_hot_start', seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override for the booth reset/start.')] = None, reset: Annotated[bool, typer.Option(help='Reset into the booth profile when the runtime is already running.')] = True, foreground: Annotated[bool, typer.Option(help='Run in the foreground instead of using the background daemon.')] = False, show_kiosk: Annotated[bool, typer.Option(help='Print the kiosk panel after startup so you can copy the Pi display URL quickly.')] = True) -> None`
- `makerfaire_autostart(project_dir: Annotated[Path, typer.Option(help='Edge project directory created by `iints edge setup`.')] = Path('.'), workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime') -> None`
- `makerfaire_watchdog(project_dir: Annotated[Path, typer.Option(help='Edge project directory created by `iints edge setup`.')] = Path('.'), workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime', scenario_profile: Annotated[str, typer.Option(help='Booth-safe scenario profile to restore if the runtime must be restarted.')] = 'expo_hot_start', seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override used when the watchdog has to restart the runtime.')] = None, quiet: Annotated[bool, typer.Option(help='Print only minimal output. Useful for watchdog scripts and timers.')] = False) -> None`
- `edge_kiosk(project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace: Annotated[Optional[Path], typer.Option(help='Optional runtime workspace override.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime') -> None`
- `edge_reset(project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace: Annotated[Optional[Path], typer.Option(help='Optional runtime workspace override.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime', scenario_profile: Annotated[Optional[str], typer.Option(help='Optional profile to load after reset. Defaults to expo_hot_start.')] = None, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override for the reset profile.')] = None) -> None`
- `edge_stop(project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace: Annotated[Optional[Path], typer.Option(help='Optional runtime workspace override.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime') -> None`
- `edge_service(project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace: Annotated[Optional[Path], typer.Option(help='Optional runtime workspace override.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime', output: Annotated[Optional[Path], typer.Option(help='Optional output service file path.')] = None, service_name: Annotated[str, typer.Option(help='systemd service name without the .service suffix.')] = 'iints-digital-patient', user_name: Annotated[Optional[str], typer.Option(help='Linux user that should run the service. Defaults to the current shell user.')] = None, python_path: Annotated[Optional[Path], typer.Option(help='Python executable used in ExecStart. Defaults to the current interpreter.')] = None) -> None`
- `edge_status(workspace: Annotated[Optional[Path], typer.Option(help='Workspace directory for the persistent digital patient state.')] = None, project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime') -> None`
- `edge_bundle(workspace: Annotated[Optional[Path], typer.Option(help='Workspace directory for the persistent digital patient state.')] = None, project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime', output: Annotated[Path, typer.Option(help='ZIP archive written for workstation-side analysis.')] = Path('results/edge_runtime_bundle.zip'), include_log: Annotated[bool, typer.Option(help='Include the patient log in the archive.')] = True, include_database: Annotated[bool, typer.Option(help='Include the SQLite runtime database in the archive.')] = True) -> None`
- `edge_update(output_script: Annotated[Path, typer.Option(help='Where to write the edge update shell script.')] = Path('update_edge_runtime.sh'), profile: Annotated[str, typer.Option(help='Install profile to upgrade: edge or full.')] = 'edge', version_pin: Annotated[Optional[str], typer.Option(help='Optional exact SDK version pin, for example 1.5.2.')] = None) -> None`
- `edge_hardware_bridge(board: Annotated[str, typer.Option(help='Hardware bridge target. Currently supported: uno_q.')] = 'uno_q', output_dir: Annotated[Path, typer.Option(help='Directory where the hardware bridge scaffold should be written.')] = Path('uno_q_bridge')) -> None`
- `edge_pump_init(output_dir: Annotated[Path, typer.Option(help='Directory where the Pico pump lab workspace should be written.')] = Path('iints_pico_pump_lab'), algorithm: Annotated[Optional[Path], typer.Option(help='Optional existing SDK algorithm to copy into the lab workspace.')] = None) -> None`
- `edge_pump_firmware(output_dir: Annotated[Path, typer.Option(help='Directory where locked Pico bench firmware should be written.')] = Path('pico_pump_firmware')) -> None`
- `edge_pump_package(algorithm: Annotated[Path, typer.Option(help='SDK algorithm Python file to package for bench-only Pico testing.')], output_dir: Annotated[Path, typer.Option(help='Output bundle directory.')] = Path('pico_pump_bundle'), safety_contract: Annotated[Optional[Path], typer.Option(help='Optional zero-delivery safety contract JSON.')] = None, label: Annotated[str, typer.Option(help='Human-readable bundle label written into the manifest.')] = 'pico_pump_bench') -> None`
- `edge_pump_upload(bundle_dir: Annotated[Path, typer.Option(help='Bundle directory from `iints edge pump package`.')], mount_dir: Annotated[Path, typer.Option(help='Mounted writable Pico/CircuitPython-style drive or a test folder.')], bench_only_confirm: Annotated[str, typer.Option(help=f'Must be exactly: {PICO_PUMP_CONFIRMATION}')] = '', write: Annotated[bool, typer.Option('--write', help='Actually copy files. Without this flag, only prints the copy plan.')] = False) -> None`
- `edge_pump_serial_test(port: Annotated[str, typer.Option(help='Serial port for the Pico bench firmware, for example /dev/ttyACM0 or /dev/tty.usbmodem*.')], baudrate: Annotated[int, typer.Option(help='Serial baud rate used by the Pico bench firmware.')] = PICO_PUMP_BAUDRATE, timeout_seconds: Annotated[float, typer.Option(help='Read timeout per command in seconds.')] = 1.5) -> None`
- `pump_init(output_dir: Annotated[Path, typer.Option(help='Directory where the Pico pump lab workspace should be written.')] = Path('iints_pico_pump_lab'), algorithm: Annotated[Optional[Path], typer.Option(help='Optional existing SDK algorithm to copy into the lab workspace.')] = None) -> None`
- `pump_compile(algorithm: Annotated[Path, typer.Option(help='SDK algorithm Python file to compile/package for bench-only Pico testing.')], output_dir: Annotated[Path, typer.Option(help='Output bundle directory.')] = Path('pico_pump_bundle'), safety_contract: Annotated[Optional[Path], typer.Option(help='Optional zero-delivery safety contract JSON.')] = None, label: Annotated[str, typer.Option(help='Human-readable bundle label written into the manifest.')] = 'pico_pump_bench') -> None`
- `pump_bench_test(bundle_dir: Annotated[Path, typer.Option(help='Bundle directory from `iints pump compile`.')], output_json: Annotated[Optional[Path], typer.Option(help='Optional JSON report path.')] = None, port: Annotated[Optional[str], typer.Option(help='Optional Pico serial port for an additional non-actuating smoke test.')] = None, baudrate: Annotated[int, typer.Option(help='Serial baud rate used by the Pico bench firmware.')] = PICO_PUMP_BAUDRATE, timeout_seconds: Annotated[float, typer.Option(help='Read timeout per serial command in seconds.')] = 1.5) -> None`
- `pump_upload(bundle_dir: Annotated[Path, typer.Option(help='Bundle directory from `iints pump compile`.')], mount_dir: Annotated[Path, typer.Option(help='Mounted writable Pico/CircuitPython-style drive or a test folder.')], bench_only_confirm: Annotated[str, typer.Option(help=f'Must be exactly: {PICO_PUMP_CONFIRMATION}')] = '', write: Annotated[bool, typer.Option('--write', help='Actually copy files. Without this flag, only prints the copy plan.')] = False) -> None`
- `edge_bridge_test(port: Annotated[Optional[str], typer.Option(help='Serial port for the UNO Q STM32 side. Use `auto` or omit it if exactly one port is connected.')] = None, baudrate: Annotated[int, typer.Option(help='Serial baud rate used by the UNO Q bridge sketch.')] = UNO_Q_BRIDGE_BAUDRATE, delay_seconds: Annotated[float, typer.Option(help='Pause between test states in seconds.')] = 0.75) -> None`
- `edge_bridge_run(port: Annotated[Optional[str], typer.Option(help='Serial port for the UNO Q STM32 side. Use `auto` or omit it if exactly one port is connected.')] = None, workspace: Annotated[Optional[Path], typer.Option(help='Workspace directory for the persistent digital patient state.')] = None, project_dir: Annotated[Optional[Path], typer.Option(help='Optional edge project directory created by `iints edge setup`.')] = None, workspace_name: Annotated[str, typer.Option(help='Workspace folder inside the edge project.')] = 'patient_runtime', baudrate: Annotated[int, typer.Option(help='Serial baud rate used by the UNO Q bridge sketch.')] = UNO_Q_BRIDGE_BAUDRATE, poll_interval: Annotated[float, typer.Option(help='Polling interval in seconds while following runtime status.')] = 1.0, once: Annotated[bool, typer.Option(help='Send the current state once and exit.')] = False, max_cycles: Annotated[Optional[int], typer.Option('--max-cycles', hidden=True)] = None) -> None`
- `edge_bridge_flash(port: Annotated[str, typer.Option(help='Serial port used to upload the UNO Q bridge sketch.')], fqbn: Annotated[str, typer.Option(help='Arduino CLI FQBN for the UNO Q board package.')], project_dir: Annotated[Path, typer.Option(help='Edge project directory created by `iints edge setup`.')] = Path('.'), sketch_dir: Annotated[Optional[Path], typer.Option(help='Optional bridge sketch directory. Defaults to <project-dir>/uno_q_bridge.')] = None, arduino_cli: Annotated[str, typer.Option(help='Arduino CLI executable name or path.')] = 'arduino-cli') -> None`
- `edge_benchmark_alias(algo: Annotated[Path, typer.Option(help='Path to the insulin algorithm Python file used for the edge benchmark.')], output_json: Annotated[Path, typer.Option(help='Output JSON path for the hardware benchmark results.')] = Path('results/edge_benchmark.json'), patient_config: Annotated[str, typer.Option(help='Patient configuration name or YAML path.')] = 'default_patient', patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model type.')] = 'auto', scenario_profile: Annotated[str, typer.Option(help='Digital patient scenario profile.')] = 'normal_day', steps: Annotated[int, typer.Option(help='Number of simulated steps used for throughput measurement.')] = 72, platform_name: Annotated[str, typer.Option('--platform', help="Platform label written into the benchmark report. Use 'auto' to detect locally.")] = 'auto', api_host: Annotated[str, typer.Option(help='Host used for the local dashboard probe.')] = '127.0.0.1', api_port: Annotated[int, typer.Option(help='Port used for the local dashboard probe.')] = 8766, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override.')] = None) -> None`

### Public Constants

- `APP_HELP`
- `IINTS_ASCII_LOGO`

## `iints.cli.patient_cli`

- Source: `src/iints/cli/patient_cli.py`
- Summary: No module docstring.

### Public Functions

- `scenarios() -> None`
- `start(algo: Annotated[Path, typer.Option(help='Path to the insulin algorithm Python file.')], patient_config: Annotated[str, typer.Option(help='Patient configuration name or YAML path.')] = 'default_patient', patient_model: Annotated[str, typer.Option('--patient-model', help='Patient model type: auto, bergman, custom, simglucose.')] = 'auto', scenario_profile: Annotated[str, typer.Option(help='Live day profile: school_day, normal_day, sport_day, bad_carb_count, night_hypo_risk, relaxed_day, expo_hot_start.')] = 'normal_day', workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime'), mode: Annotated[str, typer.Option(help='Clock mode: real-time or demo-time.')] = 'demo-time', speed: Annotated[str, typer.Option(help='Acceleration factor for demo-time mode. Accepts values like 60 or 60x.')] = '60x', api_host: Annotated[str, typer.Option(help='Host for the FastAPI dashboard service. Loopback is the safe default; non-loopback requires --allow-remote-api plus a token.')] = '127.0.0.1', api_port: Annotated[int, typer.Option(help='Port for the local FastAPI dashboard service.')] = 8765, allow_remote_api: Annotated[bool, typer.Option(help='Allow the dashboard API to listen on a non-loopback host. Use this only with --api-token-env, --api-token-file, or --api-token.')] = False, api_token: Annotated[Optional[str], typer.Option(help='Bearer token for dashboard and control access. Prefer --api-token-env or --api-token-file in shared environments.')] = None, api_token_env: Annotated[Optional[str], typer.Option(help='Environment variable name that stores the dashboard/control bearer token.')] = None, api_token_file: Annotated[Optional[Path], typer.Option(help='Path to a file containing the dashboard/control bearer token.')] = None, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic simulation seed.')] = None, foreground: Annotated[bool, typer.Option('--foreground', hidden=True)] = False, max_steps: Annotated[Optional[int], typer.Option('--max-steps', hidden=True)] = None, reset: Annotated[bool, typer.Option('--reset', hidden=True)] = False) -> None`
- `status(workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime')) -> None`
- `inject_meal(carbs: Annotated[float, typer.Option(help='Carbohydrate amount to inject immediately into the live patient.')], workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime')) -> None`
- `pause(workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime')) -> None`
- `resume(workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime')) -> None`
- `expo_reset(scenario_profile: Annotated[Optional[str], typer.Option(help='Optional profile to load after reset. Defaults to expo_hot_start.')] = None, seed: Annotated[Optional[int], typer.Option(help='Optional deterministic seed override for the reset profile.')] = None, workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime')) -> None`
- `stop(workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime')) -> None`
- `export_service(workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime'), output: Annotated[Optional[Path], typer.Option(help='Optional output service file path.')] = None, service_name: Annotated[str, typer.Option(help='systemd service name without the .service suffix.')] = 'iints-digital-patient', user_name: Annotated[Optional[str], typer.Option(help='Linux user that should run the service. Defaults to the current shell user.')] = None, python_path: Annotated[Optional[Path], typer.Option(help='Python executable used in ExecStart. Defaults to the current interpreter.')] = None) -> None`
- `export_uno_bridge(output_dir: Annotated[Path, typer.Option(help='Directory where the UNO Q bridge scaffold should be written.')] = Path('./uno_q_bridge')) -> None`
- `hardware_bridge(board: Annotated[str, typer.Option(help='Hardware bridge target. Currently supported: uno_q.')] = 'uno_q', output_dir: Annotated[Path, typer.Option(help='Directory where the hardware bridge scaffold should be written.')] = Path('./uno_q_bridge')) -> None`
- `kiosk(workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime')) -> None`
- `review(workspace: Annotated[Path, typer.Option(help='Workspace directory for the persistent digital patient state.')] = Path('./digital_patient_runtime'), model: Annotated[str, typer.Option(help='Local Ollama model used for the realism review.')] = DEFAULT_MINISTRAL_MODEL, mdmp_cert: Annotated[Optional[Path], typer.Option(help='Optional MDMP certificate path. If omitted, the live bundle gets a local dev cert.')] = None, output: Annotated[Optional[Path], typer.Option(help='Optional output markdown file for the review.')] = None, ollama_host: Annotated[Optional[str], typer.Option(help='Optional Ollama base URL override.')] = None, minimum_grade: Annotated[str, typer.Option(help='Minimum MDMP grade required before review runs.')] = 'research_grade') -> None`

## `iints.core`

- Source: `src/iints/core/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.algorithms`

- Source: `src/iints/core/algorithms/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.algorithms.battle_runner`

- Source: `src/iints/core/algorithms/battle_runner.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `BattleRunner` | `BattleRunner` | Runs a battle between different insulin algorithms. |

#### `BattleRunner` methods

- `run_battle(self, isf_override: Optional[float] = None, icr_override: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]`
- `print_battle_report(self, battle_report: Dict[str, Any])`

## `iints.core.algorithms.clinical_baseline`

- Source: `src/iints/core/algorithms/clinical_baseline.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ClinicalBaselineAlgorithm` | `ClinicalBaselineAlgorithm(InsulinAlgorithm)` | Conservative clinician-style baseline. |

#### `ClinicalBaselineAlgorithm` methods

- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

## `iints.core.algorithms.correction_bolus`

- Source: `src/iints/core/algorithms/correction_bolus.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `CorrectionBolus` | `CorrectionBolus(InsulinAlgorithm)` | An insulin algorithm that calculates a meal bolus based on carbohydrates and adds a correction bolus if current glucose is above a target. |

#### `CorrectionBolus` methods

- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

## `iints.core.algorithms.discovery`

- Source: `src/iints/core/algorithms/discovery.py`
- Summary: No module docstring.

### Public Functions

- `discover_algorithms() -> Dict[str, Type[InsulinAlgorithm]]`

## `iints.core.algorithms.fixed_basal_bolus`

- Source: `src/iints/core/algorithms/fixed_basal_bolus.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `FixedBasalBolus` | `FixedBasalBolus(InsulinAlgorithm)` | A simple insulin algorithm that delivers a fixed basal rate and a meal bolus based on carbohydrate intake. |

#### `FixedBasalBolus` methods

- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

## `iints.core.algorithms.hybrid_algorithm`

- Source: `src/iints/core/algorithms/hybrid_algorithm.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.algorithms.imitation_controller`

- Source: `src/iints/core/algorithms/imitation_controller.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ExperimentalImitationController` | `ExperimentalImitationController(InsulinAlgorithm)` | Research-only local policy that imitates previously supervised insulin actions. |

#### `ExperimentalImitationController` methods

- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

## `iints.core.algorithms.lstm_algorithm`

- Source: `src/iints/core/algorithms/lstm_algorithm.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.algorithms.mock_algorithms`

- Source: `src/iints/core/algorithms/mock_algorithms.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ConstantDoseAlgorithm` | `ConstantDoseAlgorithm(InsulinAlgorithm)` | Always returns a fixed insulin dose for CI and smoke testing. |
| `RandomDoseAlgorithm` | `RandomDoseAlgorithm(InsulinAlgorithm)` | Returns a random dose within a safe range for stochastic testing. |
| `RunawayAIAlgorithm` | `RunawayAIAlgorithm(InsulinAlgorithm)` | Delivers a maximal bolus during glucose decline to stress-test supervisor. |
| `StackingAIAlgorithm` | `StackingAIAlgorithm(InsulinAlgorithm)` | Delivers repeated boluses over consecutive steps to simulate stacking. |

#### `ConstantDoseAlgorithm` methods

- `get_algorithm_metadata(self) -> AlgorithmMetadata`
- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

#### `RandomDoseAlgorithm` methods

- `get_algorithm_metadata(self) -> AlgorithmMetadata`
- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

#### `RunawayAIAlgorithm` methods

- `get_algorithm_metadata(self) -> AlgorithmMetadata`
- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

#### `StackingAIAlgorithm` methods

- `get_algorithm_metadata(self) -> AlgorithmMetadata`
- `reset(self) -> None`
- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

## `iints.core.algorithms.neural_controller`

- Source: `src/iints/core/algorithms/neural_controller.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ExperimentalNeuralController` | `ExperimentalNeuralController(InsulinAlgorithm)` | Research-only PyTorch policy trained from safety-supervised teacher actions. |

#### `ExperimentalNeuralController` methods

- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

## `iints.core.algorithms.pid_controller`

- Source: `src/iints/core/algorithms/pid_controller.py`
- Summary: Industry-Standard PID Controller - IINTS-AF Simple PID implementation for algorithm comparison

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PIDController` | `PIDController(InsulinAlgorithm)` | Industry-standard PID controller for glucose management |

#### `PIDController` methods

- `predict_insulin(self, data: AlgorithmInput)`
- `reset(self)`
- `get_algorithm_info(self)`

## `iints.core.algorithms.standard_pump_algo`

- Source: `src/iints/core/algorithms/standard_pump_algo.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `StandardPumpAlgorithm` | `StandardPumpAlgorithm(InsulinAlgorithm)` | A simplified algorithm representing a standard insulin pump. It delivers a fixed basal rate and a simple bolus based on carbs, with minimal correction for high glucose. |

#### `StandardPumpAlgorithm` methods

- `predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]`

## `iints.core.device`

- Source: `src/iints/core/device.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.device_manager`

- Source: `src/iints/core/device_manager.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `DeviceManager` | `DeviceManager` | Manages hardware device detection for cross-platform compatibility. Detects MPS (Apple Silicon), CUDA (NVIDIA GPUs), or falls back to CPU. |

#### `DeviceManager` methods

- `get_device(self)`

## `iints.core.devices`

- Source: `src/iints/core/devices/__init__.py`
- Summary: No module docstring.
- Explicit exports: `SensorModel, PumpModel, SENSOR_PROFILES, create_sensor_model`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.devices.models`

- Source: `src/iints/core/devices/models.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SensorReading` | `SensorReading` | No module docstring. |
| `SensorModel` | `SensorModel` | Sensor error model for CGM readings. |
| `PumpDelivery` | `PumpDelivery` | No module docstring. |
| `PumpModel` | `PumpModel` | Pump error model for insulin delivery. |

#### `SensorModel` methods

- `reset(self) -> None`
- `read(self, true_glucose: float, current_time: float) -> SensorReading`
- `get_state(self) -> Dict[str, Any]`
- `set_state(self, state: Dict[str, Any]) -> None`

#### `PumpModel` methods

- `reset(self) -> None`
- `deliver(self, requested_units: float, time_step_minutes: float) -> PumpDelivery`
- `get_state(self) -> Dict[str, Any]`
- `set_state(self, state: Dict[str, Any]) -> None`

### Public Functions

- `create_sensor_model(*, profile: str = 'clinical_cgm', seed: Optional[int] = None, **overrides: Any) -> SensorModel`

### Public Constants

- `SENSOR_PROFILES`

## `iints.core.patient`

- Source: `src/iints/core/patient/__init__.py`
- Summary: No module docstring.
- Explicit exports: `PatientProfile, PatientModel, BergmanPatientModel`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.patient.bergman_model`

- Source: `src/iints/core/patient/bergman_model.py`
- Summary: Bergman Minimal Model — IINTS-AF ================================== ODE-based patient model inspired by the Bergman Minimal Model with an additional gut absorption compartment for realistic carbohydrate dynamics.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `BergmanParameters` | `BergmanParameters` | Physiological parameters for the Bergman Minimal Model. |
| `BergmanPatientModel` | `BergmanPatientModel` | ODE-based patient model providing the same interface as ``CustomPatientModel`` for drop-in use with the IINTS Simulator. |

#### `BergmanPatientModel` methods

- `reset(self) -> None`
- `start_exercise(self, intensity: float) -> None`
- `stop_exercise(self) -> None`
- `update(self, time_step: float, delivered_insulin: float, carb_intake: float = 0.0, current_time: Optional[float] = None, **kwargs) -> float`
- `get_current_glucose(self) -> float`
- `trigger_event(self, event_type: str, value: Any) -> None`
- `get_patient_state(self) -> Dict[str, float]`
- `get_ratio_state(self) -> Dict[str, float]`
- `set_ratio_state(self, isf: Optional[float] = None, icr: Optional[float] = None, basal_rate: Optional[float] = None, dia_minutes: Optional[float] = None) -> None`
- `get_state(self) -> Dict[str, Any]`
- `set_state(self, state: Dict[str, Any]) -> None`

## `iints.core.patient.models`

- Source: `src/iints/core/patient/models.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `CustomPatientModel` | `CustomPatientModel` | A simplified patient model for simulating blood glucose dynamics. This model is intended for educational and stress-testing purposes, not for clinical accuracy. |

#### `CustomPatientModel` methods

- `reset(self)`
- `start_exercise(self, intensity: float)`
- `stop_exercise(self)`
- `update(self, time_step: float, delivered_insulin: float, carb_intake: float = 0.0, current_time: Optional[float] = None, **kwargs) -> float`
- `get_current_glucose(self) -> float`
- `trigger_event(self, event_type: str, value: Any)`
- `get_patient_state(self) -> Dict[str, float]`
- `get_ratio_state(self) -> Dict[str, float]`
- `set_ratio_state(self, isf: Optional[float] = None, icr: Optional[float] = None, basal_rate: Optional[float] = None, dia_minutes: Optional[float] = None) -> None`
- `get_state(self) -> Dict[str, Any]`
- `set_state(self, state: Dict[str, Any]) -> None`

## `iints.core.patient.patient_factory`

- Source: `src/iints/core/patient/patient_factory.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PatientFactory` | `PatientFactory` | Factory for creating different types of patient models. |
| `SimglucosePatientWrapper` | `SimglucosePatientWrapper` | Wrapper for simglucose patients to match CustomPatientModel interface. |

#### `PatientFactory` methods

- `create_patient(patient_type = 'auto', patient_id = None, initial_glucose = 120.0, **kwargs)`
- `get_patient_diversity_set()`

#### `SimglucosePatientWrapper` methods

- `reset(self)`
- `get_current_glucose(self)`
- `update(self, time_step, delivered_insulin, carb_intake = 0.0, **kwargs)`
- `insulin_on_board(self)`
- `carbs_on_board(self)`
- `trigger_event(self, event_type, value)`
- `get_patient_state(self)`

## `iints.core.patient.profile`

- Source: `src/iints/core/patient/profile.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PatientProfile` | `PatientProfile` | User-facing patient profile that maps to the simulator config. |
| `PatientProfilePreset` | `PatientProfilePreset` | Named starter profile with a concise purpose statement. |

#### `PatientProfile` methods

- `to_patient_config(self) -> Dict[str, Any]`

#### `PatientProfilePreset` methods

- `build_profile(self) -> PatientProfile`

### Public Functions

- `get_patient_profile_preset(name: str) -> PatientProfilePreset`

### Public Constants

- `PATIENT_PROFILE_PRESETS`

## `iints.core.physiology_variation`

- Source: `src/iints/core/physiology_variation.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `EmpiricalResidualProfile` | `EmpiricalResidualProfile` | No module docstring. |
| `EmpiricalResidualModel` | `EmpiricalResidualModel` | Small additive model-discrepancy layer sampled from real CGM day residuals. |

#### `EmpiricalResidualProfile` methods

- `from_dict(cls, payload: dict[str, Any]) -> 'EmpiricalResidualProfile'`

#### `EmpiricalResidualModel` methods

- `from_profile_id(cls, profile_id: str, *, seed: int | None = None, scale: float = 1.0) -> 'EmpiricalResidualModel'`
- `reset(self) -> None`
- `offset_at(self, current_time_minutes: float) -> float`
- `get_state(self) -> dict[str, Any]`
- `set_state(self, state: dict[str, Any]) -> None`

### Public Functions

- `load_empirical_residual_profiles() -> list[EmpiricalResidualProfile]`
- `get_empirical_residual_profile(profile_id: str) -> EmpiricalResidualProfile`

## `iints.core.safety`

- Source: `src/iints/core/safety/__init__.py`
- Summary: No module docstring.
- Explicit exports: `SafetyConfig, SafetySupervisor, InputValidator`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.safety.config`

- Source: `src/iints/core/safety/config.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SafetyConfig` | `SafetyConfig` | Central safety configuration for simulator, input validation, and supervisor. |

### Public Constants

- `SENSOR_GLUCOSE_MAX_MGDL`
- `SENSOR_GLUCOSE_MIN_MGDL`
- `SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL`
- `SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL`
- `SIMULATION_GLUCOSE_CEILING_MGDL`
- `SIMULATION_GLUCOSE_FLOOR_MGDL`

## `iints.core.safety.input_validator`

- Source: `src/iints/core/safety/input_validator.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `InputValidator` | `InputValidator` | A biological validation filter for sensor inputs to ensure they are physiologically plausible before being used by an algorithm. This component makes the system robust against common sensor errors. |

#### `InputValidator` methods

- `reset(self) -> None`
- `get_state(self) -> Dict[str, Any]`
- `set_state(self, state: Dict[str, Any]) -> None`
- `validate_glucose(self, glucose_value: float, current_time: float) -> float`
- `validate_insulin(self, dose: float) -> float`

## `iints.core.safety.supervisor`

- Source: `src/iints/core/safety/supervisor.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `IndependentSupervisor` | `IndependentSupervisor(FullSupervisor)` | Safety supervisor that operates independently to validate insulin delivery. Enforces hard constraints on insulin delivery to prevent hypoglycemia and other hazards. |

#### `IndependentSupervisor` methods

- `validate_insulin_dose(self, proposed_dose: float, current_glucose: float, active_insulin: float, time_since_last_dose: float) -> float`

## `iints.core.simulation`

- Source: `src/iints/core/simulation/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.core.simulation.scenario_parser`

- Source: `src/iints/core/simulation/scenario_parser.py`
- Summary: No module docstring.

### Public Functions

- `parse_scenario(file_path: str) -> Tuple[Dict[str, Any], List[StressEvent]]`

## `iints.core.simulator`

- Source: `src/iints/core/simulator.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SimulationLimitError` | `SimulationLimitError(RuntimeError)` | Raised when a simulation violates critical safety limits. |
| `StressEvent` | `StressEvent` | Represents a discrete event that can occur during a simulation for stress testing. |
| `Simulator` | `Simulator` | Orchestrates the interaction between a patient model and an insulin algorithm over a simulated period, including stress-test scenarios. |

#### `Simulator` methods

- `add_stress_event(self, event: StressEvent) -> None`
- `run(self, duration_minutes: int) -> Tuple[pd.DataFrame, Dict[str, Any]]`
- `run_batch(self, duration_minutes: int) -> Tuple[pd.DataFrame, Dict[str, Any]]`
- `export_audit_trail(self, simulation_results_df: pd.DataFrame, output_dir: str) -> Dict[str, str]`
- `run_live(self, duration_minutes: int) -> Generator[Dict[str, Any], None, None]`
- `save_state(self) -> Dict[str, Any]`
- `load_state(self, state: Dict[str, Any]) -> None`

## `iints.core.supervisor`

- Source: `src/iints/core/supervisor.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SafetyLevel` | `SafetyLevel(Enum)` | No module docstring. |
| `SafetyViolation` | `SafetyViolation` | No module docstring. |
| `SafetyDecision` | `SafetyDecision` | No module docstring. |
| `IndependentSupervisor` | `IndependentSupervisor` | Independent safety supervisor that operates separately from algorithms. Implements hard safety limits and emergency overrides. |

#### `IndependentSupervisor` methods

- `evaluate_safety(self, current_glucose: float, proposed_insulin: float, current_time: float, current_iob: float = 0.0, predicted_glucose_30min: Optional[float] = None, basal_insulin_units: Optional[float] = None, basal_limit_units: Optional[float] = None) -> Dict[str, Any]`
- `get_safety_report(self) -> Dict[str, Any]`
- `reset(self) -> None`
- `get_state(self) -> Dict[str, Any]`
- `set_state(self, state: Dict[str, Any]) -> None`

## `iints.data`

- Source: `src/iints/data/__init__.py`
- Summary: IINTS-AF Data Module Universal data ingestion and quality validation.
- Explicit exports: `DataAdapter, ColumnMapper, ColumnMapping, ImportResult, export_demo_csv, export_standard_csv, guess_column_mapping, import_carelink_csv, import_carelink_timeline, import_cgm_csv, import_cgm_dataframe, load_carelink_event_log, load_demo_dataframe, scenario_from_csv, scenario_from_dataframe, summarize_carelink_csv, DataQualityChecker, QualityReport, DataGap, DataAnomaly, REALISM_VERDICT_ORDER, MealResponse, RealismCheck, RealismReport, realism_verdict_meets_minimum, validate_realism_csv, validate_realism_dataset, write_realism_report, RealDataGateProfile, RealDataGateResult, STRICT_REAL_DATA_RESEARCH_PROFILE, review_real_data_realism, rank_real_data_sources, ReferenceBand, ReferenceComparison, RealismReferenceProfile, get_realism_reference, list_realism_reference_ids, load_realism_reference_registry, build_realism_dashboard_html, write_realism_dashboard, certify_csv, certify_dataset, render_certification_dashboard, write_certification_dashboard, write_certification_report, AVAILABLE_STUDY_CORRUPTIONS, apply_study_corruptions, write_corrupted_study_csv, UniversalParser, StandardDataPack, ParseResult, load_dataset_registry, get_dataset, list_dataset_ids, fetch_dataset, DEFAULT_RESEARCH_DATASET_IDS, build_research_dataset_matrix, resolve_research_dataset_entries, write_research_dataset_plan, NightscoutConfig, import_nightscout, TidepoolClient, TidepoolConfig, fetch_tidepool_dataframe, import_tidepool, load_openapi_spec, MedtronicLiveClient, MedtronicLiveConfig, fetch_medtronic_live_dataframe, fetch_medtronic_live_timeline, import_medtronic_live, normalize_medtronic_live_payload, StreamSpec, FeatureSpec, LabelSpec, ValidationSpec, ProcessSpec, ModelReadyContract, compile_contract, parse_contract, load_contract_yaml, ContractRunner, ValidationResult, CheckResult, MDMP_PROTOCOL_VERSION, MDMP_GRADE_ORDER, classify_mdmp_grade, mdmp_grade_meets_minimum, dataframe_fingerprint, build_mdmp_dashboard_html, mdmp_gate, MDMPGateError, generate_synthetic_mirror, SyntheticMirrorArtifact`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.data.adapter`

- Source: `src/iints/data/adapter.py`
- Summary: IINTS-AF Universal Data Adapter Professional data import layer with schema validation

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `DataAdapter` | `DataAdapter` | Universal data adapter for IINTS-AF framework |

#### `DataAdapter` methods

- `load_data_pack(self, pack_name: str) -> Dict`
- `load_ohio_dataset(self, patient_id: str) -> pd.DataFrame`
- `get_available_ohio_patients(self) -> List[str]`
- `clinical_benchmark_comparison(self, patient_id: str, algorithms: List[str]) -> Dict[str, Any]`

### Public Functions

- `main()`

## `iints.data.certify`

- Source: `src/iints/data/certify.py`
- Summary: No module docstring.

### Public Functions

- `certify_dataset(contract: Any | str | Path, dataframe: pd.DataFrame, *, apply_builtin_transforms: bool = True) -> MDMPValidationResult`
- `certify_csv(contract_path: str | Path, input_csv: str | Path, *, apply_builtin_transforms: bool = True) -> MDMPValidationResult`
- `render_certification_dashboard(report: Mapping[str, Any] | MDMPValidationResult, *, title: str = 'IINTS Data Certification Dashboard') -> str`
- `write_certification_report(report: Mapping[str, Any] | MDMPValidationResult, output_path: str | Path) -> Path`
- `write_certification_dashboard(report: Mapping[str, Any] | MDMPValidationResult, output_path: str | Path, *, title: str = 'IINTS Data Certification Dashboard') -> Path`

## `iints.data.column_mapper`

- Source: `src/iints/data/column_mapper.py`
- Summary: Column Mapper - IINTS-AF Maps various column names to standard IINTS format

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ColumnMapping` | `ColumnMapping` | Result of column mapping operation |
| `ColumnMapper` | `ColumnMapper` | Maps various column name aliases to standard IINTS format. |

#### `ColumnMapper` methods

- `detect_source(self, columns: List[str]) -> Optional[str]`
- `normalize_column_name(self, column_name: str) -> str`
- `find_standard_mapping(self, column_name: str) -> Optional[str]`
- `map_columns(self, columns: List[str]) -> ColumnMapping`
- `apply_mapping(self, df, mapping: ColumnMapping) -> 'pd.DataFrame'`
- `get_recommended_parser(self, source: str) -> str`
- `get_source_info(self, source: str) -> Dict`

### Public Functions

- `demo_column_mapping()`

## `iints.data.contracts`

- Source: `src/iints/data/contracts.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `StreamSpec` | `StreamSpec` | No module docstring. |
| `FeatureSpec` | `FeatureSpec` | No module docstring. |
| `LabelSpec` | `LabelSpec` | No module docstring. |
| `ValidationSpec` | `ValidationSpec` | No module docstring. |
| `ProcessSpec` | `ProcessSpec` | No module docstring. |
| `ModelReadyContract` | `ModelReadyContract` | No module docstring. |

#### `StreamSpec` methods

- `to_dict(self) -> Dict[str, Any]`

#### `FeatureSpec` methods

- `to_dict(self) -> Dict[str, Any]`

#### `LabelSpec` methods

- `to_dict(self) -> Dict[str, Any]`

#### `ValidationSpec` methods

- `to_dict(self) -> Dict[str, Any]`

#### `ProcessSpec` methods

- `to_dict(self) -> Dict[str, Any]`

#### `ModelReadyContract` methods

- `to_dict(self) -> Dict[str, Any]`
- `fingerprint(self) -> str`

### Public Functions

- `canonicalize_contract(payload: Dict[str, Any]) -> str`
- `compile_contract(payload: Dict[str, Any]) -> Dict[str, Any]`
- `parse_contract(payload: Dict[str, Any]) -> ModelReadyContract`
- `load_contract_yaml(path: Path) -> ModelReadyContract`

## `iints.data.demo`

- Source: `src/iints/data/demo/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.data.evidence`

- Source: `src/iints/data/evidence.py`
- Summary: No module docstring.

### Public Functions

- `rank_real_data_sources() -> List[Dict[str, Any]]`

### Public Constants

- `CONTROLLED_ACCESS_MARKERS`
- `OPEN_ACCESS_MARKERS`

## `iints.data.guardians`

- Source: `src/iints/data/guardians.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `MDMPGateError` | `MDMPGateError(RuntimeError)` | No module docstring. |

### Public Functions

- `mdmp_gate(contract: ContractInput, *, min_grade: str = 'research_grade', fail_mode: GateFailMode = 'raise', dataframe_arg: Optional[str] = None, apply_builtin_transforms: bool = True, transform_hooks: Optional[Iterable[Callable[[pd.DataFrame], pd.DataFrame]]] = None, on_result: Optional[Callable[[ValidationResult], None]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]`

### Public Constants

- `LOGGER`

## `iints.data.importer`

- Source: `src/iints/data/importer.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ImportResult` | `ImportResult` | No module docstring. |

### Public Functions

- `summarize_carelink_csv(path: Union[str, Path]) -> Dict[str, Any]`
- `load_carelink_event_log(path: Union[str, Path]) -> tuple[pd.DataFrame, Dict[str, Any]]`
- `import_carelink_timeline(path: Union[str, Path], *, source: Optional[str] = None, event_tolerance_minutes: float = 7.5) -> pd.DataFrame`
- `import_carelink_csv(path: Union[str, Path], *, source: Optional[str] = None, event_tolerance_minutes: float = 7.5) -> pd.DataFrame`
- `guess_column_mapping(columns: Iterable[str], data_format: str = 'generic') -> Dict[str, Optional[str]]`
- `validate_import_schema(columns: Iterable[str], data_format: str, column_map: Optional[Dict[str, str]] = None) -> None`
- `import_cgm_dataframe(df: pd.DataFrame, data_format: str = 'generic', column_map: Optional[Dict[str, str]] = None, time_unit: str = 'minutes', source: Optional[str] = None) -> pd.DataFrame`
- `import_cgm_csv(path: Union[str, Path], data_format: str = 'generic', column_map: Optional[Dict[str, str]] = None, time_unit: str = 'minutes', source: Optional[str] = None) -> pd.DataFrame`
- `scenario_from_dataframe(df: pd.DataFrame, scenario_name: str, scenario_version: str = '1.0', description: str = 'Imported CGM scenario', carb_threshold: float = 0.1, absorption_delay_minutes: int = 10, duration_minutes: int = 60) -> Dict[str, Any]`
- `scenario_from_csv(path: Union[str, Path], scenario_name: str = 'Imported CGM Scenario', scenario_version: str = '1.0', data_format: str = 'generic', column_map: Optional[Dict[str, str]] = None, time_unit: str = 'minutes', carb_threshold: float = 0.1) -> ImportResult`
- `export_standard_csv(df: pd.DataFrame, output_path: Union[str, Path]) -> str`
- `load_demo_dataframe() -> pd.DataFrame`
- `export_demo_csv(output_path: Union[str, Path]) -> str`

### Public Constants

- `DEFAULT_MAPPINGS`
- `IMPORT_FORMAT_SCHEMAS`

## `iints.data.ingestor`

- Source: `src/iints/data/ingestor.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `DataIngestor` | `DataIngestor` | Standardized Data Bridge for ingesting various diabetes datasets into a universal IINTS-AF format. |

#### `DataIngestor` methods

- `get_patient_model(self, file_path: Union[str, Path], data_type: str) -> pd.DataFrame`

## `iints.data.mdmp_visualizer`

- Source: `src/iints/data/mdmp_visualizer.py`
- Summary: No module docstring.

### Public Functions

- `build_mdmp_dashboard_html(report: Dict[str, Any], *, title: str = 'IINTS MDMP Certification Dashboard', generated_at_utc: Optional[str] = None) -> str`

## `iints.data.medtronic_live`

- Source: `src/iints/data/medtronic_live.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `MedtronicLiveConfig` | `MedtronicLiveConfig` | Configuration for an authorized read-only Medtronic/CareLink live relay. |
| `MedtronicLiveClient` | `MedtronicLiveClient` | Small authenticated JSON client for a read-only Medtronic live data relay. |

#### `MedtronicLiveClient` methods

- `get_json(self, path: str, *, query: Optional[Dict[str, Any]] = None) -> Any`

### Public Functions

- `normalize_medtronic_live_payload(payload: Any, *, event_tolerance_minutes: float = 7.5, source: str = 'medtronic_carelink_live') -> pd.DataFrame`
- `fetch_medtronic_live_payload(config: MedtronicLiveConfig) -> Any`
- `fetch_medtronic_live_timeline(config: MedtronicLiveConfig) -> pd.DataFrame`
- `medtronic_live_timeline_to_standard(timeline: pd.DataFrame, *, source: str = 'medtronic_carelink_live') -> pd.DataFrame`
- `fetch_medtronic_live_dataframe(config: MedtronicLiveConfig) -> pd.DataFrame`
- `import_medtronic_live(config: MedtronicLiveConfig, scenario_name: str = 'Medtronic CareLink Live Import', scenario_version: str = '1.0', carb_threshold: float = 0.1) -> ImportResult`
- `poll_medtronic_live_timeline(config: MedtronicLiveConfig, *, samples: int = 1, poll_seconds: float = 30.0) -> Iterable[pd.DataFrame]`
- `write_latest_medtronic_live_snapshot(timeline: pd.DataFrame, output_dir: Path) -> Dict[str, str]`

### Public Constants

- `CARB_KEYS`
- `GLUCOSE_KEYS`
- `GLUCOSE_TYPES`
- `INSULIN_KEYS`
- `LIST_KEYS`
- `SINGLE_RECORD_KEYS`
- `TIMESTAMP_KEYS`
- `TYPE_KEYS`
- `UNIT_KEYS`

## `iints.data.nightscout`

- Source: `src/iints/data/nightscout.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `NightscoutConfig` | `NightscoutConfig` | No module docstring. |

### Public Functions

- `fetch_nightscout_dataframe(config: NightscoutConfig) -> pd.DataFrame`
- `import_nightscout(config: NightscoutConfig, scenario_name: str = 'Nightscout Import', scenario_version: str = '1.0', carb_threshold: float = 0.1) -> ImportResult`

## `iints.data.quality_checker`

- Source: `src/iints/data/quality_checker.py`
- Summary: Data Quality Checker - IINTS-AF Validates data quality and calculates confidence scores with gap detection.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `QualityReport` | `QualityReport` | Comprehensive data quality report |
| `DataGap` | `DataGap` | Represents a gap in the data |
| `DataAnomaly` | `DataAnomaly` | Represents an anomalous data point |
| `DataQualityChecker` | `DataQualityChecker` | Validates data quality and calculates confidence scores. |

#### `QualityReport` methods

- `to_dict(self) -> Dict`

#### `DataGap` methods

- `to_dict(self) -> Dict`
- `get_warning_message(self) -> str`

#### `DataAnomaly` methods

- `to_dict(self) -> Dict`

#### `DataQualityChecker` methods

- `check_completeness(self, df: pd.DataFrame) -> Tuple[float, List[DataGap]]`
- `check_consistency(self, df: pd.DataFrame) -> float`
- `check_validity(self, df: pd.DataFrame) -> Tuple[float, List[DataAnomaly]]`
- `check(self, df: pd.DataFrame) -> QualityReport`
- `get_confidence_score(self, df: pd.DataFrame) -> float`
- `print_report(self, report: QualityReport)`

### Public Functions

- `demo_quality_checker()`

## `iints.data.realism_dashboard`

- Source: `src/iints/data/realism_dashboard.py`
- Summary: No module docstring.

### Public Functions

- `build_realism_dashboard_html(report: RealismReport, dataframe: pd.DataFrame, *, title: str = 'IINTS Physiological Realism Dashboard', source_label: Optional[str] = None) -> str`
- `write_realism_dashboard(report: RealismReport, dataframe: pd.DataFrame, output_path: str | Path, *, title: str = 'IINTS Physiological Realism Dashboard', source_label: Optional[str] = None) -> Path`

## `iints.data.realism_governance`

- Source: `src/iints/data/realism_governance.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `RealDataGateProfile` | `RealDataGateProfile` | Strict research gate for traces used as realism evidence or AI training data. |
| `RealDataGateResult` | `RealDataGateResult` | No module docstring. |

#### `RealDataGateResult` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `review_real_data_realism(report: RealismReport, *, profile: RealDataGateProfile = STRICT_REAL_DATA_RESEARCH_PROFILE) -> RealDataGateResult`

### Public Constants

- `STRICT_REAL_DATA_RESEARCH_PROFILE`

## `iints.data.realism_reference`

- Source: `src/iints/data/realism_reference.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ReferenceBand` | `ReferenceBand` | No module docstring. |
| `ReferenceComparison` | `ReferenceComparison` | No module docstring. |
| `RealismReferenceProfile` | `RealismReferenceProfile` | No module docstring. |

#### `ReferenceBand` methods

- `from_dict(cls, payload: Dict[str, Any]) -> 'ReferenceBand'`
- `to_dict(self) -> Dict[str, float]`

#### `ReferenceComparison` methods

- `to_dict(self) -> Dict[str, Any]`

#### `RealismReferenceProfile` methods

- `from_dict(cls, payload: Dict[str, Any]) -> 'RealismReferenceProfile'`
- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `load_realism_reference_registry() -> List[RealismReferenceProfile]`
- `list_realism_reference_ids() -> List[str]`
- `get_realism_reference(reference_id: str) -> RealismReferenceProfile`
- `compare_to_reference_band(metric_key: str, label: str, observed_value: float | None, band: ReferenceBand) -> ReferenceComparison`
- `metric_label(metric_key: str) -> str`
- `build_reference_comparisons(observed_metrics: Dict[str, Any], profile: RealismReferenceProfile) -> List[ReferenceComparison]`

## `iints.data.realism_validator`

- Source: `src/iints/data/realism_validator.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `RealismCheck` | `RealismCheck` | No module docstring. |
| `MealResponse` | `MealResponse` | No module docstring. |
| `RealismReport` | `RealismReport` | No module docstring. |

#### `RealismCheck` methods

- `to_dict(self) -> Dict[str, Any]`

#### `MealResponse` methods

- `to_dict(self) -> Dict[str, float | None]`

#### `RealismReport` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `realism_verdict_meets_minimum(verdict: str, minimum: str) -> bool`
- `validate_realism_dataset(dataframe: pd.DataFrame, *, expected_interval_minutes: int = 5, min_meal_grams: float = 10.0, reference: str | RealismReferenceProfile | None = None) -> RealismReport`
- `validate_realism_csv(input_csv: str | Path, *, data_format: str = 'generic', column_map: Optional[Dict[str, str]] = None, time_unit: str = 'minutes', source: Optional[str] = None, expected_interval_minutes: int = 5, min_meal_grams: float = 10.0, reference: str | RealismReferenceProfile | None = None) -> RealismReport`
- `write_realism_report(report: RealismReport, output_path: str | Path) -> Path`

### Public Constants

- `REALISM_VERDICT_ORDER`

## `iints.data.registry`

- Source: `src/iints/data/registry.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `DatasetRegistryError` | `DatasetRegistryError(RuntimeError)` | No module docstring. |
| `DatasetFetchError` | `DatasetFetchError(RuntimeError)` | No module docstring. |

### Public Functions

- `load_dataset_registry() -> List[Dict[str, Any]]`
- `get_dataset(dataset_id: str) -> Dict[str, Any]`
- `list_dataset_ids() -> List[str]`
- `fetch_dataset(dataset_id: str, output_dir: Path, extract: bool = True, verify: bool = True) -> List[Path]`

## `iints.data.research_catalog`

- Source: `src/iints/data/research_catalog.py`
- Summary: No module docstring.

### Public Functions

- `resolve_research_dataset_entries(dataset_ids: Sequence[str] | None = None) -> list[Dict[str, Any]]`
- `build_research_dataset_matrix(dataset_ids: Sequence[str] | None = None) -> list[dict[str, Any]]`
- `write_research_dataset_plan(output_dir: Path, dataset_ids: Sequence[str] | None = None) -> dict[str, Any]`

### Public Constants

- `DEFAULT_RESEARCH_DATASET_IDS`
- `TASK_COLUMNS`

## `iints.data.runner`

- Source: `src/iints/data/runner.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `CheckResult` | `CheckResult` | No module docstring. |
| `ValidationResult` | `ValidationResult` | No module docstring. |
| `ContractRunner` | `ContractRunner` | Lightweight executor for model-ready data contracts. |

#### `CheckResult` methods

- `to_dict(self) -> Dict[str, Any]`

#### `ValidationResult` methods

- `to_dict(self) -> Dict[str, Any]`

#### `ContractRunner` methods

- `run(self, df: pd.DataFrame, *, transform_hooks: Optional[Iterable[TransformHook]] = None, apply_builtin_transforms: bool = True) -> ValidationResult`

### Public Functions

- `dataframe_fingerprint(df: pd.DataFrame) -> str`
- `classify_mdmp_grade(score: float, is_compliant: bool) -> str`
- `mdmp_grade_meets_minimum(actual_grade: str, minimum_grade: str) -> bool`

### Public Constants

- `MDMP_GRADE_ORDER`
- `MDMP_PROTOCOL_VERSION`

## `iints.data.study_corruption`

- Source: `src/iints/data/study_corruption.py`
- Summary: No module docstring.

### Public Functions

- `apply_study_corruptions(dataframe: pd.DataFrame, *, modes: list[str], seed: int = 42, timestamp_shift_minutes: int = 60, missing_fraction: float = 0.1, duplicate_fraction: float = 0.05, spike_fraction: float = 0.03, spike_magnitude_mgdl: float = 60.0) -> tuple[pd.DataFrame, dict[str, Any]]`
- `write_corrupted_study_csv(input_csv: str | Path, *, output_csv: str | Path, modes: list[str], manifest_output: str | Path | None = None, seed: int = 42, timestamp_shift_minutes: int = 60, missing_fraction: float = 0.1, duplicate_fraction: float = 0.05, spike_fraction: float = 0.03, spike_magnitude_mgdl: float = 60.0) -> dict[str, str]`

### Public Constants

- `AVAILABLE_STUDY_CORRUPTIONS`

## `iints.data.synthetic_mirror`

- Source: `src/iints/data/synthetic_mirror.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SyntheticMirrorArtifact` | `SyntheticMirrorArtifact` | No module docstring. |

#### `SyntheticMirrorArtifact` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `generate_synthetic_mirror(source_df: pd.DataFrame, contract: ContractInput, *, rows: Optional[int] = None, seed: int = 42, noise_scale: float = 0.05, timestamp_column: str = 'timestamp') -> Tuple[pd.DataFrame, SyntheticMirrorArtifact]`

## `iints.data.tidepool`

- Source: `src/iints/data/tidepool.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `TidepoolConfig` | `TidepoolConfig` | No module docstring. |
| `TidepoolClient` | `TidepoolClient` | Small authenticated Tidepool Data Platform client for read-only imports. |

#### `TidepoolClient` methods

- `get_json(self, path: str, *, query: Optional[dict[str, Any]] = None) -> Any`
- `current_user(self) -> dict[str, Any]`
- `fetch_device_data(self, user_id: str, *, start: Optional[str] = None, end: Optional[str] = None, types: Iterable[str] = ('cbg', 'bolus', 'wizard', 'food')) -> list[dict[str, Any]]`

### Public Functions

- `fetch_tidepool_dataframe(config: TidepoolConfig) -> pd.DataFrame`
- `import_tidepool(config: TidepoolConfig, scenario_name: str = 'Tidepool Import', scenario_version: str = '1.0', carb_threshold: float = 0.1) -> ImportResult`
- `load_openapi_spec(path: str) -> Dict[str, Any]`

### Public Constants

- `MMOL_L_TO_MG_DL`

## `iints.data.universal_parser`

- Source: `src/iints/data/universal_parser.py`
- Summary: Universal Data Parser - IINTS-AF Universal ingestion engine for any CSV/JSON data format.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `StandardDataPack` | `StandardDataPack` | Standard data format for IINTS-AF. |
| `ParseResult` | `ParseResult` | Result of a parse operation |
| `UniversalParser` | `UniversalParser` | Universal data parser for IINTS-AF. |

#### `StandardDataPack` methods

- `duration_hours(self) -> float`
- `data_points(self) -> int`
- `confidence_score(self) -> float`

#### `ParseResult` methods

- `to_dict(self) -> Dict`

#### `UniversalParser` methods

- `detect_format(self, file_path: str) -> str`
- `detect_delimiter(self, file_path: str) -> Optional[str]`
- `parse_datetime(self, value: Any) -> Optional[float]`
- `parse_glucose(self, value: Any, unit: str = 'mg/dL') -> Optional[float]`
- `parse_csv(self, file_path: str) -> pd.DataFrame`
- `parse_json(self, file_path: str) -> pd.DataFrame`
- `convert_to_standard(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]`
- `normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame`
- `validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]`
- `parse(self, file_path: str, validate: Optional[bool] = None, metadata: Optional[Dict] = None) -> ParseResult`
- `parse_string(self, content: str, format_type: str = 'csv', validate: bool = True) -> ParseResult`

### Public Functions

- `demo_universal_parser()`

## `iints.demo_assets`

- Source: `src/iints/demo_assets.py`
- Summary: No module docstring.

### Public Functions

- `export_live_stage_demo(output_dir: str | Path = '.', *, overwrite: bool = False) -> dict[str, str]`

## `iints.emulation`

- Source: `src/iints/emulation/__init__.py`
- Summary: No module docstring.
- Explicit exports: `LegacyEmulator, PumpBehavior, PIDParameters, SafetyLimits, SafetyLevel, EmulatorDecision, Medtronic780GEmulator, Medtronic780GBehavior, TandemControlIQEmulator, TandemControlIQBehavior, Omnipod5Emulator, Omnipod5Behavior`

### Public Functions

- `get_emulator(pump_type: str) -> InsulinAlgorithm`
- `list_available_emulators() -> list`

### Public Constants

- `EMULATORS`

## `iints.emulation.legacy_base`

- Source: `src/iints/emulation/legacy_base.py`
- Summary: Legacy Emulator Base Class - IINTS-AF Base class for commercial insulin pump emulation.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SafetyLevel` | `SafetyLevel(Enum)` | Safety level classification for pump behaviors |
| `SafetyLimits` | `SafetyLimits` | Pump-specific safety constraints |
| `PIDParameters` | `PIDParameters` | PID controller parameters |
| `PumpBehavior` | `PumpBehavior` | Complete pump behavior profile |
| `EmulatorDecision` | `EmulatorDecision` | Decision output from a pump emulator |
| `LegacyEmulator` | `LegacyEmulator(InsulinAlgorithm)` | Abstract base class for commercial insulin pump emulation. |

#### `PumpBehavior` methods

- `to_dict(self) -> Dict`

#### `EmulatorDecision` methods

- `to_dict(self) -> Dict`

#### `LegacyEmulator` methods

- `get_sources(self) -> List[Dict[str, str]]`
- `reset(self)`
- `set_safety_mode(self, level: SafetyLevel)`
- `get_behavior_profile(self) -> PumpBehavior`
- `emulate_decision(self, glucose: float, velocity: float, insulin_on_board: float, carbs: float, current_time: float = 0) -> EmulatorDecision`
- `predict_insulin(self, algo_input: AlgorithmInput) -> Dict[str, Any]`
- `get_decision_history(self) -> List[EmulatorDecision]`
- `export_behavior_report(self) -> Dict`
- `compare_with_new_ai(self, new_ai_decisions: List[Dict]) -> Dict`

### Public Functions

- `demo_legacy_emulator()`

## `iints.emulation.medtronic_780g`

- Source: `src/iints/emulation/medtronic_780g.py`
- Summary: Medtronic 780G Emulator - IINTS-AF Emulates the Medtronic MiniMed 780G with SmartGuard algorithm.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `Medtronic780GBehavior` | `Medtronic780GBehavior(PumpBehavior)` | Medtronic 780G specific behavior profile |
| `Medtronic780GEmulator` | `Medtronic780GEmulator(LegacyEmulator)` | Emulates Medtronic 780G SmartGuard algorithm. |

#### `Medtronic780GEmulator` methods

- `get_sources(self) -> List[Dict[str, str]]`
- `emulate_decision(self, glucose: float, velocity: float, insulin_on_board: float, carbs: float, current_time: float = 0) -> EmulatorDecision`
- `get_algorithm_personality(self) -> Dict`

### Public Functions

- `demo_medtronic_780g()`

## `iints.emulation.omnipod_5`

- Source: `src/iints/emulation/omnipod_5.py`
- Summary: Omnipod 5 Emulator - IINTS-AF Emulates the Omnipod 5 with Horizon Algorithm.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `Omnipod5Behavior` | `Omnipod5Behavior(PumpBehavior)` | Omnipod 5 specific behavior profile |
| `Omnipod5Emulator` | `Omnipod5Emulator(LegacyEmulator)` | Emulates Omnipod 5 with Horizon algorithm. |

#### `Omnipod5Emulator` methods

- `get_sources(self) -> List[Dict[str, str]]`
- `set_activity_mode(self, enabled: bool, mode_type: str = 'exercise')`
- `emulate_decision(self, glucose: float, velocity: float, insulin_on_board: float, carbs: float, current_time: float = 0) -> EmulatorDecision`
- `get_algorithm_personality(self) -> Dict`

### Public Functions

- `demo_omnipod5()`

## `iints.emulation.tandem_controliq`

- Source: `src/iints/emulation/tandem_controliq.py`
- Summary: Tandem Control-IQ Emulator - IINTS-AF Emulates the Tandem t:slim X2 with Control-IQ algorithm.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `TandemControlIQBehavior` | `TandemControlIQBehavior(PumpBehavior)` | Tandem Control-IQ specific behavior profile |
| `TandemControlIQEmulator` | `TandemControlIQEmulator(LegacyEmulator)` | Emulates Tandem Control-IQ algorithm. |

#### `TandemControlIQEmulator` methods

- `get_sources(self) -> List[Dict[str, str]]`
- `set_exercise_mode(self, enabled: bool)`
- `emulate_decision(self, glucose: float, velocity: float, insulin_on_board: float, carbs: float, current_time: float = 0) -> EmulatorDecision`
- `get_algorithm_personality(self) -> Dict`

### Public Functions

- `demo_tandem_controliq()`

## `iints.highlevel`

- Source: `src/iints/highlevel.py`
- Summary: No module docstring.

### Public Functions

- `run_simulation(algorithm: Union[InsulinAlgorithm, type], scenario: Optional[Union[str, Path, Dict[str, Any]]] = None, patient_config: Union[str, Path, Dict[str, Any], PatientProfile] = 'default_patient', patient_model_type: str = 'auto', sensor_noise_std: Optional[float] = None, sensor_lag_minutes: Optional[int] = None, sensor_dropout_prob: Optional[float] = None, sensor_bias: Optional[float] = None, sensor_profile: Optional[str] = None, duration_minutes: int = 720, time_step: int = 5, seed: Optional[int] = None, output_dir: Optional[Union[str, Path]] = None, compare_baselines: bool = True, export_audit: bool = True, generate_report: bool = True, safety_config: Optional[SafetyConfig] = None, predictor: Optional[object] = None, physiology_variation_profile: Optional[str] = None, physiology_variation_scale: float = 1.0) -> Dict[str, Any]`
- `run_full(algorithm: Union[InsulinAlgorithm, type], scenario: Optional[Union[str, Path, Dict[str, Any]]] = None, patient_config: Union[str, Path, Dict[str, Any], PatientProfile] = 'default_patient', patient_model_type: str = 'auto', sensor_noise_std: Optional[float] = None, sensor_lag_minutes: Optional[int] = None, sensor_dropout_prob: Optional[float] = None, sensor_bias: Optional[float] = None, sensor_profile: Optional[str] = None, duration_minutes: int = 720, time_step: int = 5, seed: Optional[int] = None, output_dir: Optional[Union[str, Path]] = None, enable_profiling: bool = True, safety_config: Optional[SafetyConfig] = None, predictor: Optional[object] = None) -> Dict[str, Any]`
- `run_population(algo_path: Optional[Union[str, Path]] = None, algo_class_name: Optional[str] = None, n_patients: int = 100, scenario: Optional[Union[str, Path, Dict[str, Any]]] = None, patient_config: Union[str, Path, Dict[str, Any], PatientProfile] = 'default_patient', duration_minutes: int = 720, time_step: int = 5, seed: Optional[int] = None, output_dir: Optional[Union[str, Path]] = None, max_workers: Optional[int] = None, safety_config: Optional[SafetyConfig] = None, safety_weights: Optional[Dict[str, float]] = None, patient_model_type: str = 'auto', population_cv: Optional[Dict[str, float]] = None) -> Dict[str, Any]`

## `iints.jetson`

- Source: `src/iints/jetson/__init__.py`
- Summary: No module docstring.
- Explicit exports: `ENDURANCE_PROFILES, EnduranceConfig, JetsonEnduranceError, build_endurance_service_file, collect_jetson_hardware_info, export_endurance_archive, load_endurance_status, parse_duration_to_minutes, run_endurance_study, stop_endurance_study`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.jetson.endurance`

- Source: `src/iints/jetson/endurance.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `JetsonEnduranceError` | `JetsonEnduranceError(RuntimeError)` | No module docstring. |
| `EnduranceConfig` | `EnduranceConfig` | No module docstring. |

#### `EnduranceConfig` methods

- `expected_steps(self) -> int`
- `simulator_end_minutes(self) -> int`
- `wall_clock_target_seconds(self) -> int`

### Public Functions

- `utc_now_iso() -> str`
- `parse_duration_to_minutes(value: str) -> int`
- `collect_jetson_hardware_info() -> Dict[str, Any]`
- `run_endurance_study(*, algorithm: InsulinAlgorithm, predictor: Optional[object], config: EnduranceConfig, progress_callback: Optional[Any] = None, monotonic_fn: Any = time.monotonic, sleep_fn: Any = time.sleep) -> Dict[str, Any]`
- `load_endurance_status(output_dir: str | Path) -> Dict[str, Any]`
- `stop_endurance_study(output_dir: str | Path, *, generate_report: bool = False) -> Dict[str, Any]`
- `export_endurance_archive(output_dir: str | Path, output: str | Path) -> Path`
- `build_endurance_service_file(*, algo: str, duration: str, output_dir: str, predictor: Optional[str] = None, profile: str = 'mixed_adversarial', seed: Optional[int] = None, wall_clock: bool = False, working_directory: Optional[str] = None) -> str`

### Public Constants

- `ENDURANCE_EXECUTION_MODES`
- `ENDURANCE_PROFILES`

## `iints.jetson.research_pipeline`

- Source: `src/iints/jetson/research_pipeline.py`
- Summary: No module docstring.

### Public Functions

- `finalize_endurance_research(output_dir: Path, *, repo_root: Path, train_predictor: bool = True, predictor_config_path: Path | None = None, train_neural: bool = True, evaluation_presets: Iterable[str] = DEFAULT_HELD_OUT_PRESETS, evaluation_seeds: Iterable[int] = (101, 202, 303), evaluation_duration_minutes: int = 1440) -> Dict[str, Any]`

## `iints.learning`

- Source: `src/iints/learning/__init__.py`
- Summary: No module docstring.
- Explicit exports: `AutonomousLearningSystem, ClinicalTeacher, ClinicalConstraints`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.learning.autonomous_optimizer`

- Source: `src/iints/learning/autonomous_optimizer.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ClinicalConstraints` | `ClinicalConstraints` | Physiological constraints based on medical literature. |

## `iints.learning.learning_system`

- Source: `src/iints/learning/learning_system.py`
- Summary: IINTS-AF Learning System Implements real learning with parameter adaptation and validation

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `LearningSystem` | `LearningSystem` | Real learning system that adapts to patient-specific patterns |

#### `LearningSystem` methods

- `save_learned_parameters(self, patient_id: str, parameters: Dict, performance_metrics: Dict)`
- `load_learned_parameters(self, patient_id: str) -> Optional[Dict]`
- `simulate_learning_process(self, patient_id: str, glucose_data: List[float]) -> Tuple[Dict, List[float]]`
- `validate_learning_safety(self, parameters: Dict, patient_id: str) -> Tuple[bool, str]`
- `get_learning_status(self, patient_id: str) -> str`

## `iints.live_patient`

- Source: `src/iints/live_patient/__init__.py`
- Summary: No module docstring.
- Explicit exports: `create_edge_bundle, export_edge_setup, summarize_edge_workspace, write_edge_update_script, render_edge_long_study_config_template, load_edge_long_study_config, run_edge_long_study, create_edge_study_snapshot, export_edge_study_archive, create_patient_app, run_edge_benchmark, export_uno_q_bridge, DIRECT_PUMP_READ_ONLY_CONFIRMATION, DirectPumpConfig, PumpSnapshot, SimulatedMedtronicPumpTransport, stream_direct_pump_snapshots, write_direct_pump_snapshot, bench_test_pico_pump_bundle, build_pico_pump_bundle, create_pico_pump_lab, LivePatientDaemon, PatientRuntimeConfig, PatientRuntimeStore, get_runtime_scenario_profile, list_runtime_scenario_profiles, load_runtime_status, is_process_alive`

### Public Functions

- `run_edge_benchmark(*args: Any, **kwargs: Any) -> Any`

## `iints.live_patient.api`

- Source: `src/iints/live_patient/api.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `MealRequest` | `MealRequest(BaseModel)` | No module docstring. |
| `ScenarioResetRequest` | `ScenarioResetRequest(BaseModel)` | No module docstring. |

### Public Functions

- `create_patient_app(workspace: str | Path, api_token: str | None = None) -> FastAPI`

### Public Constants

- `CONTROL_HEADER_NAME`
- `CONTROL_HEADER_VALUE`
- `SECURITY_RESPONSE_HEADERS`

## `iints.live_patient.daemon`

- Source: `src/iints/live_patient/daemon.py`
- Summary: No module docstring.

### Public Functions

- `main() -> None`

## `iints.live_patient.edge_benchmark`

- Source: `src/iints/live_patient/edge_benchmark.py`
- Summary: No module docstring.

### Public Functions

- `run_edge_benchmark(*, algo_path: Path, patient_config: str = 'default_patient', patient_model_type: str = 'auto', scenario_profile: str = 'normal_day', steps: int = 72, platform_name: str = 'auto', api_host: str = '127.0.0.1', api_port: int = 8766, seed: int | None = None) -> dict[str, Any]`

## `iints.live_patient.edge_ops`

- Source: `src/iints/live_patient/edge_ops.py`
- Summary: No module docstring.

### Public Functions

- `summarize_edge_workspace(workspace: str | Path) -> dict[str, Any]`
- `create_edge_bundle(workspace: str | Path, *, output_path: str | Path, include_log: bool = True, include_database: bool = True) -> dict[str, Any]`
- `render_edge_update_script(*, profile: str = 'edge', version_pin: str | None = None) -> str`
- `write_edge_update_script(output_path: str | Path, *, profile: str = 'edge', version_pin: str | None = None) -> Path`
- `export_edge_setup(output_dir: str | Path, *, board: str = 'raspberry_pi', workspace_name: str = 'patient_runtime', scenario_profile: str = 'normal_day', patient_config: str = 'default_patient', patient_model_type: str = 'auto', mode: str = 'demo-time', speed: float = 60.0, api_host: str = '127.0.0.1', api_port: int = 8765, seed: int | None = None, service_name: str = 'iints-digital-patient', user_name: str | None = None, include_uno_bridge: bool = False, uno_bridge_port: str | None = None, uno_bridge_baudrate: int = UNO_Q_BRIDGE_BAUDRATE, uno_bridge_service_name: str = 'iints-uno-q-bridge') -> dict[str, str]`
- `deploy_edge_project(*, host: str, user_name: str | None = None, ssh_port: int = 22, remote_dir: str = '~/iints_pi_demo', local_output_dir: str | Path = 'iints_pi_demo', board: str = 'raspberry_pi', workspace_name: str = 'patient_runtime', scenario_profile: str = 'expo_hot_start', patient_config: str = 'default_patient', patient_model_type: str = 'auto', mode: str = 'demo-time', speed: float = 60.0, api_host: str = '127.0.0.1', api_port: int = 8765, seed: int | None = None, service_name: str = 'iints-digital-patient', include_uno_bridge: bool = False, uno_bridge_port: str | None = None, uno_bridge_baudrate: int = UNO_Q_BRIDGE_BAUDRATE, uno_bridge_service_name: str = 'iints-uno-q-bridge', install_autostart: bool = True, start_runtime: bool = True, enable_connect_linger: bool = True, flash_uno_bridge: bool = False, uno_fqbn: str | None = None, arduino_cli: str = 'arduino-cli', dry_run: bool = False, ssh_timeout_seconds: float = 300.0, ssh_retries: int = 1, progress_callback: Callable[[str], None] | None = None) -> dict[str, Any]`
- `build_edge_offline_bundle(output_path: str | Path, *, board: str = 'raspberry_pi', workspace_name: str = 'patient_runtime', scenario_profile: str = 'expo_hot_start', patient_config: str = 'default_patient', patient_model_type: str = 'auto', mode: str = 'demo-time', speed: float = 60.0, api_host: str = '127.0.0.1', api_port: int = 8765, seed: int | None = None, service_name: str = 'iints-digital-patient', user_name: str | None = None, include_uno_bridge: bool = False, progress_callback: Callable[[str], None] | None = None) -> dict[str, str]`
- `run_remote_edge_command(*, host: str, user_name: str | None = None, ssh_port: int = 22, remote_dir: str = '~/iints_pi_demo', action: str = 'status', scenario_profile: str | None = None, seed: int | None = None, timeout_seconds: float = 60.0, retries: int = 0) -> dict[str, str]`

## `iints.live_patient.long_study`

- Source: `src/iints/live_patient/long_study.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `EdgeLongStudyConfig` | `EdgeLongStudyConfig` | No module docstring. |
| `LongStudyAlgorithmSpec` | `LongStudyAlgorithmSpec` | No module docstring. |
| `LongStudySnapshotResult` | `LongStudySnapshotResult` | No module docstring. |
| `LongStudyConfigError` | `LongStudyConfigError(ValueError)` | No module docstring. |
| `EdgeLongStudyExecutionError` | `EdgeLongStudyExecutionError(RuntimeError)` | No module docstring. |

#### `EdgeLongStudyConfig` methods

- `weekday_for_day(self, day_index: int) -> str`
- `profile_for_day(self, day_index: int) -> str`
- `snapshot_every_days(self) -> int`
- `to_dict(self) -> dict[str, Any]`

#### `LongStudyAlgorithmSpec` methods

- `to_dict(self) -> dict[str, str]`

#### `LongStudySnapshotResult` methods

- `to_dict(self) -> dict[str, str]`

### Public Functions

- `render_edge_long_study_config_template(*, output_dir: str = '/media/pi/usb_ssd/results/long_study', scratch_dir: str = '/tmp/iints_edge_long_study', algorithms: list[str] | None = None) -> str`
- `load_edge_long_study_config(config_path: Path) -> EdgeLongStudyConfig`
- `build_long_study_day_scenario(*, profile: RuntimeScenarioProfile, day_number: int, weekday: str, sequence_seed: int) -> dict[str, Any]`
- `create_edge_study_snapshot(input_dir: str | Path, *, output: str | Path) -> LongStudySnapshotResult`
- `export_edge_study_archive(input_dir: str | Path, *, output: str | Path) -> dict[str, str]`
- `run_edge_long_study(*, config_path: str | Path, project_dir: str | Path = '.', resume: bool = False, progress_callback: Callable[[str], None] | None = None) -> dict[str, Any]`

### Public Constants

- `WEEKDAY_ORDER`

## `iints.live_patient.medtronic_direct`

- Source: `src/iints/live_patient/medtronic_direct.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PumpTransport` | `PumpTransport(Protocol)` | Read-only direct pump transport protocol. |
| `PumpSnapshot` | `PumpSnapshot` | No module docstring. |
| `DirectPumpConfig` | `DirectPumpConfig` | No module docstring. |
| `SimulatedMedtronicPumpTransport` | `SimulatedMedtronicPumpTransport` | Bench-only direct pump transport for SDK integration testing. |
| `OfficialModulePumpTransport` | `OfficialModulePumpTransport` | Wrapper for an approved internal Medtronic read-only transport factory. |

#### `PumpTransport` methods

- `connect(self) -> None`
- `read_snapshot(self) -> 'PumpSnapshot | Mapping[str, Any]'`
- `disconnect(self) -> None`

#### `PumpSnapshot` methods

- `from_mapping(cls, payload: Mapping[str, Any]) -> 'PumpSnapshot'`
- `validate(self) -> None`
- `to_json_dict(self) -> dict[str, Any]`

#### `SimulatedMedtronicPumpTransport` methods

- `connect(self) -> None`
- `disconnect(self) -> None`
- `read_snapshot(self) -> PumpSnapshot`

#### `OfficialModulePumpTransport` methods

- `connect(self) -> None`
- `read_snapshot(self) -> PumpSnapshot | Mapping[str, Any]`
- `disconnect(self) -> None`

### Public Functions

- `create_direct_pump_transport(config: DirectPumpConfig) -> PumpTransport`
- `stream_direct_pump_snapshots(config: DirectPumpConfig, *, samples: int = 1, poll_seconds: float = 30.0) -> Iterable[PumpSnapshot]`
- `snapshots_to_dataframes(snapshots: Iterable[PumpSnapshot]) -> tuple[pd.DataFrame, pd.DataFrame]`
- `write_direct_pump_snapshot(snapshots: Iterable[PumpSnapshot], output_dir: str | Path) -> dict[str, str]`

### Public Constants

- `AUTHORIZED_IDENTITY_MODE`
- `COMMAND_LIKE_KEYS`
- `DIRECT_PUMP_READ_ONLY_CONFIRMATION`
- `DIRECT_PUMP_SOURCE`
- `DISALLOWED_IDENTITY_MODES`
- `GLUCOSE_KEYS`
- `TIMESTAMP_KEYS`

## `iints.live_patient.pico_pump`

- Source: `src/iints/live_patient/pico_pump.py`
- Summary: No module docstring.

### Public Functions

- `export_pico_pump_firmware(output_dir: str | Path) -> dict[str, str]`
- `create_pico_pump_lab(output_dir: str | Path, *, algorithm_path: str | Path | None = None) -> dict[str, str]`
- `build_pico_pump_bundle(algorithm_path: str | Path, output_dir: str | Path, *, safety_contract_path: str | Path | None = None, label: str = 'pico_pump_bench') -> dict[str, Any]`
- `bench_test_pico_pump_bundle(bundle_dir: str | Path) -> dict[str, Any]`
- `upload_pico_pump_bundle(bundle_dir: str | Path, mount_dir: str | Path, *, bench_only_confirmation: str, write: bool = False) -> dict[str, Any]`
- `run_pico_pump_serial_self_test(port: str, *, baudrate: int = PICO_PUMP_BAUDRATE, timeout_seconds: float = 1.5) -> list[dict[str, str | None]]`

### Public Constants

- `BENCH_ALGORITHM_TEMPLATE`
- `DEFAULT_PICO_PUMP_SAFETY_CONTRACT`
- `PICO_PUMP_BAUDRATE`
- `PICO_PUMP_CONFIRMATION`
- `PICO_PUMP_DEVICE_ID`
- `PICO_PUMP_READY_BANNER`

## `iints.live_patient.runtime`

- Source: `src/iints/live_patient/runtime.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `DailyEventTemplate` | `DailyEventTemplate` | No module docstring. |
| `RuntimeScenarioProfile` | `RuntimeScenarioProfile` | No module docstring. |
| `PatientRuntimeConfig` | `PatientRuntimeConfig` | No module docstring. |
| `PatientRuntimeStore` | `PatientRuntimeStore` | No module docstring. |
| `DailySchedulePlanner` | `DailySchedulePlanner` | No module docstring. |
| `LivePatientDaemon` | `LivePatientDaemon` | No module docstring. |

#### `PatientRuntimeConfig` methods

- `workspace_path(self) -> Path`
- `algo_file(self) -> Path`
- `db_path(self) -> Path`
- `snapshot_path(self) -> Path`
- `pid_path(self) -> Path`
- `log_path(self) -> Path`
- `config_path(self) -> Path`
- `bundle_dir(self) -> Path`
- `dashboard_url(self) -> str`
- `api_url(self) -> str`
- `to_json(self) -> dict[str, Any]`
- `from_path(cls, path: Path) -> 'PatientRuntimeConfig'`

#### `PatientRuntimeStore` methods

- `update_status(self, **fields: Any) -> None`
- `read_status(self) -> dict[str, Any]`
- `append_reading(self, payload: dict[str, Any], *, event_summary: str = '') -> None`
- `get_recent_readings(self, limit: int = 288) -> list[dict[str, Any]]`
- `get_latest_reading(self) -> dict[str, Any] | None`
- `enqueue_command(self, command: str, payload: dict[str, Any] | None = None) -> int`
- `fetch_pending_commands(self) -> list[dict[str, Any]]`
- `complete_command(self, command_id: int, *, status: str, result: dict[str, Any] | None = None) -> None`
- `await_command(self, command_id: int, timeout_seconds: float = 5.0) -> dict[str, Any] | None`
- `clear_runtime_data(self) -> None`
- `build_audit_summary(self) -> dict[str, Any]`

#### `DailySchedulePlanner` methods

- `set_templates(self, templates: Iterable[DailyEventTemplate]) -> None`
- `reset(self) -> None`
- `schedule_for_time(self, simulator: Simulator, current_time_minutes: int) -> str`
- `event_labels_for_time(self, current_time_minutes: int) -> list[str]`
- `describe_clock(current_time_minutes: int) -> str`

#### `LivePatientDaemon` methods

- `install_signal_handlers(self) -> None`
- `bootstrap(self, *, reset: bool = False) -> None`
- `advance_once(self) -> dict[str, Any]`
- `run(self, *, max_steps: int | None = None) -> None`
- `shutdown(self) -> None`

### Public Functions

- `get_runtime_scenario_profile(name: str) -> RuntimeScenarioProfile`
- `list_runtime_scenario_profiles() -> list[RuntimeScenarioProfile]`
- `is_process_alive(pid: int | None) -> bool`
- `load_runtime_status(workspace: Path) -> dict[str, Any]`

## `iints.live_patient.service_export`

- Source: `src/iints/live_patient/service_export.py`
- Summary: No module docstring.

### Public Functions

- `service_file_text(config: PatientRuntimeConfig, *, service_name: str, user_name: str, python_path: str) -> str`
- `service_instructions_text(service_path: Path, service_name: str) -> str`
- `makerfaire_kiosk_script_text(config: PatientRuntimeConfig) -> str`
- `makerfaire_desktop_entry_text(project_root: Path) -> str`
- `makerfaire_watchdog_name(service_name: str) -> str`
- `makerfaire_watchdog_script_text(*, cli_path: str, scenario_profile: str, seed: int | None) -> str`
- `makerfaire_watchdog_service_text(*, project_root: Path, watchdog_name: str, user_name: str) -> str`
- `makerfaire_watchdog_timer_text(*, watchdog_name: str, interval_seconds: int = 30) -> str`
- `makerfaire_install_script_text(*, service_name: str) -> str`
- `makerfaire_autostart_instructions_text(*, project_root: Path, service_path: Path, desktop_entry_path: Path, kiosk_script_path: Path, install_script_path: Path, watchdog_script_path: Path, watchdog_service_path: Path, watchdog_timer_path: Path, service_name: str) -> str`
- `write_service_artifacts(config: PatientRuntimeConfig, *, output_path: Path, service_name: str = 'iints-digital-patient', user_name: str | None = None, python_path: Path | None = None) -> dict[str, str]`
- `write_makerfaire_autostart_artifacts(config: PatientRuntimeConfig, *, project_root: Path, service_path: Path, service_name: str = 'iints-digital-patient', user_name: str | None = None, cli_path: str | Path | None = None) -> dict[str, str]`
- `uno_q_bridge_service_text(*, project_root: Path, user_name: str, cli_path: str, port: str, baudrate: int, workspace_name: str = 'patient_runtime', service_name: str = 'iints-uno-q-bridge', patient_service_name: str = 'iints-digital-patient') -> str`
- `uno_q_bridge_service_instructions_text(service_path: Path, service_name: str) -> str`
- `write_uno_q_bridge_service_artifact(*, project_root: Path, output_path: Path, user_name: str, cli_path: str | Path, port: str, baudrate: int, workspace_name: str = 'patient_runtime', service_name: str = 'iints-uno-q-bridge', patient_service_name: str = 'iints-digital-patient') -> dict[str, str]`

## `iints.live_patient.uno_q`

- Source: `src/iints/live_patient/uno_q.py`
- Summary: No module docstring.

### Public Functions

- `list_uno_q_serial_ports() -> list[str]`
- `resolve_uno_q_port(port: str | None) -> str`
- `uno_q_bridge_environment_report(*, arduino_cli: str = 'arduino-cli') -> dict[str, Any]`
- `export_uno_q_bridge(output_dir: str | Path) -> dict[str, str]`
- `bridge_state_from_runtime_status(status: Mapping[str, Any] | None) -> str`
- `send_uno_q_bridge_state(port: str | None, state: str, *, baudrate: int = UNO_Q_BRIDGE_BAUDRATE, timeout_seconds: float = 1.5, expect_response: bool = True) -> dict[str, Any]`
- `run_uno_q_bridge_test(port: str | None, *, baudrate: int = UNO_Q_BRIDGE_BAUDRATE, delay_seconds: float = 0.75) -> list[dict[str, Any]]`
- `run_uno_q_bridge_forwarder(workspace: str | Path, port: str | None, *, baudrate: int = UNO_Q_BRIDGE_BAUDRATE, poll_interval: float = 1.0, once: bool = False, max_cycles: int | None = None) -> dict[str, Any]`
- `flash_uno_q_bridge(sketch_dir: str | Path, *, port: str, fqbn: str, arduino_cli: str = 'arduino-cli') -> dict[str, Any]`

### Public Constants

- `UNO_Q_BRIDGE_BAUDRATE`
- `UNO_Q_BRIDGE_BOOT_DELAY_SECONDS`
- `UNO_Q_BRIDGE_READY_BANNER`
- `UNO_Q_BRIDGE_READ_POLL_SECONDS`
- `UNO_Q_BRIDGE_STATES`

## `iints.mdmp`

- Source: `src/iints/mdmp/__init__.py`
- Summary: MDMP public API (separated namespace).
- Explicit exports: `StreamSpec, FeatureSpec, LabelSpec, ValidationSpec, ProcessSpec, ModelReadyContract, compile_contract, parse_contract, load_contract_yaml, ContractRunner, ValidationResult, CheckResult, MDMP_PROTOCOL_VERSION, MDMP_GRADE_ORDER, classify_mdmp_grade, mdmp_grade_meets_minimum, dataframe_fingerprint, mdmp_gate, MDMPGateError, generate_synthetic_mirror, SyntheticMirrorArtifact, build_mdmp_dashboard_html, MDMPValidationResult, MDMPCheckResult, BACKEND_MDMP_GRADE_ORDER, backend_mdmp_grade_meets_minimum, get_backend, is_mdmp_available, active_mdmp_backend, load_mdmp_contract, run_mdmp_validation, build_mdmp_dashboard_html_with_backend, CORE_AI_PACT_CONTROLS, EU_AI_PACT_CONTROL_DESCRIPTIONS, HIGH_RISK_READINESS_CONTROLS, EUAIPactReadinessResult, review_eu_ai_pact_readiness`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.mdmp.backend`

- Source: `src/iints/mdmp/backend.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `MDMPCheckResult` | `MDMPCheckResult` | No module docstring. |
| `MDMPValidationResult` | `MDMPValidationResult` | No module docstring. |

#### `MDMPCheckResult` methods

- `to_dict(self) -> dict[str, Any]`

#### `MDMPValidationResult` methods

- `to_dict(self) -> dict[str, Any]`

### Public Functions

- `mdmp_grade_meets_minimum(actual_grade: str, minimum_grade: str) -> bool`
- `is_mdmp_available() -> bool`
- `get_backend() -> str`
- `active_mdmp_backend() -> str`
- `load_mdmp_contract(path: Path) -> Any`
- `run_mdmp_validation(contract: Any, df: pd.DataFrame, *, apply_builtin_transforms: bool = True) -> MDMPValidationResult`
- `build_mdmp_dashboard_html(report: dict[str, Any], *, title: str) -> str`

### Public Constants

- `BACKEND_BUILTIN`
- `BACKEND_MDMP`
- `MDMP_GRADE_ORDER`

## `iints.mdmp.eu_ai_pact`

- Source: `src/iints/mdmp/eu_ai_pact.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `EUAIPactReadinessResult` | `EUAIPactReadinessResult` | No module docstring. |

#### `EUAIPactReadinessResult` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `review_eu_ai_pact_readiness(payload: Mapping[str, Any], *, strict: bool = True) -> EUAIPactReadinessResult`

### Public Constants

- `CORE_AI_PACT_CONTROLS`
- `EU_AI_PACT_CONTROL_DESCRIPTIONS`
- `HIGH_RISK_READINESS_CONTROLS`

## `iints.metrics`

- Source: `src/iints/metrics.py`
- Summary: No module docstring.

### Public Functions

- `calculate_gmi(glucose: pd.Series) -> float`
- `calculate_cv(glucose: pd.Series) -> float`
- `calculate_lbgi(glucose: pd.Series) -> float`
- `calculate_hbgi(glucose: pd.Series) -> float`
- `calculate_tir(glucose: pd.Series, low: float = 70, high: float = 180) -> float`
- `calculate_full_metrics(glucose: pd.Series, duration_hours: Optional[float] = None)`

## `iints.population`

- Source: `src/iints/population/__init__.py`
- Summary: No module docstring.
- Explicit exports: `PopulationGenerator, PopulationConfig, ParameterDistribution, PopulationRunner, PopulationResult, PatientResult`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.population.generator`

- Source: `src/iints/population/generator.py`
- Summary: Population Generator — IINTS-AF ================================ Generates a virtual population of N patients with physiological variation around a base patient profile. Each parameter is drawn from a configurable distribution (truncated normal or log-normal) whose bounds respect the clinically valid ranges defined in the SDK validation schemas.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ParameterDistribution` | `ParameterDistribution` | Distribution specification for a single patient parameter. |
| `PopulationConfig` | `PopulationConfig` | Configuration for virtual population generation. |
| `PopulationGenerator` | `PopulationGenerator` | Generates *N* virtual :class:`PatientProfile` instances with physiological variation drawn from configurable distributions. |

#### `PopulationGenerator` methods

- `generate(self) -> List[PatientProfile]`

## `iints.population.runner`

- Source: `src/iints/population/runner.py`
- Summary: Population Runner — IINTS-AF ============================== Runs N virtual patients through the simulator in parallel using ``concurrent.futures.ProcessPoolExecutor``. Each worker loads the algorithm from a file path (same pattern as ``run-parallel`` in the CLI) so that all algorithm classes are safely picklable.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PatientResult` | `PatientResult` | Result for a single patient in the population. |
| `PopulationResult` | `PopulationResult` | Aggregate result for the entire population run. |
| `PopulationRunner` | `PopulationRunner` | Runs a virtual patient population through the simulator in parallel. |

#### `PopulationRunner` methods

- `run(self, profiles) -> PopulationResult`

## `iints.presets`

- Source: `src/iints/presets/__init__.py`
- Summary: Built-in clinic-safe presets.
- Explicit exports: `load_presets, get_preset`

### Public Functions

- `load_presets() -> List[Dict[str, Any]]`
- `get_preset(name: str) -> Dict[str, Any]`

## `iints.research`

- Source: `src/iints/research/__init__.py`
- Summary: No module docstring.
- Explicit exports: `PredictorConfig, TrainingConfig, build_sequences, subject_split, FeatureScaler, load_parquet, save_parquet, load_dataset, save_dataset, compute_dataset_lineage, LSTMPredictor, load_predictor, PredictorService, load_predictor_service, QuantileLoss, SafetyWeightedMSE, BandWeightedMSE, regression_metrics, band_regression_metrics, interval_coverage_metrics, forecast_error_report, hypoglycemia_detection_report, uncertainty_reliability_report, subgroup_error_report, feature_drift_report, audit_subject_split_and_leakage, ForecastCalibrationGate, evaluate_calibration_gate, load_calibration_gate_profiles, PromotionResult, append_registry_entry, list_registry, load_registry, promote_registry_run, write_registry, CONTROL_FEATURE_COLUMNS, CONTROL_TARGET_COLUMN, build_control_dataset_from_runs, evaluate_controller_predictions, load_linear_controller, predict_linear_controller, save_linear_controller, summarize_control_dataset, train_linear_imitation_controller, NeuralControllerConfig, instantiate_neural_controller_model, load_neural_controller, predict_neural_controller, save_neural_controller, train_neural_imitation_controller, PREDICTOR_OPTIONAL_COLUMNS, PREDICTOR_REQUIRED_COLUMNS, blend_predictor_datasets, DEFAULT_HELD_OUT_PRESETS, evaluate_controller_factories, DEFAULT_LOCAL_AI_SAFETY_PROFILE, LocalAIGateResult, LocalAISafetyProfile, review_closed_loop_evaluation, review_controller_training_artifacts, build_predictor_dataset_from_runs, run_local_ai_lab`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.research.audit`

- Source: `src/iints/research/audit.py`
- Summary: No module docstring.

### Public Functions

- `audit_subject_split_and_leakage(df: pd.DataFrame, *, history_steps: int, horizon_steps: int, feature_columns: List[str], target_column: str, subject_column: str = 'subject_id', segment_column: Optional[str] = None, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42) -> Dict[str, Any]`

## `iints.research.calibration_gate`

- Source: `src/iints/research/calibration_gate.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ForecastCalibrationGate` | `ForecastCalibrationGate` | No module docstring. |

### Public Functions

- `load_calibration_gate_profiles(path: Optional[Path] = None) -> Dict[str, ForecastCalibrationGate]`
- `evaluate_calibration_gate(report: Dict[str, Any], gate: ForecastCalibrationGate) -> Dict[str, Dict[str, Any]]`

## `iints.research.config`

- Source: `src/iints/research/config.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PredictorConfig` | `PredictorConfig` | No module docstring. |
| `TrainingConfig` | `TrainingConfig` | No module docstring. |

#### `PredictorConfig` methods

- `history_steps(self) -> int`
- `horizon_steps(self) -> int`

## `iints.research.control`

- Source: `src/iints/research/control.py`
- Summary: No module docstring.

### Public Functions

- `build_control_dataset_from_runs(run_dirs: Iterable[Tuple[str, Path]], *, output_path: Path, manifest_path: Path | None = None) -> Dict[str, Any]`
- `summarize_control_dataset(df: pd.DataFrame) -> Dict[str, Any]`
- `train_linear_imitation_controller(df: pd.DataFrame, *, ridge_lambda: float = 0.001) -> Dict[str, Any]`
- `predict_linear_controller(model: Dict[str, Any], df: pd.DataFrame) -> np.ndarray`
- `evaluate_controller_predictions(df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]`
- `save_linear_controller(model: Dict[str, Any], path: Path) -> None`
- `load_linear_controller(path: Path) -> Dict[str, Any]`

### Public Constants

- `CONTROL_FEATURE_COLUMNS`
- `CONTROL_TARGET_COLUMN`

## `iints.research.control_eval`

- Source: `src/iints/research/control_eval.py`
- Summary: No module docstring.

### Public Functions

- `evaluate_controller_factories(factories: Dict[str, ControllerFactory], *, output_dir: Path, presets: Iterable[str] = DEFAULT_HELD_OUT_PRESETS, seeds: Iterable[int] = (101, 202, 303), duration_minutes: int = 1440, time_step_minutes: int = 5, sensor_profile: str = 'clinical_cgm') -> Dict[str, Any]`

### Public Constants

- `DEFAULT_HELD_OUT_PRESETS`

## `iints.research.data_blend`

- Source: `src/iints/research/data_blend.py`
- Summary: No module docstring.

### Public Functions

- `blend_predictor_datasets(sources: Iterable[Tuple[str, Path]], *, output_path: Path, manifest_path: Path | None = None) -> Dict[str, Any]`

### Public Constants

- `PREDICTOR_OPTIONAL_COLUMNS`
- `PREDICTOR_REQUIRED_COLUMNS`

## `iints.research.dataset`

- Source: `src/iints/research/dataset.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `FeatureScaler` | `FeatureScaler` | Fit-transform scaler for LSTM feature arrays. |

#### `FeatureScaler` methods

- `fit(self, X: np.ndarray) -> 'FeatureScaler'`
- `transform(self, X: np.ndarray) -> np.ndarray`
- `fit_transform(self, X: np.ndarray) -> np.ndarray`
- `inverse_transform(self, X: np.ndarray) -> np.ndarray`
- `to_dict(self) -> dict`
- `from_dict(cls, d: dict) -> 'FeatureScaler'`

### Public Functions

- `build_sequences(df: pd.DataFrame, history_steps: int, horizon_steps: int, feature_columns: List[str], target_column: str, subject_column: Optional[str] = 'subject_id', segment_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]`
- `subject_split(df: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.15, subject_column: str = 'subject_id', seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- `save_parquet(df: pd.DataFrame, path: Path) -> None`
- `save_dataset(df: pd.DataFrame, path: Path) -> None`
- `load_parquet(path: Path) -> pd.DataFrame`
- `load_dataset(path: Path) -> pd.DataFrame`
- `compute_dataset_lineage(df: pd.DataFrame, source_path: Optional[Path] = None, subject_column: str = 'subject_id', time_column: str = 'time_minutes') -> Dict[str, Any]`
- `concat_runs(frames: Iterable[pd.DataFrame]) -> pd.DataFrame`
- `basic_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]`

## `iints.research.evaluation`

- Source: `src/iints/research/evaluation.py`
- Summary: No module docstring.

### Public Functions

- `forecast_error_report(observed: np.ndarray, predicted: np.ndarray, predicted_std: Optional[np.ndarray] = None) -> Dict[str, Any]`
- `hypoglycemia_detection_report(observed: np.ndarray, predicted: np.ndarray, *, threshold_mgdl: float = 70.0) -> Dict[str, Any]`
- `uncertainty_reliability_report(observed: np.ndarray, predicted: np.ndarray, predicted_std: np.ndarray, *, bins: int = 5, confidence: float = 0.95) -> Dict[str, Any]`
- `subgroup_error_report(observed: np.ndarray, predicted: np.ndarray, groups: Sequence[Any] | np.ndarray, *, predicted_std: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]`
- `feature_drift_report(reference_features: np.ndarray, candidate_features: np.ndarray, *, feature_names: Iterable[str]) -> Dict[str, Any]`

## `iints.research.local_ai`

- Source: `src/iints/research/local_ai.py`
- Summary: No module docstring.

### Public Functions

- `build_predictor_dataset_from_runs(run_dirs: Iterable[Tuple[str, Path]], *, output_path: Path, manifest_path: Path | None = None) -> Dict[str, Any]`
- `run_local_ai_lab(run_dirs: Iterable[Tuple[str, Path]], *, output_dir: Path, repo_root: Path, train_predictor: bool = True, predictor_config_path: Path | None = None, train_neural: bool = True, evaluate: bool = True, evaluation_presets: Iterable[str] = DEFAULT_HELD_OUT_PRESETS, evaluation_seeds: Iterable[int] = (101, 202, 303), evaluation_duration_minutes: int = 1440) -> Dict[str, Any]`

### Public Constants

- `PREDICTOR_EXPORT_COLUMNS`

## `iints.research.local_ai_gate`

- Source: `src/iints/research/local_ai_gate.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `LocalAISafetyProfile` | `LocalAISafetyProfile` | No module docstring. |
| `LocalAIGateResult` | `LocalAIGateResult` | No module docstring. |

#### `LocalAIGateResult` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `review_controller_training_artifacts(controller_summary: Dict[str, Any], *, train_metrics: Dict[str, Any] | None = None, profile: LocalAISafetyProfile = DEFAULT_LOCAL_AI_SAFETY_PROFILE) -> LocalAIGateResult`
- `review_closed_loop_evaluation(algorithms: Dict[str, Dict[str, Any]], *, baseline_name: str = 'clinical_baseline', profile: LocalAISafetyProfile = DEFAULT_LOCAL_AI_SAFETY_PROFILE) -> LocalAIGateResult`

### Public Constants

- `DEFAULT_LOCAL_AI_SAFETY_PROFILE`

## `iints.research.losses`

- Source: `src/iints/research/losses.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.research.metrics`

- Source: `src/iints/research/metrics.py`
- Summary: No module docstring.

### Public Functions

- `regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]`
- `band_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, low_threshold: float = 70.0, high_threshold: float = 180.0) -> Dict[str, Dict[str, float]]`
- `interval_coverage_metrics(y_true: np.ndarray, mean_pred: np.ndarray, std_pred: np.ndarray, confidence: float = 0.95, low_threshold: float = 70.0, high_threshold: float = 180.0) -> Dict[str, Any]`

## `iints.research.model_registry`

- Source: `src/iints/research/model_registry.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `PromotionResult` | `PromotionResult` | No module docstring. |

#### `PromotionResult` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `load_registry(path: Path) -> List[Dict[str, Any]]`
- `write_registry(path: Path, rows: List[Dict[str, Any]]) -> None`
- `append_registry_entry(path: Path, entry: Dict[str, Any]) -> None`
- `list_registry(path: Path) -> List[Dict[str, Any]]`
- `promote_registry_run(path: Path, *, run_id: str, stage: ModelStage, force: bool = False) -> PromotionResult`

## `iints.research.neural_control`

- Source: `src/iints/research/neural_control.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `NeuralControllerConfig` | `NeuralControllerConfig` | No module docstring. |

### Public Functions

- `instantiate_neural_controller_model(payload: Dict[str, Any]) -> Any`
- `train_neural_imitation_controller(df: pd.DataFrame, *, config: NeuralControllerConfig | None = None) -> Dict[str, Any]`
- `predict_neural_controller(payload: Dict[str, Any], df: pd.DataFrame) -> np.ndarray`
- `save_neural_controller(payload: Dict[str, Any], path: Path) -> None`
- `load_neural_controller(path: Path) -> Dict[str, Any]`

## `iints.research.predictor`

- Source: `src/iints/research/predictor.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `BaselinePredictor` | `BaselinePredictor(Protocol)` | No module docstring. |
| `LastValueBaseline` | `LastValueBaseline` | Naïve last-value (persistence) baseline for glucose forecasting. |
| `LinearTrendBaseline` | `LinearTrendBaseline` | Linear-trend extrapolation baseline. |
| `PredictorService` | `PredictorService` | No module docstring. |

#### `BaselinePredictor` methods

- `predict(self, X: np.ndarray) -> np.ndarray`
- `name(self) -> str`

#### `LastValueBaseline` methods

- `predict(self, X: np.ndarray) -> np.ndarray`
- `name(self) -> str`

#### `LinearTrendBaseline` methods

- `predict(self, X: np.ndarray) -> np.ndarray`
- `name(self) -> str`

#### `PredictorService` methods

- `predict(self, x: np.ndarray) -> np.ndarray`
- `predict_with_uncertainty(self, x: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]`

### Public Functions

- `evaluate_baselines(X: np.ndarray, y: np.ndarray, horizon_steps: int, time_step_minutes: float = 5.0) -> dict`
- `load_predictor(model_path: Path) -> Tuple['LSTMPredictor', dict]`
- `load_predictor_service(model_path: Path) -> PredictorService`
- `predict_batch(model: 'LSTMPredictor', x: np.ndarray) -> np.ndarray`

## `iints.scenarios`

- Source: `src/iints/scenarios/__init__.py`
- Summary: No module docstring.
- Explicit exports: `ScenarioGeneratorConfig, generate_random_scenario, build_eucys_arm_scenario, build_eucys_study_pack, build_official_study_pack, export_eucys_study_pack, export_official_study_pack`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.scenarios.generator`

- Source: `src/iints/scenarios/generator.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ScenarioGeneratorConfig` | `ScenarioGeneratorConfig` | No module docstring. |

### Public Functions

- `generate_random_scenario(config: ScenarioGeneratorConfig) -> Dict[str, Any]`

## `iints.scenarios.study_pack`

- Source: `src/iints/scenarios/study_pack.py`
- Summary: No module docstring.

### Public Functions

- `build_official_study_pack(*, seeds: list[int] | None = None) -> dict[str, Any]`
- `build_eucys_study_pack(*, seeds: list[int] | None = None) -> dict[str, Any]`
- `build_eucys_arm_scenario(base_scenario: dict[str, Any], *, arm_id: str) -> tuple[dict[str, Any], dict[str, Any]]`
- `export_official_study_pack(output_dir: str | Path, *, seeds: list[int] | None = None) -> dict[str, str]`
- `export_eucys_study_pack(output_dir: str | Path, *, seeds: list[int] | None = None) -> dict[str, str]`

## `iints.templates`

- Source: `src/iints/templates/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.templates.default_algorithm`

- Source: `src/iints/templates/default_algorithm.py`
- Summary: Source could not be parsed as normal Python.
- Parse note: `invalid syntax at line 4`

This file is documented as a source artifact rather than a normal importable Python module.

## `iints.templates.demos`

- Source: `src/iints/templates/demos/__init__.py`
- Summary: Bundled demo script templates for installed IINTS users.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.templates.demos.live_stage_demo`

- Source: `src/iints/templates/demos/live_stage_demo.py`
- Summary: No module docstring.

### Public Functions

- `main() -> None`

### Public Constants

- `DURATION_MINUTES`
- `OUTPUT_DIR`
- `PATIENT_CONFIG`
- `PREPARE_AI`
- `SCENARIOS`
- `SDK_SRC`
- `SEED`
- `TIME_STEP_MINUTES`

## `iints.templates.pico_pump.code`

- Source: `src/iints/templates/pico_pump/code.py`
- Summary: IINTS Pico Pump Bench Firmware.

### Public Constants

- `BOOT_TIME`
- `DEVICE_ID`
- `HARDWARE_ACTUATION_ENABLED`
- `LOCKED`
- `READY_BANNER`

## `iints.templates.scenarios`

- Source: `src/iints/templates/scenarios/__init__.py`
- Summary: No module docstring.

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.utils`

- Source: `src/iints/utils/__init__.py`
- Summary: No module docstring.
- Explicit exports: `apply_plot_style`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.utils.csv_safety`

- Source: `src/iints/utils/csv_safety.py`
- Summary: No module docstring.

### Public Functions

- `sanitize_csv_cell(value: Any) -> Any`
- `sanitize_csv_mapping(row: Mapping[str, Any]) -> dict[str, Any]`
- `sanitize_csv_dataframe(df: pd.DataFrame) -> pd.DataFrame`

## `iints.utils.plotting`

- Source: `src/iints/utils/plotting.py`
- Summary: No module docstring.

### Public Functions

- `apply_plot_style(dpi: int = 150, font_scale: float = 1.1, palette: Optional[Iterable[str]] = None) -> List[str]`

### Public Constants

- `IINTS_BLUE`
- `IINTS_GOLD`
- `IINTS_NAVY`
- `IINTS_ORANGE`
- `IINTS_RED`
- `IINTS_TEAL`

## `iints.utils.run_io`

- Source: `src/iints/utils/run_io.py`
- Summary: No module docstring.

### Public Functions

- `resolve_seed(seed: Optional[int]) -> int`
- `generate_run_id(seed: int) -> str`
- `resolve_output_dir(output_dir: Optional[Union[str, Path]], run_id: str) -> Path`
- `write_json(path: Path, payload: Dict[str, Any]) -> None`
- `get_sdk_version(package_name: str = 'iints-sdk-python35') -> str`
- `build_run_metadata(run_id: str, seed: int, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]`
- `compute_sha256(path: Path) -> str`
- `build_run_manifest(output_dir: Path, files: Dict[str, Path]) -> Dict[str, Any]`
- `maybe_sign_manifest(manifest_path: Path) -> Optional[Path]`

### Public Constants

- `RESULTS_CSV_FORMAT_VERSION`
- `RUN_MANIFEST_FORMAT_VERSION`
- `RUN_METADATA_FORMAT_VERSION`

## `iints.utils.url_safety`

- Source: `src/iints/utils/url_safety.py`
- Summary: No module docstring.

### Public Functions

- `validate_service_base_url(raw_url: str, *, label: str) -> str`

## `iints.validation`

- Source: `src/iints/validation/__init__.py`
- Summary: No module docstring.
- Explicit exports: `StressEventModel, ScenarioModel, PatientConfigModel, LATEST_SCHEMA_VERSION, ValidationRule, ValidationProfile, ValidationCheckResult, RunValidationReport, SafetyContractSpec, SafetyContractViolation, SafetyContractVerificationReport, load_scenario, validate_scenario_dict, scenario_warnings, build_stress_events, scenario_to_payloads, validate_patient_config_dict, load_patient_config, load_patient_config_by_name, load_validation_profiles, compute_run_metrics, evaluate_run, load_contract_spec, apply_contract_to_config, verify_safety_contract, ReplayRunDigest, ReplayCheckResult, run_deterministic_replay_check, GoldenScenarioSpec, GoldenBenchmarkPack, load_golden_benchmark_pack, evaluate_expected_ranges, format_validation_error, migrate_scenario_dict`

### Public Functions

- `migrate_scenario_dict(data: Dict[str, Any]) -> Dict[str, Any]`
- `load_scenario(path: Union[str, Path]) -> ScenarioModel`
- `validate_scenario_dict(data: Dict[str, Any]) -> ScenarioModel`
- `scenario_warnings(model: ScenarioModel) -> List[str]`
- `build_stress_events(payloads: List[Dict[str, Any]]) -> List[StressEvent]`
- `scenario_to_payloads(model: ScenarioModel) -> List[Dict[str, Any]]`
- `validate_patient_config_dict(data: Dict[str, Any]) -> PatientConfigModel`
- `load_patient_config(path: Union[str, Path]) -> PatientConfigModel`
- `load_patient_config_by_name(name: str) -> PatientConfigModel`
- `format_validation_error(error: ValidationError) -> List[str]`

## `iints.validation.golden`

- Source: `src/iints/validation/golden.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `GoldenScenarioSpec` | `GoldenScenarioSpec` | No module docstring. |
| `GoldenBenchmarkPack` | `GoldenBenchmarkPack` | No module docstring. |

### Public Functions

- `load_golden_benchmark_pack(path: Optional[Path] = None) -> GoldenBenchmarkPack`
- `evaluate_expected_ranges(metrics: Dict[str, float], expected: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]`

## `iints.validation.replay`

- Source: `src/iints/validation/replay.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ReplayRunDigest` | `ReplayRunDigest` | No module docstring. |
| `ReplayCheckResult` | `ReplayCheckResult` | No module docstring. |

#### `ReplayRunDigest` methods

- `to_dict(self) -> Dict[str, Any]`

#### `ReplayCheckResult` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `run_deterministic_replay_check(*, algorithm: InsulinAlgorithm, scenario: Optional[Dict[str, Any]], patient_config: Any, duration_minutes: int, time_step: int, seed: int, repeats: int = 2, safety_config: Optional[SafetyConfig] = None, predictor: Optional[object] = None) -> ReplayCheckResult`

### Public Constants

- `NON_DETERMINISTIC_COLUMNS`
- `NON_DETERMINISTIC_REPORT_KEYS`

## `iints.validation.run_doctor`

- Source: `src/iints/validation/run_doctor.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `RunDoctorCheck` | `RunDoctorCheck` | No module docstring. |
| `RunDoctorReport` | `RunDoctorReport` | No module docstring. |

#### `RunDoctorCheck` methods

- `to_dict(self) -> Dict[str, Any]`

#### `RunDoctorReport` methods

- `failed(self) -> bool`
- `warned(self) -> bool`
- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `inspect_run_setup(*, algo_path: Optional[Path] = None, patient_config_path: Optional[Path] = None, scenario_path: Optional[Path] = None, duration_minutes: int = 1440, time_step_minutes: int = 5, output_dir: Optional[Path] = None, patient_config_name: Optional[str] = None) -> RunDoctorReport`

## `iints.validation.run_validation`

- Source: `src/iints/validation/run_validation.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `ValidationRule` | `ValidationRule` | No module docstring. |
| `ValidationProfile` | `ValidationProfile` | No module docstring. |
| `ValidationCheckResult` | `ValidationCheckResult` | No module docstring. |
| `RunValidationReport` | `RunValidationReport` | No module docstring. |

#### `ValidationCheckResult` methods

- `to_dict(self) -> Dict[str, Any]`

#### `RunValidationReport` methods

- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `load_validation_profiles(path: Optional[Path] = None) -> Dict[str, ValidationProfile]`
- `compute_run_metrics(results_df: pd.DataFrame, *, safety_report: Optional[Dict[str, Any]] = None, duration_minutes: Optional[int] = None) -> Dict[str, float]`
- `evaluate_run(results_df: pd.DataFrame, *, profile: ValidationProfile, safety_report: Optional[Dict[str, Any]] = None, duration_minutes: Optional[int] = None) -> RunValidationReport`

## `iints.validation.safety_contract`

- Source: `src/iints/validation/safety_contract.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `SafetyContractSpec` | `SafetyContractSpec` | No module docstring. |
| `SafetyContractViolation` | `SafetyContractViolation` | No module docstring. |
| `SafetyContractVerificationReport` | `SafetyContractVerificationReport` | No module docstring. |

#### `SafetyContractVerificationReport` methods

- `passed(self) -> bool`
- `to_dict(self) -> Dict[str, Any]`

### Public Functions

- `load_contract_spec(path: Optional[Path] = None) -> SafetyContractSpec`
- `apply_contract_to_config(spec: SafetyContractSpec, base: Optional[SafetyConfig] = None) -> SafetyConfig`
- `verify_safety_contract(spec: SafetyContractSpec, *, glucose_values: List[float], trend_values: List[float], proposed_doses: List[float], iob_values: Optional[List[float]] = None) -> SafetyContractVerificationReport`

## `iints.validation.schemas`

- Source: `src/iints/validation/schemas.py`
- Summary: No module docstring.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `StressEventModel` | `StressEventModel(BaseModel)` | No module docstring. |
| `ScenarioModel` | `ScenarioModel(BaseModel)` | No module docstring. |
| `PatientConfigModel` | `PatientConfigModel(BaseModel)` | No module docstring. |

### Public Constants

- `LATEST_SCHEMA_VERSION`

## `iints.visualization`

- Source: `src/iints/visualization/__init__.py`
- Summary: IINTS-AF Visualization Module Professional medical visualizations for diabetes algorithm research.
- Explicit exports: `UncertaintyCloud, UncertaintyData, VisualizationConfig, ClinicalCockpit, CockpitConfig, DashboardState`

No public classes, functions, or all-caps constants are declared directly in this module.

## `iints.visualization.cockpit`

- Source: `src/iints/visualization/cockpit.py`
- Summary: Clinical Control Center - IINTS-AF Professional medical dashboard for diabetes algorithm research.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `CockpitConfig` | `CockpitConfig` | Configuration for the clinical cockpit |
| `DashboardState` | `DashboardState` | Current state of the dashboard |
| `ClinicalCockpit` | `ClinicalCockpit` | Professional clinical dashboard for diabetes algorithm research. |

#### `DashboardState` methods

- `to_dict(self) -> Dict`

#### `ClinicalCockpit` methods

- `visualize_results(self, simulation_data: pd.DataFrame, predictions: Optional[np.ndarray] = None, reasoning_logs: Optional[List[Dict]] = None, metrics: Optional[Dict] = None, personality: Optional[Dict] = None, save_path: Optional[str] = None) -> plt.Figure`
- `compare_battle(self, battle_report, save_path: Optional[str] = None) -> plt.Figure`
- `update(self, state: DashboardState)`
- `export_state(self) -> Dict`

### Public Functions

- `demo_clinical_cockpit()`

## `iints.visualization.uncertainty_cloud`

- Source: `src/iints/visualization/uncertainty_cloud.py`
- Summary: Uncertainty Cloud Visualizer - IINTS-AF Creates visualization of AI confidence as shadow around glucose predictions.

### Public Classes

| Class | Signature | Summary |
| --- | --- | --- |
| `UncertaintyData` | `UncertaintyData` | Data structure for uncertainty visualization |
| `VisualizationConfig` | `VisualizationConfig` | Configuration for uncertainty cloud visualization |
| `UncertaintyCloud` | `UncertaintyCloud` | Creates uncertainty cloud visualizations for glucose predictions. |

#### `UncertaintyCloud` methods

- `plot(self, data: UncertaintyData, time_range: Optional[Tuple[float, float]] = None, save_path: Optional[str] = None) -> plt.Figure`
- `plot_comparison(self, data_list: List[Tuple[str, UncertaintyData]], save_path: Optional[str] = None) -> plt.Figure`
- `create_dashboard_widget(self, data: UncertaintyData, width: int = 400, height: int = 200) -> plt.Figure`

### Public Functions

- `generate_sample_data() -> UncertaintyData`
- `demo_uncertainty_cloud()`
