from .config import PredictorConfig, TrainingConfig
from .dataset import (
    build_sequences,
    subject_split,
    FeatureScaler,
    load_parquet,
    save_parquet,
    load_dataset,
    save_dataset,
    compute_dataset_lineage,
)
from .predictor import LSTMPredictor, load_predictor, PredictorService, load_predictor_service
from .losses import (
    BandWeightedMSE,
    BandWeightedPINNLoss,
    PhysiologicalPINNLoss,
    QuantileLoss,
    SafetyWeightedMSE,
)
from .metrics import regression_metrics, band_regression_metrics, interval_coverage_metrics
from .evaluation import (
    feature_drift_report,
    forecast_error_report,
    hypoglycemia_detection_report,
    subgroup_error_report,
    uncertainty_reliability_report,
)
from .audit import audit_subject_split_and_leakage
from .calibration_gate import (
    ForecastCalibrationGate,
    evaluate_calibration_gate,
    load_calibration_gate_profiles,
)
from .model_registry import (
    PromotionResult,
    append_registry_entry,
    list_registry,
    load_registry,
    promote_registry_run,
    write_registry,
)
from .control import (
    CONTROL_FEATURE_COLUMNS,
    CONTROL_TARGET_COLUMN,
    build_control_dataset_from_runs,
    evaluate_controller_predictions,
    load_linear_controller,
    predict_linear_controller,
    save_linear_controller,
    summarize_control_dataset,
    train_linear_imitation_controller,
)
from .neural_control import (
    NeuralControllerConfig,
    instantiate_neural_controller_model,
    load_neural_controller,
    predict_neural_controller,
    save_neural_controller,
    train_neural_imitation_controller,
)
from .data_blend import (
    PREDICTOR_OPTIONAL_COLUMNS,
    PREDICTOR_REQUIRED_COLUMNS,
    blend_predictor_datasets,
)
from .control_eval import (
    DEFAULT_HELD_OUT_PRESETS,
    evaluate_controller_factories,
)
from .local_ai_gate import (
    DEFAULT_LOCAL_AI_SAFETY_PROFILE,
    LocalAIGateResult,
    LocalAISafetyProfile,
    review_closed_loop_evaluation,
    review_controller_training_artifacts,
)
from .local_ai import (
    build_predictor_dataset_from_runs,
    run_local_ai_lab,
)
from .forecasting import (
    DEFAULT_FORECAST_FEATURE_COLUMNS,
    ForecastConfig,
    PhysiologyAwareBaseline,
    assess_forecast_risk,
    attach_forecasts_to_frame,
    resolve_forecast_input,
    summarize_forecast_frame,
    write_forecast_bundle,
)
from .jetson_hf_trainer import (
    JetsonHFTrainingResult,
    model_score as jetson_hf_model_score,
    run_jetson_hf_training,
)
from .results_manager import (
    ResultsIndexBundle,
    build_artifact_inventory,
    discover_result_csvs,
    index_results,
    summarize_results_csv,
)
from .academic_bundle import AcademicBundleResult, build_academic_bundle
from .mechanistic_models import (
    MechanisticRunResult,
    SBMLModelSummary,
    inspect_sbml_model,
    roadrunner_status,
    run_sbml_model,
)
from .copasi_models import (
    COPASIModelSummary,
    COPASIRunResult,
    copasi_status,
    inspect_copasi_model,
    run_copasi_model,
)
from .cellml_models import (
    CellMLModelSummary,
    CellMLValidationResult,
    inspect_cellml_model,
    opencor_status,
    validate_cellml_model,
)
from .fmi_models import (
    FMUModelSummary,
    FMURunResult,
    fmpy_status,
    inspect_fmu_model,
    run_fmu_model,
)
from .binding_evidence import BindingEvidenceResult, query_bindingdb_uniprot
from .clinvar_engine import ClinVarEngine, normalize_protein_variant
from .regenerative_islet import (
    RegenerativeEvidencePlan,
    RegenerativeComparisonResult,
    RegenerativeProteinPanel,
    RegenerativeProteinTarget,
    build_regenerative_evidence_plan,
    compare_regenerative_islet_proteomics,
    get_regenerative_protein_panel,
    load_regenerative_protein_panels,
)
from .proteomics_importer import (
    ProteomicsImportResult,
    load_sample_metadata,
    import_maxquant_protein_groups,
    import_diann_report,
    import_wide_proteomics_matrix,
    import_and_validate_proteomics,
)
from .glucose_model import (
    GLUCOSE_MODEL_FEATURE_COLUMNS,
    GLUCOSE_MODEL_ID,
    GlucoseModelComparisonBundle,
    GlucoseModelSpec,
    GlucoseTrainingPack,
    build_glucose_training_pack,
    compare_glucose_models,
    glucose_model_config_payload,
    horizon_error_rows,
    parse_model_specs,
    physiological_violation_report,
    public_manifest_from_private,
    render_hf_comparison_interpretation,
    standardize_glucose_forecast_frame,
    write_glucose_model_config,
    write_huggingface_export_bundle,
)

__all__ = [
    "PredictorConfig",
    "TrainingConfig",
    "build_sequences",
    "subject_split",
    "FeatureScaler",
    "load_parquet",
    "save_parquet",
    "load_dataset",
    "save_dataset",
    "compute_dataset_lineage",
    "LSTMPredictor",
    "load_predictor",
    "PredictorService",
    "load_predictor_service",
    "QuantileLoss",
    "SafetyWeightedMSE",
    "BandWeightedMSE",
    "PhysiologicalPINNLoss",
    "BandWeightedPINNLoss",
    "regression_metrics",
    "band_regression_metrics",
    "interval_coverage_metrics",
    "forecast_error_report",
    "hypoglycemia_detection_report",
    "uncertainty_reliability_report",
    "subgroup_error_report",
    "feature_drift_report",
    "audit_subject_split_and_leakage",
    "ForecastCalibrationGate",
    "evaluate_calibration_gate",
    "load_calibration_gate_profiles",
    "PromotionResult",
    "append_registry_entry",
    "list_registry",
    "load_registry",
    "promote_registry_run",
    "write_registry",
    "CONTROL_FEATURE_COLUMNS",
    "CONTROL_TARGET_COLUMN",
    "build_control_dataset_from_runs",
    "evaluate_controller_predictions",
    "load_linear_controller",
    "predict_linear_controller",
    "save_linear_controller",
    "summarize_control_dataset",
    "train_linear_imitation_controller",
    "NeuralControllerConfig",
    "instantiate_neural_controller_model",
    "load_neural_controller",
    "predict_neural_controller",
    "save_neural_controller",
    "train_neural_imitation_controller",
    "PREDICTOR_OPTIONAL_COLUMNS",
    "PREDICTOR_REQUIRED_COLUMNS",
    "blend_predictor_datasets",
    "DEFAULT_HELD_OUT_PRESETS",
    "evaluate_controller_factories",
    "DEFAULT_LOCAL_AI_SAFETY_PROFILE",
    "LocalAIGateResult",
    "LocalAISafetyProfile",
    "review_closed_loop_evaluation",
    "review_controller_training_artifacts",
    "build_predictor_dataset_from_runs",
    "run_local_ai_lab",
    "DEFAULT_FORECAST_FEATURE_COLUMNS",
    "ForecastConfig",
    "PhysiologyAwareBaseline",
    "assess_forecast_risk",
    "attach_forecasts_to_frame",
    "resolve_forecast_input",
    "summarize_forecast_frame",
    "write_forecast_bundle",
    "ResultsIndexBundle",
    "build_artifact_inventory",
    "discover_result_csvs",
    "index_results",
    "summarize_results_csv",
    "AcademicBundleResult",
    "build_academic_bundle",
    "MechanisticRunResult",
    "SBMLModelSummary",
    "inspect_sbml_model",
    "roadrunner_status",
    "run_sbml_model",
    "COPASIModelSummary",
    "COPASIRunResult",
    "copasi_status",
    "inspect_copasi_model",
    "run_copasi_model",
    "CellMLModelSummary",
    "CellMLValidationResult",
    "inspect_cellml_model",
    "opencor_status",
    "validate_cellml_model",
    "FMUModelSummary",
    "FMURunResult",
    "fmpy_status",
    "inspect_fmu_model",
    "run_fmu_model",
    "BindingEvidenceResult",
    "query_bindingdb_uniprot",
    "ClinVarEngine",
    "normalize_protein_variant",
    "RegenerativeEvidencePlan",
    "RegenerativeComparisonResult",
    "RegenerativeProteinPanel",
    "RegenerativeProteinTarget",
    "build_regenerative_evidence_plan",
    "compare_regenerative_islet_proteomics",
    "get_regenerative_protein_panel",
    "load_regenerative_protein_panels",
    "ProteomicsImportResult",
    "load_sample_metadata",
    "import_maxquant_protein_groups",
    "import_diann_report",
    "import_wide_proteomics_matrix",
    "import_and_validate_proteomics",
    "GLUCOSE_MODEL_FEATURE_COLUMNS",
    "GLUCOSE_MODEL_ID",
    "GlucoseModelComparisonBundle",
    "GlucoseModelSpec",
    "GlucoseTrainingPack",
    "build_glucose_training_pack",
    "compare_glucose_models",
    "glucose_model_config_payload",
    "horizon_error_rows",
    "parse_model_specs",
    "physiological_violation_report",
    "public_manifest_from_private",
    "render_hf_comparison_interpretation",
    "standardize_glucose_forecast_frame",
    "write_glucose_model_config",
    "JetsonHFTrainingResult",
    "jetson_hf_model_score",
    "run_jetson_hf_training",
    "DualStreamDecomposition",
    "decompose_dual_stream",
    "extract_dual_stream_pre_meal_features",
    "PPGRTrajectoryMetrics",
    "PPGRBenchmarkResult",
    "BasePPGRModel",
    "CarbOnlyLinearPPGR",
    "MultiMacroLinearPPGR",
    "DualStreamGlucoFMPPGR",
    "compute_trajectory_metrics",
    "build_ppgr_dataset",
    "run_ppgr_benchmark",
    "CGMJEPAConfig",
    "CGMJEPAEncoder",
    "load_cgm_jepa_model",
    "extract_cgm_jepa_embeddings",
    "SimulationJEPAEmbeddingResult",
    "prepare_cgm_jepa_window",
    "bridge_simulation_to_jepa",
    "PhysiologicalSensitivityResult",
    "simulate_physiological_cgm_24h",
    "add_sensor_noise_and_dropouts",
    "run_cgm_jepa_parameter_experiment",
    "ConfounderPairResult",
    "PhysiologicalConfounderStudyResult",
    "generate_confounded_physiological_pair",
    "run_physiological_confounder_experiment",
]

from .dual_stream import (
    DualStreamDecomposition,
    decompose_dual_stream,
    extract_dual_stream_pre_meal_features,
)
from .ppgr import (
    PPGRTrajectoryMetrics,
    PPGRBenchmarkResult,
    BasePPGRModel,
    CarbOnlyLinearPPGR,
    MultiMacroLinearPPGR,
    DualStreamGlucoFMPPGR,
    compute_trajectory_metrics,
    build_ppgr_dataset,
    run_ppgr_benchmark,
)
# The modules below (CGM-JEPA, GlucoFM, the foundation-model arena, and the
# visualizer/EUCYS report generators that build on them) require torch, unlike
# everything imported above. Torch is an optional "research" extra
# (`pip install iints-sdk-python35[research]`), so a missing torch here must
# not prevent the rest of iints.research from importing; the affected names
# simply become unavailable (None) instead.
try:
    from .cgm_jepa import (
        CGMJEPAConfig,
        CGMJEPAEncoder,
        load_cgm_jepa_model,
        extract_cgm_jepa_embeddings,
    )
    from .cgm_jepa_bridge import (
        SimulationJEPAEmbeddingResult,
        prepare_cgm_jepa_window,
        bridge_simulation_to_jepa,
    )
    from .cgm_jepa_experiment import (
        PhysiologicalSensitivityResult,
        simulate_physiological_cgm_24h,
        add_sensor_noise_and_dropouts,
        run_cgm_jepa_parameter_experiment,
    )
    from .cgm_jepa_confounder import (
        ConfounderPairResult,
        PhysiologicalConfounderStudyResult,
        generate_confounded_physiological_pair,
        run_physiological_confounder_experiment,
    )
    from .glucofm import (
        GlucoFMConfig,
        GlucoFMStreamEncoder,
        GlucoFMDualStreamEncoder,
        GlucoFMDownstreamProbes,
        build_glucofm_foundation_model,
        embed_cgm_with_glucofm,
    )
    from .foundation_arena import (
        ModelArenaMetrics,
        FoundationArenaReport,
        run_foundation_model_arena,
    )
    from .visualizer import (
        ScientificVisualizationArtifacts,
        plot_foundation_arena_radar,
        plot_confounder_cosine_analysis,
        plot_glucofm_dual_stream_decomposition,
        plot_cgmacros_dualsensor_comparison,
        plot_fda_safety_mitigation_timeline,
        generate_interactive_dashboard_html,
        generate_all_scientific_visualizations,
    )
    from .eucys_playbook_generator import (
        EUCYSFigureMetadata,
        EUCYSJuryPortfolio,
        plot_clarke_error_grid,
        plot_glycemic_tir_distribution,
        plot_sc_islet_gsis_dynamics,
        plot_regenerative_graft_survival,
        plot_edge_hardware_latency_budget,
        plot_quantum_safe_mdmp_security,
        generate_complete_eucys_jury_portfolio,
    )
except ImportError:
    CGMJEPAConfig = None  # type: ignore[assignment,misc]
    CGMJEPAEncoder = None  # type: ignore[assignment,misc]
    load_cgm_jepa_model = None  # type: ignore[assignment]
    extract_cgm_jepa_embeddings = None  # type: ignore[assignment]
    SimulationJEPAEmbeddingResult = None  # type: ignore[assignment,misc]
    prepare_cgm_jepa_window = None  # type: ignore[assignment]
    bridge_simulation_to_jepa = None  # type: ignore[assignment]
    PhysiologicalSensitivityResult = None  # type: ignore[assignment,misc]
    simulate_physiological_cgm_24h = None  # type: ignore[assignment]
    add_sensor_noise_and_dropouts = None  # type: ignore[assignment]
    run_cgm_jepa_parameter_experiment = None  # type: ignore[assignment]
    ConfounderPairResult = None  # type: ignore[assignment,misc]
    PhysiologicalConfounderStudyResult = None  # type: ignore[assignment,misc]
    generate_confounded_physiological_pair = None  # type: ignore[assignment]
    run_physiological_confounder_experiment = None  # type: ignore[assignment]
    GlucoFMConfig = None  # type: ignore[assignment,misc]
    GlucoFMStreamEncoder = None  # type: ignore[assignment,misc]
    GlucoFMDualStreamEncoder = None  # type: ignore[assignment,misc]
    GlucoFMDownstreamProbes = None  # type: ignore[assignment,misc]
    build_glucofm_foundation_model = None  # type: ignore[assignment]
    embed_cgm_with_glucofm = None  # type: ignore[assignment]
    ModelArenaMetrics = None  # type: ignore[assignment,misc]
    FoundationArenaReport = None  # type: ignore[assignment,misc]
    run_foundation_model_arena = None  # type: ignore[assignment]
    ScientificVisualizationArtifacts = None  # type: ignore[assignment,misc]
    plot_foundation_arena_radar = None  # type: ignore[assignment]
    plot_confounder_cosine_analysis = None  # type: ignore[assignment]
    plot_glucofm_dual_stream_decomposition = None  # type: ignore[assignment]
    plot_cgmacros_dualsensor_comparison = None  # type: ignore[assignment]
    plot_fda_safety_mitigation_timeline = None  # type: ignore[assignment]
    generate_interactive_dashboard_html = None  # type: ignore[assignment]
    generate_all_scientific_visualizations = None  # type: ignore[assignment]
    EUCYSFigureMetadata = None  # type: ignore[assignment,misc]
    EUCYSJuryPortfolio = None  # type: ignore[assignment,misc]
    plot_clarke_error_grid = None  # type: ignore[assignment]
    plot_glycemic_tir_distribution = None  # type: ignore[assignment]
    plot_sc_islet_gsis_dynamics = None  # type: ignore[assignment]
    plot_regenerative_graft_survival = None  # type: ignore[assignment]
    plot_edge_hardware_latency_budget = None  # type: ignore[assignment]
    plot_quantum_safe_mdmp_security = None  # type: ignore[assignment]
    generate_complete_eucys_jury_portfolio = None  # type: ignore[assignment]
