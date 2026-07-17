from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceConnector:
    """One external research evidence source surfaced by the desktop app."""

    key: str
    title: str
    category: str
    integration_status: str
    why_it_matters: str
    app_use: str
    primary_url: str
    docs_url: str
    default_query: str


def list_evidence_connectors() -> list[EvidenceConnector]:
    """Return curated, allowlisted evidence sources for the native workbench.

    These entries are metadata only. The desktop shell may open their official
    documentation or portals in the system browser, but deterministic SDK
    outputs remain the source of truth for generated reports.
    """

    return [
        EvidenceConnector(
            key="alphafold-db",
            title="AlphaFold Protein Structure Database",
            category="Structural biology",
            integration_status="Bundled 3D assets + PAE evidence",
            why_it_matters="Shows predicted protein structure for insulin, glucagon, INSR, GLUT4, and GCGR.",
            app_use="Protein evidence cards, mmCIF opening, PAE matrix links, and molecular context for SDK formulas.",
            primary_url="https://alphafold.ebi.ac.uk/",
            docs_url="https://alphafold.ebi.ac.uk/",
            default_query="P01308 insulin",
        ),
        EvidenceConnector(
            key="ensembl-vep-alphamissense",
            title="Ensembl VEP + AlphaMissense",
            category="Variant interpretation",
            integration_status="Recommended next live connector",
            why_it_matters="Annotates variants and can include AlphaMissense scores for missense pathogenicity context.",
            app_use="Variant evidence panel for mutation-to-simulation workflows such as INSR V938M.",
            primary_url="https://rest.ensembl.org/",
            docs_url="https://rest.ensembl.org/documentation/info/vep_id_post",
            default_query="INSR V938M / rsID",
        ),
        EvidenceConnector(
            key="open-targets",
            title="Open Targets Platform",
            category="Target-disease evidence",
            integration_status="Recommended evidence graph connector",
            why_it_matters="Links targets, diseases, drugs, genetic evidence, tractability, and safety evidence.",
            app_use="Evidence graph cards for INSR, SLC2A4, GCGR, insulin resistance, and diabetes phenotypes.",
            primary_url="https://platform.opentargets.org/",
            docs_url="https://platform-docs.opentargets.org/data-access/graphql-api",
            default_query="SLC2A4 diabetes mellitus",
        ),
        EvidenceConnector(
            key="reactome",
            title="Reactome",
            category="Pathways",
            integration_status="Recommended pathway connector",
            why_it_matters="Expert-authored pathway knowledge for insulin signalling, glucagon signalling, and glucose metabolism.",
            app_use="Pathway panel explaining which biology sits behind each SDK equation block.",
            primary_url="https://reactome.org/",
            docs_url="https://reactome.org/dev/content-service",
            default_query="Insulin receptor signalling cascade",
        ),
        EvidenceConnector(
            key="rcsb-pdb",
            title="RCSB PDB",
            category="Experimental structures",
            integration_status="Recommended structure validation connector",
            why_it_matters="Provides experimental structures and metadata to compare against AlphaFold predictions.",
            app_use="Structure provenance view: predicted AlphaFold vs experimentally solved PDB evidence.",
            primary_url="https://www.rcsb.org/",
            docs_url="https://data.rcsb.org/",
            default_query="insulin receptor structure",
        ),
        EvidenceConnector(
            key="uniprot",
            title="UniProt",
            category="Protein annotation",
            integration_status="Recommended protein summary connector",
            why_it_matters="Canonical protein function, sequence, domains, features, variants, and cross-references.",
            app_use="Protein cards next to the 3D viewer and SDK formula links.",
            primary_url="https://www.uniprot.org/",
            docs_url="https://www.uniprot.org/help/api_queries",
            default_query="P01308",
        ),
        EvidenceConnector(
            key="human-protein-atlas",
            title="Human Protein Atlas",
            category="Tissue/protein expression",
            integration_status="Recommended expression evidence connector",
            why_it_matters="Protein and RNA expression evidence across tissues, useful for compartment plausibility.",
            app_use="Tissue expression cards for GLUT4, INSR, and GCGR.",
            primary_url="https://www.proteinatlas.org/",
            docs_url="https://www.proteinatlas.org/about/help/dataaccess",
            default_query="SLC2A4",
        ),
        EvidenceConnector(
            key="gtex",
            title="GTEx Portal API",
            category="Tissue RNA expression / eQTL",
            integration_status="Partially integrated via expression renders",
            why_it_matters="Human tissue expression and eQTL context for anatomy-aware model explanations.",
            app_use="Interactive tissue-expression plots and tissue-specific resistance stress tests.",
            primary_url="https://gtexportal.org/",
            docs_url="https://gtexportal.org/api/v2/docs",
            default_query="SLC2A4 skeletal muscle",
        ),
        EvidenceConnector(
            key="chembl",
            title="ChEMBL",
            category="Pharmacology",
            integration_status="Recommended drug/PK connector",
            why_it_matters="Drug and molecule metadata for insulin analogues, mechanisms, and pharmacology context.",
            app_use="Insulin analogue cards linked to SDK PK/PD parameters such as absorption delay and tmax.",
            primary_url="https://www.ebi.ac.uk/chembl/",
            docs_url="https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services",
            default_query="insulin lispro",
        ),
        EvidenceConnector(
            key="clinpgx-pharmgkb",
            title="ClinPGx / PharmGKB",
            category="Pharmacogenomics",
            integration_status="Recommended cautious connector",
            why_it_matters="Pharmacogenomics annotations, pathways, chemicals, genes, and clinical annotation context.",
            app_use="Research-only pharmacogenomics context cards; never dosing guidance.",
            primary_url="https://api.pharmgkb.org/",
            docs_url="https://api.pharmgkb.org/swagger/",
            default_query="INSR pharmacogenomics",
        ),
        EvidenceConnector(
            key="biomodels",
            title="BioModels",
            category="Mathematical model provenance",
            integration_status="Recommended model provenance connector",
            why_it_matters="Curated computational biology models and model metadata.",
            app_use="Model evidence library for ODE and glucose-insulin model provenance.",
            primary_url="https://www.ebi.ac.uk/biomodels/",
            docs_url="https://www.ebi.ac.uk/biomodels/dev",
            default_query="glucose insulin model",
        ),
        EvidenceConnector(
            key="string-db",
            title="STRING DB",
            category="Protein interaction networks",
            integration_status="Partially integrated via pathway renders",
            why_it_matters="Protein-protein interaction networks for insulin and glucagon signalling cascades.",
            app_use="Network evidence panels for insulin-cascade and glucagon-rescue diagrams.",
            primary_url="https://string-db.org/",
            docs_url="https://string-db.org/help/api/",
            default_query="INSR IRS1 PIK3CA AKT1 SLC2A4",
        ),
        EvidenceConnector(
            key="clinvar",
            title="ClinVar / NCBI Clinical Tables",
            category="Variant evidence",
            integration_status="Partially integrated via mutation examples",
            why_it_matters="Public variant assertions and variant search metadata for genomic context.",
            app_use="Mutation lookup companion for the genomics simulation panel.",
            primary_url="https://www.ncbi.nlm.nih.gov/clinvar/",
            docs_url="https://clinicaltables.nlm.nih.gov/apidoc/variants/v3/doc.html",
            default_query="INSR V938M",
        ),
    ]
