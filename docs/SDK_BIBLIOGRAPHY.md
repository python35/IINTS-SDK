# IINTS-AF SDK — Bibliography

Compiled 9 September 2026 · SDK 1.5.37 · Tauri Workbench 0.2.14.

## Scope and how to cite

This is a consolidated bibliography of sources traceable in the current repository,
not a claim to recover every website ever read during development. Scientific
publications, data catalogues, optional integrations and software documentation
are distinguished below. Entries use stable citation keys and a numbered,
Vancouver-like presentation. Use only references supporting the specific claims
in your submitted paper; the complete list can be supplied as a supplementary bibliography.

The role after each reference explains the connection to IINTS. A listed dataset
is not proof that it was used in a training run. An implemented connector is not
proof of biological calibration. Literature-inspired extensions, supervisor rules,
heuristic losses and graft dynamics remain project-specific assumptions unless
independently validated. Clinical studies do not certify this SDK.

English titles are retained for the international jury. “n.d.” means no publication
date was asserted for a living documentation page. Software versions must be taken
from the actual run environment, not inferred from minimum dependency requirements.
The associated BibTeX file is [SDK_BIBLIOGRAPHY.bib](SDK_BIBLIOGRAPHY.bib).
Verification notes and unresolved citations are in
[SDK_BIBLIOGRAPHY_AUDIT.md](SDK_BIBLIOGRAPHY_AUDIT.md).

## Core reading for a short paper

Start with Bergman (1979), Hovorka (2004), Dalla Man (2007), Battelino (2019),
Marling and Bunescu (2020), and the software actually used. Add Jumper (2021) and
Varadi (2024) if showing AlphaFold; Li et al. (2026, v2) if discussing the GlucoFM
reproduction; the regenerative-islet papers only if that work is part of your submission.


## 1. Mathematical models, physiology and glucose metrics

**[1]** American Diabetes Association Professional Practice Committee for Diabetes; Bajaj, Mandeep; McCoy, Rozalina G.; Balapattabi, Kirthikaa; Bannuru, Raveendhara R.; Bellini, Natalie J.; et al. (2026). *6. Glycemic Goals, Hypoglycemia, and Hyperglycemic Crises: Standards of Care in Diabetes—2026*. Diabetes Care 49(Supplement_1):S132-S149. [DOI](https://doi.org/10.2337/dc26-S006).

Role: validation_targets; source recorded in the packaged evidence registry. Citation key: `ada_2026_glycemic_goals`.

**[2]** American Diabetes Association Professional Practice Committee for Diabetes; Bajaj, Mandeep; McCoy, Rozalina G.; Balapattabi, Kirthikaa; Bannuru, Raveendhara R.; Bellini, Natalie J.; et al. (2026). *7. Diabetes Technology: Standards of Care in Diabetes—2026*. Diabetes Care 49(Supplement_1):S150-S165. [DOI](https://doi.org/10.2337/dc26-S007).

Role: cgm_and_aid_context; source recorded in the packaged evidence registry. Citation key: `ada_2026_diabetes_technology`.

**[3]** Battelino, Tadej; Danne, Thomas; Bergenstal, Richard M.; Amiel, Stephanie A.; Beck, Roy; Biester, Torben; et al. (2019). *Clinical Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus on Time in Range*. Diabetes Care 42(8):1593-1603. [DOI](https://doi.org/10.2337/dci19-0028).

Role: metrics_targets; source recorded in the packaged evidence registry. Citation key: `attd_2019_time_in_range`.

**[4]** Brown, Sue A.; Kovatchev, Boris P.; Raghinaru, Dan; Lum, John W.; Buckingham, Bruce A.; Kudva, Yogish C.; et al. (2019). *Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes*. New England Journal of Medicine 381(18):1707-1717. [DOI](https://doi.org/10.1056/NEJMoa1907863).

Role: aid_benchmarking; source recorded in the packaged evidence registry. Citation key: `nejm_2019_control_iq`.

**[5]** Choudhary, Pratik; Kolassa, Ralf; Keuthage, Winfried; Kroeger, Jens; Thivolet, Charles; Evans, Mark; et al. (2022). *Advanced hybrid closed loop therapy versus conventional treatment in adults with type 1 diabetes (ADAPT): a randomised controlled study*. The Lancet Diabetes & Endocrinology 10(10):720-731. [DOI](https://doi.org/10.1016/S2213-8587(22)00212-1).

Role: aid_benchmarking; source recorded in the packaged evidence registry. Citation key: `adapt_2022_ahcl`.

**[6]** Cobry, Erin; McFann, Kim; Messer, Laurel; Gage, Victoria; VanderWel, Brandon; Horton, Lauren; et al. (2010). *Timing of Meal Insulin Boluses to Achieve Optimal Postprandial Glycemic Control in Patients with Type 1 Diabetes*. Diabetes Technology & Therapeutics 12(3):173-177. [DOI](https://doi.org/10.1089/dia.2009.0112).

Role: prebolus_timing; source recorded in the packaged evidence registry. Citation key: `cobry_2010_meal_bolus_timing`.

**[7]** Dalla Man, Chiara; Rizza, Robert A.; Cobelli, Claudio. (2007). *Meal Simulation Model of the Glucose-Insulin System*. IEEE Transactions on Biomedical Engineering 54(10):1740-1749. [DOI](https://doi.org/10.1109/TBME.2007.893506).

Role: virtual_patient_dynamics; source recorded in the packaged evidence registry. Citation key: `dalla_man_2007_meal_model`.

**[8]** Visentin, Roberto; Campos-Náñez, Enrique; Schiavon, Michele; Lv, Dayu; Vettoretti, Martina; Breton, Marc; et al. (2018). *The UVA/Padova Type 1 Diabetes Simulator Goes From Single Meal to Single Day*. Journal of Diabetes Science and Technology 12(2):273-281. [DOI](https://doi.org/10.1177/1932296818757747).

Role: simulation_validation; source recorded in the packaged evidence registry. Citation key: `visentin_2018_uvapadova`.

**[9]** Mujahid, Omer; Contreras, Ivan; Beneyto, Aleix; Vehi, Josep. (2024). *Generative deep learning for the development of a type 1 diabetes simulator*. Communications Medicine 4(1):51. [DOI](https://doi.org/10.1038/s43856-024-00476-0).

Role: simulation_realism; source recorded in the packaged evidence registry. Citation key: `mujahid_2024_generative_t1d_simulator`.

**[10]** Riddell, Michael C; Gallen, Ian W; Smart, Carmel E; Taplin, Craig E; Adolfsson, Peter; Lumb, Alistair N; et al. (2017). *Exercise management in type 1 diabetes: a consensus statement*. The Lancet Diabetes & Endocrinology 5(5):377-390. [DOI](https://doi.org/10.1016/S2213-8587(17)30014-1).

Role: exercise_stress_scenarios; source recorded in the packaged evidence registry. Citation key: `riddell_2017_exercise_consensus`.

**[11]** HealthPartners Institute / International Diabetes Center. (n.d.). *Guide to Understanding the Ambulatory Glucose Profile (AGP) Report*. [Source](https://www.healthpartners.com/institute/wp-content/uploads/2025/05/Ambulatory-Glucose-Profile-Report-Overview.pdf).

Role: agp_style_reporting; official resource recorded in source registry. Citation key: `idc_2025_agp_report_overview`.

**[12]** Bergman, R N; Ider, Y Z; Bowden, C R; Cobelli, C. (1979). *Quantitative estimation of insulin sensitivity*. American Journal of Physiology-Endocrinology and Metabolism 236(6):E667. [DOI](https://doi.org/10.1152/ajpendo.1979.236.6.E667).

Role: virtual_patient_dynamics; source recorded in the packaged evidence registry. Citation key: `bergman_1979_minimal_model`.

**[13]** Hovorka, Roman; Canonico, Valentina; Chassin, Ludovic J; Haueter, Ulrich; Massi-Benedetti, Massimo; Federici, Marco Orsini; et al. (2004). *Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes*. Physiological Measurement 25(4):905-920. [DOI](https://doi.org/10.1088/0967-3334/25/4/010).

Role: hovorka_patient_model; source recorded in the packaged evidence registry. Citation key: `hovorka_2004_nmpc_t1d`.

**[14]** Gerich, J; Davis, J; Lorenzi, M; Rizza, R; Bohannon, N; Karam, J; et al. (1979). *Hormonal mechanisms of recovery from insulin-induced hypoglycemia in man*. American Journal of Physiology-Endocrinology and Metabolism 236(4):E380. [DOI](https://doi.org/10.1152/ajpendo.1979.236.4.E380).

Role: endogenous_counterregulation; source recorded in the packaged evidence registry. Citation key: `gerich_1979_counterregulation`.

**[15]** Cryer, Philip E. (2013). *Mechanisms of Hypoglycemia-Associated Autonomic Failure in Diabetes*. New England Journal of Medicine 369(4):362-372. [DOI](https://doi.org/10.1056/NEJMra1215228).

Role: haaf_memory_model; source recorded in the packaged evidence registry. Citation key: `cryer_2013_haaf_mechanisms`.

**[16]** Cryer, Philip E. (2013). *Hypoglycemia-associated autonomic failure in diabetes*. Handbook of Clinical Neurology 117:295-307. [DOI](https://doi.org/10.1016/B978-0-444-53491-0.00023-7).

Role: haaf_memory_model; source recorded in the packaged evidence registry. Citation key: `cryer_2013_haaf_diabetes`.

**[17]** Lv, Dayu; Breton, Marc D.; Farhy, Leon S. (2013). *Pharmacokinetics Modeling of Exogenous Glucagon in Type 1 Diabetes Mellitus Patients*. Diabetes Technology & Therapeutics 15(11):935-941. [DOI](https://doi.org/10.1089/dia.2013.0150).

Role: exogenous_glucagon_pkpd; source recorded in the packaged evidence registry. Citation key: `lv_2013_exogenous_glucagon_pk`.

**[18]** Haidar, Ahmad; Legault, Laurent; Dallaire, Maryse; Alkhateeb, Ammar; Coriati, Adèle; Messier, Virginie; et al. (2013). *Glucose-responsive insulin and glucagon delivery (dual-hormone artificial pancreas) in adults with type 1 diabetes: a randomized crossover controlled trial*. Canadian Medical Association Journal 185(4):297-305. [DOI](https://doi.org/10.1503/cmaj.121265).

Role: dual_hormone_context; source recorded in the packaged evidence registry. Citation key: `haidar_2013_dual_hormone_ap`.

**[19]** Haidar, Ahmad; Duval, Claire; Legault, Laurent; Rabasa-Lhoret, Rémi. (2013). *Pharmacokinetics of Insulin Aspart and Glucagon in Type 1 Diabetes during Closed-Loop Operation*. Journal of Diabetes Science and Technology 7(6):1507-1512. [DOI](https://doi.org/10.1177/193229681300700610).

Role: exogenous_glucagon_pkpd; source recorded in the packaged evidence registry. Citation key: `haidar_2013_insulin_glucagon_pk`.

**[20]** Ghezzi, Chiara; Loo, Donald D. F.; Wright, Ernest M. (2018). *Physiology of renal glucose handling via SGLT1, SGLT2 and GLUT2*. Diabetologia 61(10):2087-2097. [DOI](https://doi.org/10.1007/s00125-018-4656-5).

Role: renal_glucose_clearance; source recorded in the packaged evidence registry. Citation key: `hummel_2018_renal_glucose_handling`.

**[21]** DeFronzo, Ralph A.; Hompesch, Marcus; Kasichayanula, Sreeneeranj; Liu, Xiaoni; Hong, Ying; Pfister, Marc; et al. (2013). *Characterization of Renal Glucose Reabsorption in Response to Dapagliflozin in Healthy Subjects and Subjects With Type 2 Diabetes*. Diabetes Care 36(10):3169-3176. [DOI](https://doi.org/10.2337/dc13-0387).

Role: renal_glucose_clearance; source recorded in the packaged evidence registry. Citation key: `defronzo_2013_renal_reabsorption_splay`.

**[22]** Mandelbrot, Benoit B.; Van Ness, John W. (1968). *Fractional Brownian Motions, Fractional Noises and Applications*. SIAM Review 10(4):422-437. [DOI](https://doi.org/10.1137/1010093).

Role: cgm_noise_model; source recorded in the packaged evidence registry. Citation key: `mandelbrot_1968_fractional_brownian`.

**[23]** Campbell, Peter J.; Bolli, Geremia B.; Cryer, Philip E.; Gerich, John E. (1985). *Pathogenesis of the Dawn Phenomenon in Patients with Insulin-Dependent Diabetes Mellitus*. New England Journal of Medicine 312(23):1473-1479. [DOI](https://doi.org/10.1056/NEJM198506063122302).

Role: circadian_egp_model; source recorded in the packaged evidence registry. Citation key: `campbell_1985_dawn_phenomenon`.

**[24]** Richter, Erik A.; Hargreaves, Mark. (2013). *Exercise, GLUT4, and Skeletal Muscle Glucose Uptake*. Physiological Reviews 93(3):993-1017. [DOI](https://doi.org/10.1152/physrev.00038.2012).

Role: exercise_glut4_model; source recorded in the packaged evidence registry. Citation key: `richter_2013_glut4_exercise`.

**[25]** Näslund, Erik; Bogefors, Jesper; Skogar, Staffan; Grybäck, Per; Jacobsson, Hans; Holst, Jens Juul; et al. (1999). *GLP-1 slows solid gastric emptying and inhibits  insulin, glucagon, and PYY release in humans*. American Journal of Physiology-Regulatory, Integrative and Comparative Physiology 277(3):R910-R916. [DOI](https://doi.org/10.1152/ajpregu.1999.277.3.R910).

Role: glp1_gastric_emptying_model; source recorded in the packaged evidence registry. Citation key: `naslund_1999_glp1_gastric_emptying`.

**[26]** Alkhateeb, Haneen; El Fathi, Anas; Ghanbari, Milad; Haidar, Ahmad. (2021). *Modelling glucose dynamics during moderate exercise in individuals with type 1 diabetes*. PLOS ONE 16(3):e0248280. [DOI](https://doi.org/10.1371/journal.pone.0248280).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_4`.

**[27]** Romeres, Davide; Schiavon, Michele; Basu, Ananda; Cobelli, Claudio; Basu, Rita; Dalla Man, Chiara. (2021). *Exercise effect on insulin-dependent and insulin-independent glucose utilization in healthy individuals and individuals with type 1 diabetes: a modeling study*. American Journal of Physiology-Endocrinology and Metabolism 321(1):E122-E129. [DOI](https://doi.org/10.1152/ajpendo.00084.2021).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_5`.

**[28]** Mallad, Ashwini; Hinshaw, Ling; Dalla Man, Chiara; Cobelli, Claudio; Basu, Rita; Lingineni, Ravi; et al. (2015). *Nocturnal Glucose Metabolism in Type 1 Diabetes: A Study Comparing Single Versus Dual Tracer Approaches*. Diabetes Technology & Therapeutics 17(8):587-595. [DOI](https://doi.org/10.1089/dia.2015.0011).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_6`.

**[29]** Cryer, Philip E. (2001). *Hypoglycemia-associated autonomic failure in diabetes*. American Journal of Physiology-Endocrinology and Metabolism 281(6):E1115-E1121. [DOI](https://doi.org/10.1152/ajpendo.2001.281.6.E1115).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_7`.

**[30]** Fruehwald-Schultes, Bernd; Kern, Werner; Deininger, Eva; Wellhoener, Peter; Kerner, Wolfgang; Born, Jan; et al. (1999). *Protective Effect of Insulin against Hypoglycemia-Associated Counterregulatory Failure*. The Journal of Clinical Endocrinology & Metabolism 84(5):1551-1557. [DOI](https://doi.org/10.1210/jcem.84.5.5675).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_8`.

**[31]** Wendt, Sabrina Lyngbye; Ranjan, Ajenthen; Møller, Jan Kloppenborg; Schmidt, Signe; Knudsen, Carsten Boye; Holst, Jens Juul; et al. (2017). *Cross-Validation of a Glucose-Insulin-Glucagon Pharmacodynamics Model for Simulation Using Data From Patients With Type 1 Diabetes*. Journal of Diabetes Science and Technology 11(6):1101-1111. [DOI](https://doi.org/10.1177/1932296817693254).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_10`.

**[32]** Rebrin, Kerstin; Sheppard, Norman F.; Steil, Garry M. (2010). *Use of Subcutaneous Interstitial Fluid Glucose to Estimate Blood Glucose: Revisiting Delay and Sensor Offset*. Journal of Diabetes Science and Technology 4(5):1087-1098. [DOI](https://doi.org/10.1177/193229681000400507).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_11`.

**[33]** Schmidtke, David W.; Freeland, Angela C.; Heller, Adam; Bonnecaze, Roger T. (1998). *Measurement and modeling of the transient difference between blood and subcutaneous glucose concentrations in the rat after injection of insulin*. Proceedings of the National Academy of Sciences 95(1):294-299. [DOI](https://doi.org/10.1073/pnas.95.1.294).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_12`.

**[34]** Visentin, Roberto; Dalla Man, Chiara; Kudva, Yogish C.; Basu, Ananda; Cobelli, Claudio. (2015). *Circadian Variability of Insulin Sensitivity: Physiological Input for In Silico Artificial Pancreas*. Diabetes Technology & Therapeutics 17(1):1-7. [DOI](https://doi.org/10.1089/dia.2014.0192).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_13`.

**[35]** Saad, Ahmed; Dalla Man, Chiara; Nandy, Debashis K.; Levine, James A.; Bharucha, Adil E.; Rizza, Robert A.; et al. (2012). *Diurnal Pattern to Insulin Secretion and Insulin Action in Healthy Individuals*. Diabetes 61(11):2691-2700. [DOI](https://doi.org/10.2337/db11-1478).

Role: Literature anchor explicitly attached to an equation in FORMULA_REGISTRY; not evidence that all SDK extensions reproduce the paper. Citation key: `formula_source_14`.

**[36]** Man, Chiara Dalla; Micheletto, Francesco; Lv, Dayu; Breton, Marc; Kovatchev, Boris; Cobelli, Claudio. (2014). *The UVA/PADOVA Type 1 Diabetes Simulator*. Journal of Diabetes Science and Technology 8(1):26-34. [DOI](https://doi.org/10.1177/1932296813514502).

Role: Simulator comparison/background already cited in paper.bib; not a claim of UVA/Padova equivalence. Citation key: `dallaman2014`.

**[37]** Kovatchev, Boris P; Cox, Daniel J; Gonder-Frederick, Linda A; Clarke, William. (1997). *Symmetrization of the Blood Glucose Measurement Scale and Its Applications*. Diabetes Care 20(11):1655-1658. [DOI](https://doi.org/10.2337/diacare.20.11.1655).

Role: Risk-scale methodology cited in paper.bib. Citation key: `kovatchev1997`.

**[38]** Kovatchev, B P; Cox, D J; Gonder-Frederick, L A; Young-Hyman, D; Schlundt, D; Clarke, W. (1998). *Assessment of risk for severe hypoglycemia among adults with IDDM: validation of the low blood glucose index*. Diabetes Care 21(11):1870-1875. [DOI](https://doi.org/10.2337/diacare.21.11.1870).

Role: Hypoglycemia risk-index background cited in paper.bib. Citation key: `kovatchev1998`.

**[39]** Heise, Tim; Stender-Petersen, Kirstine; Hövelmann, Ulrike; Jacobsen, Jacob Bonde; Nosek, Leszek; Zijlstra, Eric; et al. (2017). *Pharmacokinetic and Pharmacodynamic Properties of Faster-Acting Insulin Aspart versus Insulin Aspart Across a Clinically Relevant Dose Range in Subjects with Type 1 Diabetes Mellitus*. Clinical Pharmacokinetics 56(6):649-660. [DOI](https://doi.org/10.1007/s40262-016-0473-5).

Role: Verified replacement reading for the unresolved faster-aspart reference; not proof of historical use or calibration. Citation key: `heise2017_corrected_context`.

**[40]** Johansen, K.; Svendsen, P.Aaby; L�rup, B. (1984). *Variations in renal threshold for glucose in Type 1 (insulin-dependent) diabetes mellitus*. Diabetologia 26(3). [DOI](https://doi.org/10.1007/BF00252403).

Role: Renal threshold source explicitly linked in FORMULA_REGISTRY. Citation key: `johansen1984`.

**[41]** Peng, Fei; Li, Xin; Xiao, Fang; Zhao, Ruxing; Sun, Zheng. (2022). *Circadian clock, diurnal glucose metabolic rhythm, and dawn phenomenon*. Trends in Neurosciences 45(6):471-482. [DOI](https://doi.org/10.1016/j.tins.2022.03.010).

Role: Circadian physiology review explicitly linked in FORMULA_REGISTRY. Citation key: `peng2022`.

## 2. AI methods and numerical computing

**[42]** Harris, Charles R.; Millman, K. Jarrod; van der Walt, Stéfan J.; Gommers, Ralf; Virtanen, Pauli; Cournapeau, David; et al. (2020). *Array programming with NumPy*. Nature 585(7825):357-362. [DOI](https://doi.org/10.1038/s41586-020-2649-2).

Role: Numerical array software used by the SDK. Citation key: `numpy2020`.

**[43]** Virtanen, Pauli; Gommers, Ralf; Oliphant, Travis E.; Haberland, Matt; Reddy, Tyler; Cournapeau, David; et al. (2020). *SciPy 1.0: fundamental algorithms for scientific computing in Python*. Nature Methods 17(3):261-272. [DOI](https://doi.org/10.1038/s41592-019-0686-2).

Role: ODE integration and numerical optimisation software used by the SDK. Citation key: `scipy2020`.

**[44]** Hunter, John D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering 9(3):90-95. [DOI](https://doi.org/10.1109/MCSE.2007.55).

Role: Scientific plotting library used by reports. Citation key: `matplotlib2007`.

**[45]** Raissi, M.; Perdikaris, P.; Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics 378:686-707. [DOI](https://doi.org/10.1016/j.jcp.2018.10.045).

Role: Methodological background added for physiology-informed losses; constraints alone do not establish an ODE-residual PINN. Citation key: `pinn2019`.

**[46]** Nelder, J. A.; Mead, R. (1965). *A Simplex Method for Function Minimization*. The Computer Journal 7(4):308-313. [DOI](https://doi.org/10.1093/comjnl/7.4.308).

Role: Methodological citation for Nelder-Mead optimisation; parameter fitting is not proof of identifiability. Citation key: `neldermead1965`.

**[47]** Li, Zechen; Natarajan, Keerthana; Zhang, Weizhi; Zhou, Menglian; Lee, Simon A.; Zhang, Yuwei; et al. (2026). *GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring*. arXiv. [Source](https://arxiv.org/abs/2605.30865v2).

Role: Version 2 preprint (25 August 2026), explicitly referenced by glucofm.py. IINTS implements an independent method reproduction, not official model weights. Citation key: `glucofm2026v2`.

## 3. Structural biology, regenerative islets and proteomics

**[48]** Gilson, Michael K.; Liu, Tiqing; Baitaluk, Michael; Nicola, George; Hwang, Linda; Chong, Jenny. (2016). *BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology*. Nucleic Acids Research 44(D1):D1045-D1053. [DOI](https://doi.org/10.1093/nar/gkv1072).

Role: measured_binding_affinity_evidence; source recorded in the packaged evidence registry. Citation key: `bindingdb_2016`.

**[49]** NCBI ClinVar. (n.d.). *Representation of classifications in ClinVar*. [Source](https://www.ncbi.nlm.nih.gov/clinvar/docs/clinsig/).

Role: genetic_variant_context; official resource recorded in source registry. Citation key: `ncbi_clinvar_classification_representation`.

**[50]** Jumper, John; Evans, Richard; Pritzel, Alexander; Green, Tim; Figurnov, Michael; Ronneberger, Olaf; et al. (2021). *Highly accurate protein structure prediction with AlphaFold*. Nature 596(7873):583-589. [DOI](https://doi.org/10.1038/s41586-021-03819-2).

Role: AlphaFold method underlying the predicted structures displayed in the app. Citation key: `jumper2021`.

**[51]** Varadi, Mihaly; Bertoni, Damian; Magana, Paulyna; Paramval, Urmila; Pidruchna, Ivanna; Radhakrishnan, Malarvizhi; et al. (2024). *AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences*. Nucleic Acids Research 52(D1):D368-D375. [DOI](https://doi.org/10.1093/nar/gkad1011).

Role: AlphaFold DB data provenance; local confidence is not functional effect size. Citation key: `varadi2024`.

**[52]** Reichman, Trevor W.; Markmann, James F.; Odorico, Jon; Witkowski, Piotr; Fung, John J.; Wijkstrom, Martin; et al. (2025). *Stem Cell–Derived, Fully Differentiated Islets for Type 1 Diabetes*. New England Journal of Medicine 393(9):858-868. [DOI](https://doi.org/10.1056/NEJMoa2506549).

Role: Clinical context cited in REGENERATIVE_ISLET_RESEARCH; does not validate the SDK graft simulator. Citation key: `reichman2025`.

**[53]** Carlsson, Per-Ola; Hu, Xiaomeng; Scholz, Hanne; Ingvast, Sofie; Lundgren, Torbjörn; Scholz, Tim; et al. (2025). *Survival of Transplanted Allogeneic Beta Cells with No Immunosuppression*. New England Journal of Medicine 393(9):887-894. [DOI](https://doi.org/10.1056/NEJMoa2503822).

Role: Transplantation context cited in regenerative research documentation; not a cure claim. Citation key: `carlsson2025`.

**[54]** Balboa, Diego; Barsby, Tom; Lithovius, Väinö; Saarimäki-Vire, Jonna; Omar-Hmeadi, Muhmmad; Dyachok, Oleg; et al. (2022). *Functional, metabolic and transcriptional maturation of human pancreatic islets derived from stem cells*. Nature Biotechnology 40(7):1042-1055. [DOI](https://doi.org/10.1038/s41587-022-01219-z).

Role: Islet maturation context explicitly cited in regenerative documentation. Citation key: `balboa2022`.

**[55]** Augsornworawat, Punn; Hogrebe, Nathaniel J.; Ishahak, Matthew; Schmidt, Mason D.; Marquez, Erica; Maestas, Marlie M.; et al. (2023). *Single-nucleus multi-omics of human stem cell-derived islets identifies deficiencies in lineage specification*. Nature Cell Biology 25(6):904-916. [DOI](https://doi.org/10.1038/s41556-023-01150-8).

Role: Multi-omics and lineage-specification context in regenerative documentation. Citation key: `augsornworawat2023`.

**[56]** Cox, Jürgen; Mann, Matthias. (2008). *MaxQuant enables high peptide identification rates, individualized p.p.b.-range mass accuracies and proteome-wide protein quantification*. Nature Biotechnology 26(12):1367-1372. [DOI](https://doi.org/10.1038/nbt.1511).

Role: MaxQuant output-format provenance; the importer reads results, not a reimplementation of MaxQuant. Citation key: `maxquant2008`.

**[57]** Demichev, Vadim; Messner, Christoph B.; Vernardis, Spyros I.; Lilley, Kathryn S.; Ralser, Markus. (2020). *DIA-NN: neural networks and interference correction enable deep proteome coverage in high throughput*. Nature Methods 17(1):41-44. [DOI](https://doi.org/10.1038/s41592-019-0638-x).

Role: DIA-NN quantification method underlying supported imported tables. Citation key: `diann2020`.

**[58]** Sintov, Elad; Nikolskiy, Igor; Barrera, Victor; Hyoje-Ryu Kenty, Jennifer; Atkin, Alexander S.; Gerace, Dario; et al. (2022). *Whole-genome CRISPR screening identifies genetic manipulations to reduce immune rejection of stem cell-derived islets*. Stem Cell Reports 17(9):1976-1990. [DOI](https://doi.org/10.1016/j.stemcr.2022.08.002).

Role: Stem-cell-derived islet immune-rejection study linked in regenerative documentation; not evidence of efficacy of an SDK intervention. Citation key: `sintov2022`.

**[59]** AlphaFold Protein Structure Database maintainers. (n.d.). *AlphaFold Protein Structure Database — official resource and documentation*. [Source](https://alphafold.ebi.ac.uk/).

Role: Desktop registry status: Bundled 3D assets + PAE evidence. pLDDT and PAE describe prediction confidence, not pathogenicity or metabolic severity. Citation key: `resource_alphafold-db`.

**[60]** Ensembl VEP + AlphaMissense maintainers. (n.d.). *Ensembl VEP + AlphaMissense — official resource and documentation*. [Source](https://rest.ensembl.org/).

Role: Desktop registry status: Recommended next live connector. External evidence requires claim-level human review before publication. Citation key: `resource_ensembl-vep-alphamissense`.

**[61]** Open Targets Platform maintainers. (n.d.). *Open Targets Platform — official resource and documentation*. [Source](https://platform.opentargets.org/).

Role: Desktop registry status: Recommended evidence graph connector. External evidence requires claim-level human review before publication. Citation key: `resource_open-targets`.

**[62]** Reactome maintainers. (n.d.). *Reactome — official resource and documentation*. [Source](https://reactome.org/).

Role: Desktop registry status: Recommended pathway connector. External evidence requires claim-level human review before publication. Citation key: `resource_reactome`.

**[63]** RCSB PDB maintainers. (n.d.). *RCSB PDB — official resource and documentation*. [Source](https://www.rcsb.org/).

Role: Desktop registry status: Recommended structure validation connector. External evidence requires claim-level human review before publication. Citation key: `resource_rcsb-pdb`.

**[64]** UniProt maintainers. (n.d.). *UniProt — official resource and documentation*. [Source](https://www.uniprot.org/).

Role: Desktop registry status: Recommended protein summary connector. External evidence requires claim-level human review before publication. Citation key: `resource_uniprot`.

**[65]** Human Protein Atlas maintainers. (n.d.). *Human Protein Atlas — official resource and documentation*. [Source](https://www.proteinatlas.org/).

Role: Desktop registry status: Recommended expression evidence connector. External evidence requires claim-level human review before publication. Citation key: `resource_human-protein-atlas`.

**[66]** GTEx Portal API maintainers. (n.d.). *GTEx Portal API — official resource and documentation*. [Source](https://gtexportal.org/).

Role: Desktop registry status: Partially integrated via expression renders. Expression context informs interpretation but does not automatically set physiology parameters. Citation key: `resource_gtex`.

**[67]** ChEMBL maintainers. (n.d.). *ChEMBL — official resource and documentation*. [Source](https://www.ebi.ac.uk/chembl/).

Role: Desktop registry status: Partially integrated molecule lookup. Chemical identity and literature context do not validate a PK parameter without manual review. Citation key: `resource_chembl`.

**[68]** ClinPGx / PharmGKB maintainers. (n.d.). *ClinPGx / PharmGKB — official resource and documentation*. [Source](https://api.pharmgkb.org/).

Role: Desktop registry status: Recommended cautious connector. External evidence requires claim-level human review before publication. Citation key: `resource_clinpgx-pharmgkb`.

**[69]** OpenCOR + Physiome Model Repository maintainers. (n.d.). *OpenCOR + Physiome Model Repository — official resource and documentation*. [Source](https://models.physiomeproject.org/cellml).

Role: Desktop registry status: Integrated CellML inspection and OpenCOR validation. Repository curation, imports, units, and biological suitability must be checked per model; OpenCOR validation is not biological validation. Citation key: `resource_opencor-physiome`.

**[70]** BindingDB maintainers. (n.d.). *BindingDB — official resource and documentation*. [Source](https://www.bindingdb.org/rwd/bind/index.jsp).

Role: Desktop registry status: Integrated read-only UniProt affinity evidence. Kd, Ki, and IC50 are assay-specific and cannot be treated as interchangeable physiological effects. Citation key: `resource_bindingdb`.

**[71]** STRING DB maintainers. (n.d.). *STRING DB — official resource and documentation*. [Source](https://string-db.org/).

Role: Desktop registry status: Partially integrated via pathway renders. External evidence requires claim-level human review before publication. Citation key: `resource_string-db`.

**[72]** ClinVar / NCBI Clinical Tables maintainers. (n.d.). *ClinVar / NCBI Clinical Tables — official resource and documentation*. [Source](https://www.ncbi.nlm.nih.gov/clinvar/).

Role: Desktop registry status: Partially integrated via mutation examples. ClinVar assertions can conflict or change and never become a physiological scalar automatically. Citation key: `resource_clinvar`.

**[73]** FAIR Principles for Research Software (FAIR4RS) maintainers. (n.d.). *FAIR Principles for Research Software (FAIR4RS) — official resource and documentation*. [Source](https://www.nature.com/articles/s41597-022-01710-x).

Role: Desktop registry status: Implemented as project guidance and audit criteria. The SDK is FAIR-oriented; this label is not an external FAIR certification. Citation key: `resource_fair4rs`.

**[74]** PubMed maintainers. (n.d.). *PubMed — official resource and documentation*. [Source](https://pubmed.ncbi.nlm.nih.gov/).

Role: Desktop registry status: Official literature portal; structured capture planned. External evidence requires claim-level human review before publication. Citation key: `resource_pubmed`.

**[75]** ClinicalTrials.gov maintainers. (n.d.). *ClinicalTrials.gov — official resource and documentation*. [Source](https://clinicaltrials.gov/).

Role: Desktop registry status: Official registry link; no automated inference. External evidence requires claim-level human review before publication. Citation key: `resource_clinicaltrials-gov`.

**[76]** Zenodo maintainers. (n.d.). *Zenodo — official resource and documentation*. [Source](https://zenodo.org/).

Role: Desktop registry status: Manual deposit only. Manual review is mandatory before any deposit; no credentials or uploads are handled by this connector. Citation key: `resource_zenodo`.

**[77]** PRIDE Archive. (n.d.). *PRIDE Archive*. [Source](https://www.ebi.ac.uk/pride/).

Role: Proteomics source repository used by the importer workflow. Citation key: `pride`.

**[78]** ProteomeXchange. (n.d.). *ProteomeXchange*. [Source](https://www.proteomexchange.org/).

Role: Dataset accession and provenance framework. Citation key: `proteomexchange`.

**[79]** PRIDE / ProteomeXchange dataset PXD001539. (n.d.). *PRIDE / ProteomeXchange dataset PXD001539*. [Source](https://www.ebi.ac.uk/pride/archive/projects/PXD001539).

Role: Named comparator source in regenerative documentation; actual analysed samples require a run manifest. Citation key: `pxd001539`.

**[80]** PRIDE / ProteomeXchange dataset PXD064528. (n.d.). *PRIDE / ProteomeXchange dataset PXD064528*. [Source](https://www.ebi.ac.uk/pride/archive/projects/PXD064528).

Role: Named comparator source in regenerative documentation; do not assume this dataset was used in every run. Citation key: `pxd064528`.

**[81]** Spectronaut documentation. (n.d.). *Spectronaut documentation*. [Source](https://biognosys.com/software/spectronaut/).

Role: Imported output-format context; not bundled software. Citation key: `spectronaut`.

**[82]** ClinicalTrials.gov. (n.d.). *ClinicalTrials.gov: NCT04786262*. [Source](https://clinicaltrials.gov/study/NCT04786262).

Role: Trial record explicitly linked in regenerative documentation. Citation key: `trial_zimislecel`.

## 4. Model interchange and reproducible research

**[83]** Hucka, Michael; Bergmann, Frank T.; Chaouiya, Claudine; Dräger, Andreas; Hoops, Stefan; Keating, Sarah M.; et al. (2019). *The Systems Biology Markup Language (SBML): Language Specification for Level 3 Version 2 Core Release 2*. Journal of Integrative Bioinformatics 16(2). [DOI](https://doi.org/10.1515/jib-2019-0021).

Role: mechanistic_reference_model_interchange; source recorded in the packaged evidence registry. Citation key: `sbml_2019_l3v2_core`.

**[84]** Somogyi, Endre T.; Bouteiller, Jean-Marie; Glazier, James A.; König, Matthias; Medley, J. Kyle; Swat, Maciej H.; et al. (2015). *libRoadRunner: a high performance SBML simulation and analysis library*. Bioinformatics 31(20):3315-3321. [DOI](https://doi.org/10.1093/bioinformatics/btv363).

Role: mechanistic_reference_model_execution; source recorded in the packaged evidence registry. Citation key: `libroadrunner_2015`.

**[85]** Hoops, Stefan; Sahle, Sven; Gauges, Ralph; Lee, Christine; Pahle, Jürgen; Simus, Natalia; et al. (2006). *COPASI—a COmplex PAthway SImulator*. Bioinformatics 22(24):3067-3074. [DOI](https://doi.org/10.1093/bioinformatics/btl485).

Role: sensitivity_and_parameter_analysis; source recorded in the packaged evidence registry. Citation key: `copasi_2006`.

**[86]** Garny, Alan; Hunter, Peter J. (2015). *OpenCOR: a modular and interoperable approach to computational biology*. Frontiers in Physiology 6. [DOI](https://doi.org/10.3389/fphys.2015.00026).

Role: cellml_reference_validation; source recorded in the packaged evidence registry. Citation key: `opencor_2015`.

**[87]** Modelica Association Project FMI. (n.d.). *Functional Mock-up Interface Specification 3.0.2*. [Source](https://fmi-standard.org/docs/3.0.2/).

Role: device_physics_model_interchange; official resource recorded in source registry. Citation key: `fmi_3_0_2_standard`.

**[88]** Sommer T. (n.d.). *FMPy: Simulate Functional Mock-up Units in Python*. [Source](https://fmpy.readthedocs.io/en/latest/).

Role: device_physics_execution; official resource recorded in source registry. Citation key: `fmpy_0_3`.

**[89]** MyVariant.info. (n.d.). *MyVariant.info v1 query API*. [Source](https://docs.myvariant.info/en/latest/doc/quick_start.html).

Role: genetic_variant_context_transport; official resource recorded in source registry. Citation key: `myvariant_query_api`.

**[90]** Wilkinson, Mark D.; Dumontier, Michel; Aalbersberg, IJsbrand Jan; Appleton, Gabrielle; Axton, Myles; Baak, Arie; et al. (2016). *The FAIR Guiding Principles for scientific data management and stewardship*. Scientific Data 3(1):160018. [DOI](https://doi.org/10.1038/sdata.2016.18).

Role: FAIR research-data framing; not a certification of project compliance. Citation key: `fair2016`.

**[91]** RO-Crate 1.2 maintainers. (n.d.). *RO-Crate 1.2 — official resource and documentation*. [Source](https://www.researchobject.org/ro-crate/).

Role: Desktop registry status: Integrated academic package export. The export is RO-Crate-compatible metadata; it is not a repository deposit or peer review. Citation key: `resource_ro-crate`.

**[92]** BioModels maintainers. (n.d.). *BioModels — official resource and documentation*. [Source](https://www.biomodels.org/).

Role: Desktop registry status: Official model library; import planned. BioModels is migrating to BioStudies; the workbench does not yet import or execute deposited models. Citation key: `resource_biomodels`.

**[93]** libRoadRunner maintainers. (n.d.). *libRoadRunner — official resource and documentation*. [Source](https://libroadrunner.readthedocs.io/).

Role: Desktop registry status: Integrated optional SBML execution engine. Execution success is not biological validation; model units and population assumptions require review. Citation key: `resource_libroadrunner`.

**[94]** COPASI maintainers. (n.d.). *COPASI — official resource and documentation*. [Source](https://copasi.org/).

Role: Desktop registry status: Integrated task inspection; explicit CopasiSE execution. IINTS never generates an objective or silently imports fitted parameters into a patient model. Citation key: `resource_copasi`.

**[95]** SED-ML maintainers. (n.d.). *SED-ML — official resource and documentation*. [Source](https://sed-ml.org/).

Role: Desktop registry status: Planned standards-based simulation export. No SED-ML file is generated yet; current reproducibility uses IINTS run metadata and RO-Crate. Citation key: `resource_sed-ml`.

**[96]** SBML maintainers. (n.d.). *SBML — official resource and documentation*. [Source](https://sbml.org/).

Role: Desktop registry status: Integrated local inspection; optional execution. Current IINTS patient models remain Python implementations and are not claimed to be SBML-compatible; external model units and assumptions require explicit review. Citation key: `resource_sbml`.

## 5. Data sources and acquisition catalogues

Catalogue membership is not experimental use. OhioT1DM is documented in the calibration workflow; all actual training/test cohorts still require run-level provenance. Restricted data are not redistributed here.

**[97]** Marling, Cindy; Bunescu, Razvan. (2020). *The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020*. Proceedings of the 5th International Workshop on Knowledge Discovery in Healthcare Data; CEUR Workshop Proceedings 2675:71--74. [Source](https://ceur-ws.org/Vol-2675/paper11.pdf).

Role: CGM data preparation, retrospective evaluation and calibration context. Data access is governed by a Data Use Agreement. Citation key: `marling_2020_ohiot1dm`.

**[98]** IINTS-AF. (n.d.). *IINTS Sample CGM (Bundled)*. [Source](https://github.com/python35/IINTS-SDK).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: bundled. Check original licence and run provenance. Citation key: `dataset_sample`.

**[99]** Dartmouth College / Scientific Data. (n.d.). *DiaTrend Dataset*. [Source](https://doi.org/10.7303/syn38187184).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: request. Check original licence and run provenance. Citation key: `dataset_diatrend`.

**[100]** University of Manchester / Zenodo. (n.d.). *T1D-UOM Longitudinal Multimodal Dataset*. [Source](https://doi.org/10.5281/zenodo.15806142).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: manual. Check original licence and run provenance. Citation key: `dataset_t1d_uom`.

**[101]** University of Granada / Zenodo. (n.d.). *T1DiabetesGranada Dataset*. [Source](https://doi.org/10.5281/zenodo.10050944).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: request. Check original licence and run provenance. Citation key: `dataset_t1d_granada`.

**[102]** Jaeb Center for Health Research. (n.d.). *AIDE T1D Public Dataset*. [Source](https://public.jaeb.org/datasets/).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: public-download. Check original licence and run provenance. Citation key: `dataset_aide_t1d`.

**[103]** Mendeley Data. (n.d.). *AZT1D: A Real-World Dataset for Type 1 Diabetes*. [Source](https://data.mendeley.com/datasets/gk9m674wcx/1).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: manual. Check original licence and run provenance. Citation key: `dataset_azt1d`.

**[104]** Mendeley Data. (n.d.). *HUPA-UCM Diabetes Dataset*. [Source](https://data.mendeley.com/datasets/3hbcscwz44/1).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: manual. Check original licence and run provenance. Citation key: `dataset_hupa_ucm`.

**[105]** OpenAPS. (n.d.). *OpenAPS Data Commons*. [Source](https://openaps.org/outcomes/data-commons/).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: request. Check original licence and run provenance. Citation key: `dataset_openaps_data_commons`.

**[106]** Tidepool. (n.d.). *Tidepool Big Data Donation*. [Source](https://www.tidepool.org/bigdata).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: request. Check original licence and run provenance. Citation key: `dataset_tidepool_bigdata`.

**[107]** NIDDK. (n.d.). *NIDDK Central Repository*. [Source](https://repository.niddk.nih.gov/).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: request. Check original licence and run provenance. Citation key: `dataset_niddk_central`.

**[108]** Jaeb Center / T1D Exchange. (n.d.). *T1D Exchange Clinic Registry*. [Source](https://datacatalog.med.nyu.edu/dataset/10129).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: request. Check original licence and run provenance. Citation key: `dataset_t1d_exchange`.

**[109]** Zenodo / AISLab HES-SO. (n.d.). *D1NAMO Open Dataset*. [Source](https://zenodo.org/records/1421616).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: manual. Check original licence and run provenance. Citation key: `dataset_d1namo`.

**[110]** Jaeb Center / Vivli / Public Study Websites. (n.d.). *Type 1 Diabetes Exercise Initiative (T1DEXI)*. [Source](https://public.jaeb.org/datasets/diabetes).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: manual. Check original licence and run provenance. Citation key: `dataset_t1dexi`.

**[111]** NIDDK Central Repository / Jaeb Center. (n.d.). *International Diabetes Closed Loop Trial (DCLP3 / iDCL)*. [Source](https://repository.niddk.nih.gov/studies/dclp3/).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: public-download. Check original licence and run provenance. Citation key: `dataset_dclp3_idcl`.

**[112]** MetaboNet / arXiv. (n.d.). *MetaboNet Consolidated T1D Dataset*. [Source](https://metabo-net.org/).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: mixed. Check original licence and run provenance. Citation key: `dataset_metabonet`.

**[113]** arXiv / Glucose-ML authors. (n.d.). *Glucose-ML Dataset Collection*. [Source](https://arxiv.org/abs/2507.14077).

Role: Dataset catalogue entry, NOT proof of use in training. Access mode recorded by SDK: collection. Check original licence and run provenance. Citation key: `dataset_glucose_ml`.

## 6. Software, hardware and implementation documentation

**[114]** Python Software Foundation. (n.d.). *Python*. [Source](https://docs.python.org/3/).

Role: SDK runtime Citation key: `python`.

**[115]** The Rust Project. (n.d.). *Rust*. [Source](https://www.rust-lang.org/).

Role: Native command boundary Citation key: `rust`.

**[116]** Tauri contributors. (n.d.). *Tauri 2*. [Source](https://v2.tauri.app/).

Role: Desktop shell and signed updater Citation key: `tauri`.

**[117]** The Qt Company. (n.d.). *Qt for Python / PySide6*. [Source](https://doc.qt.io/qtforpython-6/).

Role: Earlier/optional desktop interface Citation key: `pyside`.

**[118]** Ollama. (n.d.). *Ollama documentation*. [Source](https://docs.ollama.com/).

Role: Optional local-language-model service Citation key: `ollama`.

**[119]** Mistral AI. (n.d.). *Mistral AI model documentation*. [Source](https://docs.mistral.ai/).

Role: Model-family/runtime context; record exact model ID and digest per experiment Citation key: `mistral`.

**[120]** Hugging Face. (n.d.). *Hugging Face Hub documentation*. [Source](https://huggingface.co/docs/hub/).

Role: Checkpoint/dataset distribution and model cards Citation key: `huggingface`.

**[121]** PyTorch contributors. (n.d.). *PyTorch documentation*. [Source](https://pytorch.org/docs/stable/).

Role: Neural model implementation Citation key: `pytorch`.

**[122]** pandas contributors. (n.d.). *pandas documentation*. [Source](https://pandas.pydata.org/docs/).

Role: Tabular data processing Citation key: `pandas`.

**[123]** Plotly. (n.d.). *Plotly Python documentation*. [Source](https://plotly.com/python/).

Role: Interactive figures Citation key: `plotly`.

**[124]** SciencePlots contributors. (n.d.). *SciencePlots*. [Source](https://github.com/garrettj403/SciencePlots).

Role: Plot styles, not journal endorsement Citation key: `scienceplots`.

**[125]** seaborn contributors. (n.d.). *seaborn documentation*. [Source](https://seaborn.pydata.org/).

Role: Statistical plot styling Citation key: `seaborn`.

**[126]** fpdf2 contributors. (n.d.). *fpdf2 documentation*. [Source](https://py-pdf.github.io/fpdf2/).

Role: PDF reports Citation key: `fpdf2`.

**[127]** Schrodinger; PyMOL contributors. (n.d.). *PyMOL*. [Source](https://pymol.org/).

Role: Protein rendering workflow Citation key: `pymol`.

**[128]** OpenAPS community. (n.d.). *OpenAPS documentation*. [Source](https://openaps.readthedocs.io/).

Role: Open-source insulin/COB/IOB algorithm context; not a validation of IINTS safety logic Citation key: `openaps`.

**[129]** Xie, Jinyu. (n.d.). *simglucose*. [Source](https://github.com/jxx123/simglucose).

Role: Simulator reference already cited in paper.bib Citation key: `simglucose`.

**[130]** Raspberry Pi Ltd. (n.d.). *Raspberry Pi documentation*. [Source](https://www.raspberrypi.com/documentation/).

Role: Pi/Pico hardware and interface context Citation key: `raspberry`.

**[131]** Arduino. (n.d.). *Arduino documentation*. [Source](https://docs.arduino.cc/).

Role: Microcontroller bridge context Citation key: `arduino`.

**[132]** NVIDIA. (n.d.). *NVIDIA Jetson documentation*. [Source](https://developer.nvidia.com/embedded/learn).

Role: Edge-training/deployment environment Citation key: `jetson`.

**[133]** MkDocs contributors. (n.d.). *MkDocs*. [Source](https://www.mkdocs.org/).

Role: Documentation build Citation key: `mkdocs`.

**[134]** squidfunk; contributors. (n.d.). *Material for MkDocs*. [Source](https://squidfunk.github.io/mkdocs-material/).

Role: Documentation theme Citation key: `material`.

**[135]** Mermaid contributors. (n.d.). *Mermaid documentation*. [Source](https://mermaid.js.org/).

Role: Architecture diagrams Citation key: `mermaid`.

**[136]** pytest contributors. (n.d.). *pytest documentation*. [Source](https://docs.pytest.org/).

Role: Automated software tests Citation key: `pytest`.

**[137]** Hypothesis contributors. (n.d.). *Hypothesis documentation*. [Source](https://hypothesis.readthedocs.io/).

Role: Property-based software tests Citation key: `hypothesis`.

**[138]** Git contributors. (n.d.). *Git documentation*. [Source](https://git-scm.com/doc).

Role: Version control Citation key: `git`.

**[139]** GitHub. (n.d.). *GitHub Actions documentation*. [Source](https://docs.github.com/en/actions).

Role: Build and release automation Citation key: `github`.

## 7. Direct software dependency inventory

This appendix complements the selected software citations. It lists declared Python dependencies, including optional groups, not all transitive packages and not a claim that every package was installed for every run. Obtain a lockfile or environment export for exact versions.

| Package / official distribution record | Declared use groups |
| --- | --- |
| [aiohttp](https://pypi.org/project/aiohttp/) | nightscout |
| [certifi](https://pypi.org/project/certifi/) | base |
| [cryptography](https://pypi.org/project/cryptography/) | base, mdmp |
| [defusedxml](https://pypi.org/project/defusedxml/) | base |
| [fastapi](https://pypi.org/project/fastapi/) | base |
| [flake8](https://pypi.org/project/flake8/) | dev |
| [FMPy](https://pypi.org/project/FMPy/) | fmi, tauri-engine, desktop-all |
| [fpdf2](https://pypi.org/project/fpdf2/) | reports, full, tauri-engine, desktop-all |
| [h5py](https://pypi.org/project/h5py/) | research, desktop-all |
| [httpx](https://pypi.org/project/httpx/) | dev |
| [hypothesis](https://pypi.org/project/hypothesis/) | dev |
| [libroadrunner](https://pypi.org/project/libroadrunner/) | mechanistic, tauri-engine, desktop-all |
| [matplotlib](https://pypi.org/project/matplotlib/) | reports, full, tauri-engine, desktop-all |
| [mypy](https://pypi.org/project/mypy/) | dev |
| [numpy](https://pypi.org/project/numpy/) | base |
| [onnx](https://pypi.org/project/onnx/) | research, desktop-all |
| [onnxscript](https://pypi.org/project/onnxscript/) | research, desktop-all |
| [openpyxl](https://pypi.org/project/openpyxl/) | reports, full, tauri-engine, desktop-all |
| [packaging](https://pypi.org/project/packaging/) | base |
| [pandas](https://pypi.org/project/pandas/) | base |
| [pandas-stubs](https://pypi.org/project/pandas-stubs/) | dev |
| [pillow](https://pypi.org/project/pillow/) | reports, full, tauri-engine, desktop-all |
| [plotly](https://pypi.org/project/plotly/) | research, desktop, desktop-qt, tauri-engine, desktop-all |
| [py-nightscout](https://pypi.org/project/py-nightscout/) | nightscout |
| [pyarrow](https://pypi.org/project/pyarrow/) | research, desktop-all |
| [pydantic](https://pypi.org/project/pydantic/) | base |
| [pyinstaller](https://pypi.org/project/pyinstaller/) | desktop, desktop-qt, desktop-macos, desktop-all |
| [pyobjc-framework-Cocoa](https://pypi.org/project/pyobjc-framework-Cocoa/) | desktop-macos, desktop-all |
| [pyserial](https://pypi.org/project/pyserial/) | edge, full, desktop-all |
| [PySide6](https://pypi.org/project/PySide6/) | desktop, desktop-qt, desktop-all |
| [pytest](https://pypi.org/project/pytest/) | dev |
| [PyYAML](https://pypi.org/project/PyYAML/) | base |
| [rich](https://pypi.org/project/rich/) | base |
| [SciencePlots](https://pypi.org/project/SciencePlots/) | reports, full, tauri-engine, desktop-all |
| [scipy](https://pypi.org/project/scipy/) | base |
| [seaborn](https://pypi.org/project/seaborn/) | reports, full, tauri-engine, desktop-all |
| [setuptools](https://pypi.org/project/setuptools/) | desktop, desktop-qt, desktop-macos, desktop-all |
| [starlette](https://pypi.org/project/starlette/) | base |
| [torch](https://pypi.org/project/torch/) | torch, research, desktop-all |
| [typer](https://pypi.org/project/typer/) | base |
| [types-psutil](https://pypi.org/project/types-psutil/) | dev |
| [types-PyYAML](https://pypi.org/project/types-PyYAML/) | dev |
| [uvicorn](https://pypi.org/project/uvicorn/) | base |

## 8. Device, governance and historical references

The older device-emulation references in [SOURCE_LIBRARY.md](SOURCE_LIBRARY.md)
include manufacturer manuals and regulator identifiers. Those identifiers were
not independently resolved in this bibliography and should not be copied as
verified citations. Cite the exact manual revision actually consulted, not only a
manufacturer homepage. SDK emulation is not a reconstruction of proprietary pump algorithms.

The repository also contains governance/privacy documentation. Citing a standard
or regulation does not establish compliance. Before adding legal citations to a
paper, identify the actual intended use, jurisdiction and version assessed.

## 9. Citing IINTS itself and acknowledging assistance

Bobbaers, Rune. *IINTS-AF SDK*, version 1.5.37 [software].
[Versioned source release](https://github.com/python35/IINTS-SDK/releases/tag/v1.5.37).
For the app, separately cite [Workbench beta 0.2.14](https://github.com/python35/IINTS-SDK/releases/tag/tauri-beta-v0.2.14).
Do not invent a DOI for these releases. Check CITATION.cff against the version submitted.

Disclose any AI-assisted coding, text editing or literature discovery in an
acknowledgements/methods statement appropriate to the competition. AI output is
not a primary scientific source. This bibliography was assembled with AI assistance
and automated metadata retrieval; author review remains necessary before submission.
