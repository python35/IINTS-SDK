# Defectregister

> Status van de niet-behaalde criteria uit *IINTS-AF Wetenschappelijke toetsing*
> (12 oktober 2026, git 7f5929c1) plus de bevindingen uit de daaropvolgende
> fysiologische realisme-benchmark. Werk dit bij bij elke release: elk defect
> heeft een status, een oorzaak, een oplossing en een test die hem vastpint.
>
> Onderzoek en onderwijs. Geen klinische validatie.

## Samenvatting

| | Aantal |
|---|---|
| Niet-behaalde criteria in de audit | 15 |
| Opgelost en vastgepind met een test | 8 |
| Open (modelbeperking, gedocumenteerd) | 6 |
| Open (bron- en herkomstbeperking) | 1 |

## Opgelost

| ID | Criterium uit de audit | Voor | Na | Oorzaak en oplossing | Test |
|---|---|---|---|---|---|
| D1 | `timeline/bergman same_row_glucose_snapshot` | 1.638 mg/dL | 0.000 | De compartimentkolommen werden ná de stap gelezen en `glucose_actual_mgdl` ervóór. Alle toestandskolommen worden nu op het tijdstip van de rij vastgelegd; `interval_end_minutes` benoemt het interval expliciet en `patient_iob_units_interval_end` geeft de eindtoestand apart. | `test_d1_one_row_describes_one_instant_bergman` |
| D1 | `timeline/hovorka same_row_glucose_snapshot` | 4.888 mg/dL | 0.000 | idem | `test_d1_one_row_describes_one_instant_hovorka` |
| D1 | `timeline/advanced same_row_glucose_snapshot` | 2.186 mg/dL | 0.000 | idem | `verify_contracts.py` (D1, advanced) |
| D2 | `timeline/bergman exact_duration_integration` | 125 min | 120 min | De lus liep `while current_time <= duration_minutes` en voerde een extra interval uit. De lus stopt nu vóór het eindpunt, een niet-deelbaar laatste interval wordt ingekort, en `endpoint_observation()` levert de eindtoestand los van de intervalafgifte. | `test_d2_run_integrates_exactly_the_requested_horizon` |
| D2 | `timeline/hovorka exact_duration_integration` | 125 min | 120 min | idem | idem (geparametriseerd) |
| D2 | `timeline/advanced exact_duration_integration` | 125 min | 120 min | idem | idem |
| D3 | `controller/pid pid_sample_period` | factor 11.72 | 2.3e-14 % | De integraal groeide per aanroep en de afgeleide was een kaal verschil, dus de gains hingen aan de sampletijd. Beide termen rekenen nu per referentie-interval van 5 minuten, met trapeziumintegratie. | `test_d3_pid_hourly_dose_is_portable_across_sample_times` |
| D4 | `pump/all cap_and_quantization_compatible` | 5083 / 10.000 | 0 / 10.000 | De afgiftecap werd ná de kwantisatie toegepast en kon de dosis op een halve stap zetten. De effectieve cap wordt nu naar beneden op het stappenraster gelegd. De harde cap wordt nooit overschreden, alleen strenger. | `test_d4_every_delivery_is_a_whole_number_of_steps` |

Hermeten met `verify_contracts.py`; de uitkomsten staan in
`contract_verification.csv`.

### Gedragswijzigingen die hierbij horen

1. **Rijsemantiek.** Een rij beschrijft het halfopen interval
   `[time_minutes, interval_end_minutes)`. Een run van 120 minuten met stap 5
   levert 24 rijen in plaats van 25; de waarneming op t = 120 komt uit
   `endpoint_observation()`. `tests/core/test_safety_config.py::test_sensor_fail_soft_rate_limits_toward_incoming_value`
   vroeg 10 minuten en keek naar de rij op t = 10; die vraagt nu 15 minuten.
2. **PID op 5 minuten.** Door de trapeziumregel levert de PID bij een
   constante fout van 1 mg/dL nu 1.97 E per uur in plaats van 2.03 E, een
   verschil van 3% op het afstempunt. Dat is de prijs voor sampletijd-
   onafhankelijkheid; de gains zelf zijn ongewijzigd.
3. **Pomp.** Met cap 0.975 E en stap 0.05 E is de effectieve cap 0.95 E. Een
   cap die wél een geheel aantal stappen is, verandert niet.

## Open

| ID | Criterium | Waarde | Waarom nog open | Waar gedocumenteerd |
|---|---|---|---|---|
| M1 | `basal/custom fasting_G110.0_basal0.4/0.8/1.2` | 87.7 - 89.5 mg/dL | Het `Custom`-model heeft geen glucosebalans: er is geen term die endogene productie tegenover insulineafhankelijke opname zet, dus een geprogrammeerd basaal werkt als een permanente aftreksom. Dit is geen softwarefout maar een modelbeperking. Opties: het model als demonstratiemodel markeren en uitsluiten van therapie-experimenten, of het een echte glucosebalans geven. | `docs/PATIENT_CALIBRATION_NL.md`, realisme-benchmark B1 |
| M2 | `basal/custom fasting_G150.0_basal0.4/0.8/1.2` | 127.2 - 129.5 mg/dL | idem | idem |
| B1 | `data/all local_metadata_consistency` | 0 | De lokale Ohio-bestanden hebben tegenstrijdige metadata en geen geverifieerde herkomstketen. Dit is een bron- en gebruiksbeperking, geen codefout. Oplossing hoort bij de externe-validatiemijlpaal: geverifieerde originele data met gedocumenteerde eenheden. | auditrapport, paragraaf "Externe bronnen en herkomst" |

## Bevindingen uit de realisme-benchmark

Deze stonden niet in de oorspronkelijke audit; ze komen uit de fysiologische
toetsing van de patiëntlaag.

| ID | Bevinding | Status |
|---|---|---|
| R1 | De ingestelde ISF werd niet gerealiseerd (1 E gaf 33.9 in plaats van 50 mg/dL op Hovorka; 7.5 op Bergman) | Opgelost voor Hovorka via `from_clinical_targets()`; gemeten 49.5 mg/dL/E. Open voor Bergman/Advanced. |
| R2 | De basaalstand had geen stationair effect (spreiding 0.00 mg/dL over 0.4-1.6 E/u) | Opgelost voor Hovorka (187 mg/dL over 0.5-1.5x de afgeleide stand). Voor Bergman alleen met `insulin_action_reference="plasma_ib"`, en dan nog beperkt (17.8 mg/dL). |
| R3 | Het vastenevenwicht liep weg bij ISF ongelijk aan 50 (71 mg/dL bij ISF 40, 45 bij ISF 60) | Opgelost: gekalibreerde patiënten driften 0.00 mg/dL bij elke ISF. |
| R4 | `Bergman`/`Advanced` trekken glucose terug naar de startwaarde met tijdconstante 1/p1 ≈ 36 min, ook zonder insuline | Open, gedocumenteerd als geldigheidsbereik: niet gebruiken voor open-lus-experimenten van meerdere uren. |
| R5 | `Hovorka` verwierp alleen negatieve glucose en kon 0.0 mg/dL rapporteren | Opgelost via `min_glucose_mgdl=`; standaard blijft 0.0 voor terugwaartse compatibiliteit, zet hem expliciet in nieuwe studies. |
| R6 | Insuline-PK: plasmapiek 80 min (Bergman/Advanced) tegen 40-60 min voor aspart | Open; `insulin_absorption_tmax_minutes=` is toegevoegd om het te kunnen zetten, de standaard is ongewijzigd. |
| R7 | Maaglediging Bergman 80 min uit `carb_absorptieduur/3` tegen circa 40 min in de literatuur | Open; `gastric_emptying_minutes=` toegevoegd, standaard ongewijzigd. |

## Nazorg bij D2: afnemers die het oude contract compenseerden

Het gecorrigeerde tijdcontract legde twee afnemers bloot die de oude inclusieve
eindpuntlus *compenseerden*. Beide zijn nu gecorrigeerd; de compensatie was de
fout, niet het contract.

| ID | Afnemer | Symptoom | Oorzaak en oplossing | Gemeten na |
|---|---|---|---|---|
| D2a | `iints.jetson.endurance.EnduranceConfig.simulator_end_minutes` | De drie endurance-tests gaven 11 in plaats van 12 stappen en 1500 in plaats van 1800 s wandkloktijd | De property trok één tijdstap van de gevraagde duur af om de extra interval van de oude lus te compenseren. Nu wordt de gevraagde duur onveranderd doorgegeven. | 1 u bij dt=5: 12 stappen, horizon 60 min, 3600 s pacing. 30 min: 6 stappen, 1800 s. |
| D2b | `tools/analysis/scipy_stress_review.py::summarize` | Een normaal voltooide run werd als onvoltooid geclassificeerd | Het eindpunt is geen aparte rij; het aantal intervallen is nu `ceil(minutes / dt)`, inclusief een partiële laatste stap. De oude regressietest verwachtte nog 13 in plaats van 12 rijen voor 60 minuten. | 123 min @ dt=5 → 25/25 rijen, 100% dekking, `completed=True`; ook een duur korter dan één stap en een fractionele stap zijn getest. |
| D2c | `Simulator.endpoint_observation()` en partiële stappen | Bij 123 minuten werd 125 als eindtijd gerapporteerd | De eindtijd wordt na elke geslaagde integratiestap opgeslagen, ook vóór een live-yield en in snapshots. De controller en zijn basale veiligheidslimiet gebruiken de werkelijke intervalduur. Endurance telt het laatste interval mee en paceert tot de werkelijke eindtijd. | 0, 3 en 123 minuten getoetst op Bergman, Hovorka en Advanced; wall-clock-regressie voor 123 minuten. |

In `tools/analysis/scientific_validation_suite.py` zijn alleen de
`detail`-teksten van de twee tijdlijncriteria bijgewerkt; die beschreven nog de
oude diagnose naast een nu geslaagd criterium. De criteria zelf en hun drempels
zijn ongewijzigd, dus de suite blijft dezelfde meting doen.

## Platformdefecten (Windows)

Deze blokkeren een groene testset op Windows maar raken de fysiologische
modellen niet.

| ID | Defect | Oorzaak en oplossing | Status |
|---|---|---|---|
| P1 | `test_safety_hash_ignores_the_run_directory_but_not_the_file_name`: een rapporthash veranderde met de runmap | `_normalise_run_paths()` in `iints/validation/replay.py` herkende alleen `os.sep`, dus POSIX-paden ontsnapten aan de normalisatie op Windows (en Windows-paden op POSIX). De nieuwe `_is_run_path()` is scheidingsteken-agnostisch en accepteert zowel `/a/b` als `C:\a\b`. | Opgelost; test geslaagd |
| P2 | `test_run_edge_benchmark_returns_core_metrics`: `import resource` is Unix-only | De Windows-fallback declareert nu pointergrote proceshandles en de argument- en retourtypen van K32/psapi. Bibliotheken worden afzonderlijk geladen. | Lokaal getoetst met echte Windows-aanroepen, ook zonder de optionele psutil-import: eindige positieve piekmeting. |
| P3 | Lokale scratchmap versus Raspberry Pi-export | Lokale uitvoering gebruikt `tempfile.gettempdir()`; `export_edge_setup` geeft expliciet `/tmp/iints_edge_long_study` door voor het Linux-doel. | Lokale default- en exporttests bijgewerkt en getoetst; de export neemt geen Windows-pad meer over. |
| P4 | 5 MiB-test faalde alleen na een edge-benchmark | `runtime.is_process_alive()` gebruikte `os.kill(pid, 0)`, wat op Windows `CTRL_C_EVENT` kan versturen. Het uitvoerproces stopte met status 0xC000013A vóór het data produceerde. De Windows-route gebruikt nu OpenProcess/GetExitCodeProcess/CloseHandle zonder signalen. | De benchmark gevolgd door de 5 MiB-test slaagt gezamenlijk; een regressietest controleert liveness zonder `os.kill` en het voortbestaan van nieuwe subprocessen. |

Nog open uit de lokale testronde, met vermoedelijke oorzaak buiten de modellen:
slashvergelijkingen en Windows-paden die aan Bash worden doorgegeven, een
ongeldige file-URL en een te lange commandoregel voor Node, het ontbreken van
de bevoegdheid om een symlink te maken. De oorzaak achter de gezamenlijke
5 MiB-test is vastgesteld en gerepareerd als P4; de uitvoergrens zelf is
ongewijzigd gebleven.

## Hoe hermeten

Laatste volledige lokale verificatie: **13 oktober 2026**, Windows en Python
3.14.7 met echte SDK-afhankelijkheden: **1183 geslaagd, 16 mislukt en 20
overgeslagen vermeldingen**. De regressies voor de gerepareerde tijd-, pomp-,
controller-, geheugen- en processtatuscontracten slagen gezamenlijk. De 5 MiB-
uitvoercontrole slaagt ook na de edge-benchmark. Sommige skips betreffen hele
modules, waaronder de optionele PyTorch- en Qt-onderdelen; deze zijn niet
goedgekeurd door deze ronde.

De resterende fouten betreffen Windows-paden en slashvergelijkingen,
Bash/Node-testuitvoering, symlinkbevoegdheid en een CLI-omgevingscontrole.
De volledige suite is dus nog niet groen. De ruwe lokale uitvoer staat onder
`results/fix_contracts_20261013/`; gegenereerde resultaten blijven volgens
`.gitignore` lokaal. Deze verificatie is geen externe patiëntvalidatie.

```
python -m pytest tests/core/test_contract_regressions.py   # D1-D4
python -m pytest tests/core/test_patient_calibration.py    # R1-R3, R5
```
