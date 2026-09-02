# EUCYS Jury Playbook

Dit dossier helpt om IINTS-AF controleerbaar te presenteren. Het maakt een
strikt onderscheid tussen:

- **methodefiguren**: wat de software implementeert;
- **simulatiebewijs**: resultaten van een herhaalbare in-silico run;
- **datasetbewijs**: analyses van een geïdentificeerde databron;
- **wet-labbewijs**: experimentele metingen die buiten de SDK zijn verzameld;
- **niet beschikbaar**: een ontbrekend resultaat blijft zichtbaar ontbreken.

IINTS-AF is onderzoeks- en onderwijssoftware. Het is geen medisch hulpmiddel en
de output mag niet worden gebruikt voor behandeling of insulinedosering.

## Kernverhaal

1. Een scenario beschrijft een virtuele patiënt, maaltijden, beweging,
   sensorgedrag en kandidaat-algoritme.
2. De simulator berekent de toestand met vaste software en numerieke modellen.
3. AI mag voorspellingen of verklaringen leveren, maar berekent geen
   gezaghebbende fysiologische formule en omzeilt de safety supervisor niet.
4. De evaluatielaag bewaart invoer, configuratie, seed, modelcheckpoint,
   datasplit en meetwaarden.
5. Een figuur wordt alleen als resultaat getoond wanneer het bijbehorende
   bewijsbestand is aangeleverd.

## GlucoFM-status

De SDK bevat een **onafhankelijke, paper-aligned reproductie** van GlucoFM v2.
Dit is geen officiële Google-implementatie en bevat geen officieel Google-
checkpoint.

De geïmplementeerde methode gebruikt:

- een vast venster van `288` CGM-posities met intervallen van vijf minuten;
- een fysieke observation mask zonder dichte interpolatie van ontbrekende data;
- een causale, mask-aware Gaussian filter voor state/event-scheiding;
- `24` patches van `12` posities in beide streams;
- state- en eventtokens van `64D`;
- fusie naar `128D` tokens;
- drie transformerlagen, vier attention heads en een feed-forward-dimensie van
  `256`;
- masked-context regression, temporal-dynamics loss en een EMA target encoder.

Een embedding voor onderzoek vereist een getraind checkpoint met provenance.
Ongetrainde willekeurige gewichten worden standaard geweigerd.

## Evidencecontracten

### Foundation-modelvergelijking

Elke modelrun gebruikt schema `iints.foundation-arena.evaluation.v1` en bevat
minstens:

- modelnaam, architectuur en latent dimension;
- SHA-256 van het checkpoint;
- benchmark-, cohort- en split-ID;
- aantal groepen en samples;
- bevestiging dat de evaluatie group-disjoint is;
- metricwaarde, eenheid en optimalisatierichting;
- SHA-256 van het evaluatiebestand.

De modellen worden alleen gerangschikt als hun benchmark-ID en metricdefinities
overeenkomen.

```bash
iints research foundation-arena \
  --result results/evaluations/glucofm.json \
  --result results/evaluations/baseline.json \
  --output-dir results/foundation_arena
```

### Forecastbewijs

Een Clarke Error Grid vereist gepaarde voorspellingen en referentiewaarden uit
een held-out evaluatie. Zonder paren wordt de figuur niet gemaakt. De zones
worden uit de werkelijk getekende punten geteld.

### Confounderbewijs

De confounderfiguur verwacht per geëvalueerd paar:

```text
model_name,si_ratio,embedding_cosine_similarity
```

De SDK voegt geen fictieve vergelijkingsmodellen of cosinewaarden toe.

### Dual-sensorbewijs

De dual-sensorfiguur verwacht gepaarde metingen:

```text
timestamp,dexcom_mgdl,libre_mgdl,cohort
```

De grafiek toont beschrijvende medianen en rapporteert het aantal werkelijk
beschikbare paren.

### Safetybewijs

De safetyfiguur verwacht een in-silico trace:

```text
time_minutes,unsupervised_glucose_mgdl,supervised_glucose_mgdl
```

Deze figuur blijft simulatiebewijs. Ze is geen FDA-validatie, reproduceert geen
patiëntincident en bewijst geen klinische veiligheid.

## Figuren genereren

Een methode-only dossier kan zonder evidence worden gemaakt:

```bash
iints research visualize --output-dir results/scientific_visualizations
iints research eucys-playbook --output-dir results/eucys_jury_dossier
```

De ontbrekende resultaatfiguren worden dan gemarkeerd als `not generated`.

Voor een evidence-backed visualisatieset:

```bash
iints research visualize \
  --arena-result results/evaluations/glucofm.json \
  --arena-result results/evaluations/baseline.json \
  --confounder-evidence results/confounder_pairs.csv \
  --dual-sensor-evidence results/paired_sensors.csv \
  --safety-trace results/safety_trace.csv \
  --output-dir results/scientific_visualizations
```

## Jurycontrole

Controleer voor elke resultaatclaim:

1. Is het bronbestand aanwezig?
2. Is de SHA-256 of provenance geregistreerd?
3. Is de train/validation/test-split op subjectniveau gescheiden?
4. Is de metric rechtstreeks uit de getoonde data berekend?
5. Is een simulatie duidelijk als simulatie gelabeld?
6. Wordt literatuurprestatie niet voorgesteld als lokaal gereproduceerd resultaat?
7. Wordt een architectuurdiagram niet voorgesteld als experimenteel bewijs?
8. Zijn beperkingen, ontbrekende waarden en mislukte runs zichtbaar gebleven?

## Wat niet mag worden beweerd

- dat de onafhankelijke implementatie een officieel GlucoFM-checkpoint is;
- dat vaste of synthetische demo-getallen empirische prestaties zijn;
- dat een in-silico safetytest klinische veiligheid bewijst;
- dat een gegenereerde GSIS-curve wet-labvalidatie is;
- dat een uncalibrated graftsimulatie insuline-onafhankelijkheid voorspelt;
- dat een juridische self-assessment EU AI Act-conformiteit of CE-markering is;
- dat de SDK een arts, medische beoordeling of klinische proef vervangt.

## Bronnen

- Metwally et al., *GlucoFM: A Foundation Model for Continuous Glucose
  Monitoring*, arXiv:2605.30865v2.
- Clarke et al., *Evaluating Clinical Accuracy of Systems for Self-Monitoring
  of Blood Glucose*, Diabetes Care, 1987.
- Battelino et al., *Clinical Targets for Continuous Glucose Monitoring Data
  Interpretation*, Diabetes Care, 2019.
- Regulation (EU) 2024/1689. Juridische status in IINTS-AF blijft een
  self-assessment; dit document is geen juridisch advies.
