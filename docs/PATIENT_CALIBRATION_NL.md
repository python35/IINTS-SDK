# Fysiologische kalibratie van de virtuele patiënten

> Onderzoeks- en onderwijsfunctionaliteit. Geen klinische validatie, geen dosisadvies.

## Waarom

Een virtuele patiënt wordt ingesteld met *klinische* knoppen (basaalstand, ISF,
ICR), terwijl de ODE-backends door *fysiologische* parameters worden bepaald.
Die twee beschrijvingen zijn niet onafhankelijk:

* een vastenevenwicht op een gevraagde glucose, bij een gevraagde basaalstand, en
* een gevraagde insulinegevoeligheidsfactor (ISF)

zijn twee eisen aan dezelfde grootheid. Met één schaalfactor is er maar één
haalbaar. De bestaande Hovorka-constructor kiest de eerste: hij schaalt
S_IT/S_ID/S_IE zodat de ingestelde basaalstand een evenwicht oplevert. Gevolg:

* de basaalstand heeft geen stationair effect meer (elke waarde "klopt"), en
* de ingestelde ISF verschuift het evenwicht in plaats van de insulinerespons.

Gemeten op de ongewijzigde backend: een correctiebolus van 1 E verlaagt glucose
met 33.9 mg/dL terwijl 50 is ingesteld, de basaalstand verandert de vastenglucose
met 0.00 mg/dL over 0.4-1.6 E/u, en bij ISF 40 of 60 loopt de nuchtere patiënt
in 24 uur 71 respectievelijk 45 mg/dL weg van zijn startwaarde.

## Wat is er toegevoegd

`iints.core.patient.calibration` keert de volgorde om, wat ook fysiologisch de
eerlijke richting is: **insulinegevoeligheid is de patiënteigenschap, de
basaalbehoefte volgt daaruit.**

| Functie | Doel |
|---|---|
| `calibrate_hovorka(insulin_sensitivity, target_glucose_mgdl, ...)` | lost gevoeligheidsschaal én basaalstand op zodat beide eisen gelden |
| `calibrate_carb_factor(calibration, meal_grams=60)` | zoekt de ICR waarbij de patiënt 6 uur na een maaltijd terug op zijn uitgangswaarde staat |
| `hovorka_euglycaemic_basal_rate(params, glucose, ...)` | analytische basaalbehoefte uit de vastenbalans |
| `measure_insulin_sensitivity_factor(factory, basal)` | gerealiseerde ISF als **gepaarde** meting (run met bolus tegen run zonder) |
| `measure_fasting_drift(factory, basal)` | 24-uursdrift bij vasten |
| `measure_plasma_insulin_tmax(factory, basal)` | piektijd plasma-insuline |
| `measure_clinical_parameters(factory, basal, carb_factor=...)` | audit-overzicht voor elke backend |
| `run_protocol(model, minutes, basal, boluses=, meals=)` | uniforme aansturing; een backend die zijn domein verlaat wordt als *abort* vastgelegd |

Daarnaast:

* `HovorkaPatientModel.from_clinical_targets(...)` — gekalibreerde constructor.
* `HovorkaPatientModel.measure_clinical_parameters()` — zelfaudit: ingestelde
  naast gemeten waarden, op wegwerpkopieën, dus de eigen toestand blijft intact.
* `HovorkaPatientModel(insulin_sensitivity_scale=...)` — expliciete
  gevoeligheidsschaal in plaats van de basaal-fit. De schaal reist mee in
  `get_state()`/`set_state()`.
* `HovorkaPatientModel(min_glucose_mgdl=...)` — expliciete ondergrens. Zonder
  deze instelling verwerpt de backend alleen negatieve glucose en kan hij
  0.0 mg/dL als geldige toestand rapporteren.
* `BergmanPatientModel(insulin_action_reference="plasma_ib")` — de referentie in
  `dX/dt` wordt de fysiologische `params.Ib` in plaats van de basaalafgeleide
  waarde, zodat de basaalstand weer effect heeft.
* `BergmanPatientModel(insulin_absorption_tmax_minutes=..., gastric_emptying_minutes=...)`
  — ontkoppelt de insulineabsorptie van `insulin_peak_time` (een werkingspiek,
  geen absorptieconstante) en de maaglediging van `carb_absorptieduur/3`.

**Alle standaardwaarden zijn ongewijzigd.** Bestaande scripts, opgeslagen runs en
de vastgelegde auditresultaten veranderen niet; de nieuwe opties zijn opt-in.

## Gebruik

```python
from iints.core.patient.hovorka_model import HovorkaPatientModel

patient = HovorkaPatientModel.from_clinical_targets(
    insulin_sensitivity=50.0,      # mg/dL per eenheid, nu echt waar
    target_glucose_mgdl=120.0,
    auto_carb_factor=True,         # leidt ook de ICR af
)
print(patient.calibration.basal_insulin_rate)   # afgeleid, niet ingesteld
print(patient.measure_clinical_parameters())
```

De kalibratie draait korte simulaties (ongeveer 8 s per patiënt). Bewaar
`patient.calibration` en gebruik `HovorkaCalibration.build()` om dezelfde
patiënt zonder herberekening op te bouwen — bijvoorbeeld voor een
patiëntenpopulatie in een controllerstudie.

## Gemeten effect

Hovorka, 70 kg, ISF 50 mg/dL/E, doelglucose 120 mg/dL:

| Meting | Huidig (basaal ingesteld) | Gepubliceerde S_I, geen refit | Gekalibreerd |
|---|---|---|---|
| Basaalstand | 0.800 E/u (invoer) | 0.377 E/u | 0.643 E/u (afgeleid) |
| ICR | 10.0 g/E (invoer) | 10.0 g/E | 18.0 g/E (afgeleid) |
| Gerealiseerde ISF | 33.9 mg/dL/E | 79.8 mg/dL/E | 49.5 mg/dL/E |
| Vastendrift 24 u | 0.00 mg/dL | 0.52 mg/dL | 0.00 mg/dL |
| Basaal-dosisrespons (0.5-1.5x) | 0.0 mg/dL | 187.2 mg/dL | 187.3 mg/dL |
| Afwijking 6 u na maaltijd | -45.6 mg/dL | -67.2 mg/dL | +0.9 mg/dL |
| Laagste glucose na maaltijd | 73.0 mg/dL | 24.6 mg/dL | 119.5 mg/dL |
| Plasma-insuline t_max | 64 min | 64 min | 64 min |

Over de ISF-ruimte van de gepaarde controllerstudie:

| Ingestelde ISF | Drift huidig | Drift gekalibreerd | Afgeleide basaalstand | Gemeten ISF |
|---|---|---|---|---|
| 40 mg/dL/E | 71.5 mg/dL | 0.00 mg/dL | 0.80 E/u | 40.0 |
| 50 mg/dL/E | 0.00 mg/dL | 0.00 mg/dL | 0.64 E/u | 49.5 |
| 60 mg/dL/E | 44.8 mg/dL | 0.00 mg/dL | 0.52 E/u | 60.5 |

Dit levert ook een inhoudelijk zinnige patiëntenfamilie op: een gevoeliger
patiënt (hogere ISF) krijgt automatisch een lagere basaalbehoefte.

Bergman met `insulin_action_reference="plasma_ib"` geeft over 0.4-1.6 E/u een
spreiding van 17.8 mg/dL tegen 0.0 mg/dL in de standaardmodus. De richting klopt
daarmee, maar de minimal-modelterm `p1 x (Gb - G)` (met de literatuurwaarde
p1 = 0.028/min, tijdconstante circa 36 min) blijft het gedrag domineren. Gebruik
Bergman niet als patiënt voor open-lus-experimenten van meerdere uren.

## Beperkingen

* De kalibratie fit het model aan zijn **eigen definities**, niet aan
  patiëntdata. Het resultaat is een intern consistente virtuele patiënt, geen
  gevalideerde patiënt.
* De vastenbalans negeert de tegenregulatie onder circa 70 mg/dL; richt niet op
  doelen in dat gebied.
* De ISF-meting gebruikt een bolus van 1 E vanaf 180 mg/dL. Omdat het model
  niet-lineair is, geldt de gekalibreerde ISF strikt genomen in die buurt.
* De gevoeligheidsschaal werkt op S_IT, S_ID en S_IE tegelijk. Een
  patiëntenpopulatie met onafhankelijke spreiding per compartiment vraagt een
  parameterverdeling uit de literatuur, geen enkele schaal.

## Tests

`tests/core/test_patient_calibration.py` toetst identificeerbaarheid: een
ingestelde knop moet een meetbaar fysiologisch gevolg hebben. De assertions
gaan over **gemeten** gedrag (gepaarde simulaties), niet over opgeslagen
attributen.
