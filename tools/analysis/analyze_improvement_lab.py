"""Inspect, verify, compare and plot a completed SDK improvement experiment bank."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import sqlite3
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
COLORS={'custom':'#586270','bergman':'#2463a6','hovorka':'#c36832','advanced':'#b18620'}


def verify(out,protocol,runs):
    recorded=(out/'protocol.sha256').read_text().strip()
    assert hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()==recorded
    expected={j['run_id']:j for j in protocol['jobs']}
    assert set(runs.run_id)==set(expected)
    frames={};count=0
    for _,r in runs.iterrows():
        receipt=json.loads((out/'receipts'/f'{r.run_id}.json').read_text())
        path=out/receipt['trace_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['trace_sha256']
        assert receipt['source_fingerprint']==protocol['frozen_source_fingerprint']
        frame=pd.read_csv(path);job=expected[r.run_id]
        assert len(frame)==r.rows
        if len(frame):
            if job['family']=='closed_loop':
                valid=frame[frame.interval_start_min<job['duration_min']]
                g=valid.glucose_truth_mgdl.to_numpy()
                assert np.allclose(frame.glucose_observation_time_min,frame.interval_start_min)
                assert np.allclose(frame.compartment_time_min,frame.interval_end_min)
            else:
                assert frame.observation_time_min.iloc[0]==0
                assert np.all(np.diff(frame.observation_time_min)>0)
                g=frame.glucose_truth_mgdl.to_numpy()[:-1]
                if r.completed:assert frame.observation_time_min.iloc[-1]==job['duration_min']
            assert np.isfinite(g).all()
        else:g=np.array([])
        assert np.isclose(len(g)*job['dt_min'],r.observed_min)
        in_range=np.sum((g>=70)&(g<=180))*job['dt_min']
        assert np.isclose(r.tir_full_lower,100*in_range/job['duration_min'])
        assert np.isclose(r.tir_full_upper,100*(in_range+job['duration_min']-len(g)*job['dt_min'])/job['duration_min'])
        assert r.composite_failure==((not r.completed) or bool(np.any(g<54)))
        count+=len(frame)
        # Keep only the required open-loop comparison families in memory.
        if job['family'] in ('parameter_space','dose_response','meal_response','time_step','new_calibration_knobs'):
            frames[r.run_id]=frame
    receipt=dict(model_runs_verified=len(runs),trace_rows_verified=count,protocol_sha256=recorded,
                 all_trace_hashes_verified=True,all_tir_bounds_recomputed=True,all_failure_endpoints_recomputed=True,
                 source_version_consistent=True)
    (out/'integrity_check.json').write_bytes(json.dumps(receipt,indent=2).encode())
    return expected,frames


def responses(out,runs,jobs,frames):
    groups={}
    for rid,j in jobs.items():
        family=j['family']
        if family in ('dose_response','meal_response'):
            key=(family,j['model'],j['pair_group'])
        elif family=='parameter_space':key=(family,j['model'],j['patient_id'])
        else:continue
        groups.setdefault(key,[]).append(rid)
    complete=runs.set_index('run_id').completed.to_dict();records=[]
    for key,ids in groups.items():
        family=key[0]
        controls=[rid for rid in ids if (jobs[rid]['condition']=='fasting' if family=='parameter_space'
                                       else jobs[rid].get('bolus_u',0)==0 and jobs[rid].get('carbs_g',0)==0)]
        if not controls:continue
        reference=controls[0];base=frames[reference]
        if base.empty:continue
        for rid in ids:
            if rid==reference:continue
            j=jobs[rid];f=frames[rid]
            if f.empty:continue
            stop=min(float(base.observation_time_min.max()),float(f.observation_time_min.max()))
            times=f.loc[f.observation_time_min<=stop,'observation_time_min'].to_numpy()
            if len(times)<2:continue
            control=np.interp(times,base.observation_time_min,base.glucose_truth_mgdl)
            experiment=np.interp(times,f.observation_time_min,f.glucose_truth_mgdl)
            difference=control-experiment
            dose=j.get('bolus_u',0.);carbs=j.get('carbs_g',0.)
            signed= difference if j['condition'] in ('bolus','exercise') else -difference
            effect=float(np.max(signed));peak=int(np.argmax(signed))
            records.append(dict(run_id=rid,reference_run_id=reference,family=family,model=j['model'],
                                condition=j['condition'],patient_id=j['patient_id'],pair_group=str(key[-1]),
                                nominal_isf=j['patient_params']['insulin_sensitivity'],nominal_icr=j['patient_params']['carb_factor'],
                                initial_glucose=j['patient_params']['initial_glucose'],dose_u=dose,carbs_g=carbs,
                                peak_effect_mgdl=effect,peak_effect_time_min=float(times[peak]),
                                peak_effect_per_u=effect/dose if dose else None,
                                peak_effect_per_g=effect/carbs if carbs else None,
                                integrated_difference_mgdl_min=float(np.trapezoid(signed,times)),
                                common_observed_min=stop,both_completed=bool(complete[rid] and complete[reference])))
    frame=pd.DataFrame(records);frame.to_csv(out/'paired_responses.csv',index=False)
    with sqlite3.connect(out/'experiments.sqlite') as db:frame.to_sql('paired_responses',db,if_exists='replace',index=False)
    return frame


def splits(out,index):
    index=index.copy()
    index['parameter_group']=index.parameters_json.map(lambda x:hashlib.sha256(x.encode()).hexdigest()[:16])
    index['fold']=index.parameter_group.map(lambda x:int(x,16)%5)
    index['suggested_split']=index.fold.map(lambda x:'test' if x==4 else ('validation' if x==3 else 'train'))
    index.to_csv(out/'grouped_splits.csv',index=False)
    with sqlite3.connect(out/'experiments.sqlite') as db:index.to_sql('grouped_splits',db,if_exists='replace',index=False)
    return index


def artifacts(out,runs,params,response,devices,jobs,manifest):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    gallery=[]
    def save(fig,name,title,note):
        fig.savefig(out/'figures'/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)
        gallery.append(dict(src=f'figures/{name}.png',title=title,note=note))
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    data=runs.groupby('family').size();ax.barh(data.index,data.values,color='#2463a6');ax.set(xlabel='Modelproeven',title='Omvang per experimentfamilie')
    save(fig,'01_bank_size','Omvang van het experimentenpakket','Modelproeven staan los van de 195 device-reeksen. Meetrijen zijn geen onafhankelijke patiënten.')
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    s=runs.groupby(['family','model'],sort=False).completed.mean().unstack()
    cmap=plt.get_cmap('cividis').copy();cmap.set_bad('#d9e0e9')
    im=ax.imshow(np.ma.masked_invalid(s.to_numpy()*100),cmap=cmap,vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(len(s.columns)),s.columns);ax.set_yticks(range(len(s.index)),s.index)
    for row in range(len(s.index)):
        for col in range(len(s.columns)):
            value=s.iloc[row,col]
            ax.text(col,row,f'{100*value:.1f}%' if pd.notna(value) else 'niet getest',ha='center',va='center',color='white' if pd.notna(value) and value<.5 else 'black')
    fig.colorbar(im,ax=ax,label='Volledig afgerond (%)');ax.set(title='Volledigheid per model en familie')
    save(fig,'02_completion','Volledigheid','Brede parameterproeven blijven verkennend. Een domeinstop is zichtbaar en wordt niet als volledige proef gerekend.')
    for family,amount,name,title in [('dose_response','dose_u','03_dose','Bolusrespons'),('meal_response','carbs_g','04_meals','Maaltijdrespons')]:
        fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
        for ax,m in zip(axes,COLORS):
            f=response[(response.family==family)&(response.model==m)&(response.initial_glucose==150)]
            valuecol='nominal_isf' if family=='dose_response' else 'nominal_icr'
            for value,g in f.groupby(valuecol):
                g=g.sort_values(amount);ax.plot(g[amount],g.peak_effect_mgdl,label=f'{valuecol}={value:g}',alpha=.75)
                incomplete=g[~g.both_completed];ax.scatter(incomplete[amount],incomplete.peak_effect_mgdl,marker='x',color='black',s=50)
            ax.set(title=m,xlabel='Bolus (U)' if family=='dose_response' else 'Maaltijd (g)',ylabel='Max. verschil tegenover controle (mg/dL)');ax.legend(fontsize=8)
        save(fig,name,title,'Gepaarde vergelijking met nul-dosis/maaltijd bij gelijke parameters. Kruis: minstens één traject afgebroken; effect geldt uitsluitend voor hun gemeenschappelijk waargenomen deel.')
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    f=response[(response.family=='parameter_space')&(response.condition=='bolus')&response.both_completed]
    for m,g in f.groupby('model'):ax.scatter(g.nominal_isf,g.peak_effect_per_u,label=m,color=COLORS[m],alpha=.6)
    ax.plot([25,85],[25,85],'--',color='black',label='Gelijke nominale en gemeten waarde');ax.set(xlabel='Nominale ISF-setting',ylabel='Gepaarde piekrespons / 0.5 U (mg/dL/U)',title='Nominale instelling versus gemeten modelrespons');ax.legend(fontsize=9)
    save(fig,'05_sensitivity','Nominale en effectieve respons','Alleen volledige controle/bolusparen. Piekrespons in deze 6-uursproef is een operationele modelmaat, geen klinisch gemeten ISF.')
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    closed=runs[runs.family=='closed_loop'];g=closed.groupby(['model','algorithm'],sort=False).composite_failure.mean()
    ax.barh([f'{m}/{a}' for m,a in g.index],100*g.values,color='#c36832');ax.set(xlim=(0,100),xlabel='Afgebroken of glucose <54 (%)',title='Closed-loop uitkomsten op de bevroren huidige versie')
    save(fig,'06_control','Controlleruitkomsten','32 parametercombinaties per backend, vier scenario’s, drie controllers en 12 uur. Geen externe patiëntvalidatie of vergelijking met dezelfde broncode als de eerdere studie.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ar=devices[devices.kind=='sensor_AR1'];axes[0].scatter(ar.sigma,ar.sample_sd,color='#2463a6',alpha=.5);axes[0].plot([2,15],[2,15],'--',color='black');axes[0].set(xlabel='Ingestelde SD (mg/dL)',ylabel='Gemeten SD na burn-in',title='AR(1)-sensorruis')
    axes[1].scatter(ar.phi,ar.sample_acf1,color='#b18620',alpha=.5);axes[1].plot([0,1],[0,1],'--',color='black');axes[1].set(xlabel='Ingestelde phi',ylabel='Gemeten lag-1 autocorrelatie',title='Ruisgeheugen')
    save(fig,'07_noise','Sensorruis over parameterbereik','96 reeksen: vier phi’s, drie SD’s en acht seeds. Onderlinge samples binnen één reeks zijn gecorreleerd.')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    d=devices[devices.kind=='dropout_occupancy'];ax.scatter(d.theoretical_occupancy,d.occupancy,color='#2463a6',alpha=.5);ax.plot([0,1],[0,1],'--',color='black');ax.set(xlabel='Theoretische hold-occupancy',ylabel='Waargenomen hold-occupancy',title='Dropouttrigger versus werkelijke uitvalduur')
    save(fig,'08_dropout','Sensoruitvalduur','72 reeksen. Bij vaste holdduur L en triggerkans p is de asymptotische occupancy pL/(1+p(L-1)); de triggerkans is per sample, niet per minuut.')
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    d=devices[devices.kind=='pump_boundary'].groupby(['cap','quantum']).quantization_violations.mean().unstack();im=ax.imshow(d.to_numpy(),cmap='cividis',aspect='auto');ax.set_xticks(range(len(d.columns)),[f'q={x:g}' for x in d.columns]);ax.set_yticks(range(len(d.index)),[f'cap={x:g}' for x in d.index]);fig.colorbar(im,ax=ax,label='Gem. niet-gekwantiseerde samples per 10.000');ax.set(title='Cap en kwantisatie: gerichte grensgevallen')
    save(fig,'09_pump','Pompgrensgevallen','27 reeksen met drie caps, drie kwantums en drie ruismagnitudes. Geen echte pompmetingen.')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    e=runs[runs.family=='endurance'];ax.barh([f'{r.model}/{r.condition}' for _,r in e.iterrows()],e.observed_min/1440,color='#2463a6');ax.set(xlim=(0,7),xlabel='Ondersteunde simulatieduur (dagen)',title='Zevendaagse endurance-proeven')
    save(fig,'10_endurance','Langdurige simulatie','Afgebroken trajecten zijn niet aangevuld. Worker-RSS-metingen bevatten ook caches en eerdere jobs en zijn geen bewijs voor een geheugenlek.')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    k=runs[runs.family=='new_calibration_knobs'];g=k.groupby(['model','condition']).completed.mean();ax.barh([f'{m}/{c}' for m,c in g.index],100*g.values,color='#b18620');ax.set(xlim=(0,100),xlabel='Volledig afgerond (%)',title='Nieuwe kalibratieknoppen: diagnostische proeven')
    save(fig,'11_calibration','Nieuwe kalibratie-instellingen','Expliciete Hovorka-sensitivity scales met analytisch basal-evenwicht; Bergman/Advanced reference modes en absorptietijden. Geen fitting op patiëntdata.')
    (out/'gallery.json').write_bytes(json.dumps(gallery,indent=2).encode())
    worst=closed[closed.composite_failure].sort_values(['coverage_pct','below54_observed_min'],ascending=[False,False]).head(30)
    worst[['run_id','model','condition','algorithm','coverage_pct','glucose_min','termination_json','trace_path']].to_csv(out/'reproduction_cases.csv',index=False)
    return gallery


def documents(out,manifest,gallery,runs,response):
    readme=f'''# SDK improvement data bank

## Wat er is uitgevoerd
{manifest['trials']:,} modelproeven met {manifest['patient_trace_rows']:,} trajectrijen; {manifest['device_series']} device-reeksen met {manifest['device_samples']:,} samples. Samen {manifest['actual_measurement_rows']:,} daadwerkelijk opgeslagen meetrijen. {manifest['complete_trials']:,} modelproeven ronden volledig af.

Dit zijn **synthetische onderzoeksgegevens**. Het aantal meetrijen is geen aantal onafhankelijke patiënten. Iedere proef, parametercombinatie, pairing, seed, broncodeversie en uitvalreden is terug te vinden.

## Direct gebruiken
- `experiments.sqlite`: tabellen runs, experiments, device_runs, paired_responses en grouped_splits.
- `runs.csv` en `experiment_index.csv`: uitkomsten en configuraties per modelproef.
- `paired_responses.csv`: controlevergelijkingen voor dosis, maaltijden, inspanning en ziekte. Niet extrapoleren voorbij common_observed_min.
- `traces/<family>/<run_id>.csv.gz`: volledige modeltrajecten, inclusief compartimenten waar beschikbaar.
- `devices/*.csv.gz`: afzonderlijke sensor- en pompreeksen.
- `reproduction_cases.csv`: 30 geselecteerde probleemgevallen om opnieuw uit te voeren.
- `grouped_splits.csv`: voorgestelde groepen voor train/validation/test. Dezelfde patiëntparameters blijven samen, ook over controllers en backends. Deze splits zijn geen externe klinische holdout.
- `protocol.json`, `receipts/*.json`, `manifest.json`, `integrity_check.json`: configuraties, checksums en verificatie.
- `frozen_sdk/src`: de exacte huidige SDK-broncode waarop alle proeven zijn uitgevoerd, inclusief lokale wijzigingen. Vergelijk de percentages niet rechtstreeks met eerdere broncodeversies.

Voorbeeld in Python:
```python
import sqlite3
import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parent
with sqlite3.connect(root / 'experiments.sqlite') as connection:
    failures = pd.read_sql_query('SELECT * FROM runs WHERE completed=0', connection)
trajectory = pd.read_csv(root / failures.iloc[0]['trace_path'])
```

Voer `analysis_examples.py` uit voor SQL-selecties en het laden van een volledig traject. `replay_experiment.py RUN_ID --sdk-root PAD_NAAR_REPO --output NIEUWE_MAP` voert een opgeslagen proef uit op een andere SDK-checkout. Zonder --sdk-root gebruikt replay de bevroren versie. Gebruik een verse Python-proces en nieuwe outputmap per replay.

## Tijd, eenheden en uitval
Zie DATA_DICTIONARY.md. Open-loop-proeven integreren exact de gevraagde intervallen en bewaren de initiële en eindtoestand. Closed-loop-proeven bewaren de originele SDK-uitvoer, met aparte tijdkolommen voor glucose en compartimenten en een vlag voor het extra terminale interval. Dat interval telt niet mee in 12-uurs-TIR.

Volledige TIR-ondergrens: ontbrekende intervallen buiten 70-180 mg/dL. Bovengrens: ontbrekende intervallen binnen 70-180. Dit zijn grenzen vanwege uitval, geen statistische betrouwbaarheidsintervallen. Samengesteld falen betekent afgebroken of glucose <54 in een waargenomen interval-startsnapshot. De laatste endpointwaarde blijft in het traject beschikbaar.

De brede parameterbank toetst gebruiksgrenzen. Domeinstops zijn onderzoeksuitkomsten en niet automatisch softwaredefecten. Nominale ISF, ICR en basal kunnen onderling fysiologisch inconsistent zijn. Controle- en behandelparen zijn identiek geconfigureerd; verschillen tussen backends zijn geen patiëntvalidatie.

## Betrouwbaarheid van dit pakket
Alle {manifest['trials']:,} tracechecksums, bronversielabels en herberekende TIR/falen-uitkomsten zijn gecontroleerd. De datapipeline heeft afzonderlijke tests voor eindtijd, gecensureerde trajecten, resume en corruptiedetectie. Parquet was door lokaal toepassingsbeleid onbruikbaar; het volledige pakket gebruikt portable gzip-CSV en standaard SQLite.
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    dictionary='''# DATA DICTIONARY

Units: time in minutes; glucose mg/dL; insulin dose U per interval; basal U/hour; carbs grams. Each run's constructor parameters are in experiment_index.parameters_json and protocol.jobs. The exact compartment schema is in the frozen SDK source.

Open loop: observation_time_min marks an initial or post-interval state. interval_start_min/interval_end_min describe the preceding successful integration interval; the initial row has no preceding interval. insulin_dose_u and carbs_delivered_g belong to that preceding interval. iob_units and cob_grams are the backend's reported trackers, not necessarily physical compartment mass.

Closed loop: glucose_observation_time_min and interval_start_min are the SDK pre-step time; compartment_time_min and interval_end_min are post-step. patient_state_* holds post-step compartments. terminal_extra_interval flags records at or after the requested horizon: saved for diagnosis, excluded from full-window summary denominators. Raw step_latency_ms is host-specific instrumented wall time under concurrent worker load.

Observations after an abort are absent. observed_min is supported integrated duration. Runtime errors and domain stops are preserved, including constructor failures. A missing value is unavailable, not zero. Worker RSS is whole-process memory and cannot isolate leaks.

Source SHA-256 in every receipt ties the trial to the copied working-tree source. run_id deterministically derives from the source fingerprint and canonical experiment configuration. protocol.sha256 hashes exact file bytes. Re-running produces new wall-time and RSS measurements; physical output should reproduce with the same source/seed/config.

Devices: one run_id represents one sensor-noise sequence, dropout sequence or pump grid. Samples within a sequence are dependent. Dropout probabilities are per-sample trigger probabilities, not overall occupancy. Quantization and cap metrics concern the software abstraction.

Grouped split: parameter_group hashes patient constructor parameters. All sibling treatments and backends remain in the same split. Do not randomly split trajectory rows for training/testing; adjacent timepoints and paired experiments leak information.
'''
    (out/'DATA_DICTIONARY.md').write_text(dictionary,encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(g["title"])}</h2><p>{html.escape(g["note"])}</p><img src="{g["src"]}" alt="{html.escape(g["title"])}"></section>' for g in gallery)
    body=f'''<!doctype html><html lang="nl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SDK improvement data bank</title><style>body{{font:17px/1.6 system-ui;color:#233142;background:#f3f5f8;margin:0}}main{{max-width:1150px;margin:auto;padding:32px}}section{{background:white;padding:24px;margin:24px 0;border:1px solid #d9e0e9}}img{{width:100%;height:auto}}a{{color:#2463a6}}.notice{{background:#fff5da;padding:20px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.tablewrap{{overflow-x:auto}}</style><main><h1>SDK improvement data bank</h1><p class="notice">{manifest['trials']:,} modelproeven · {manifest['actual_measurement_rows']:,} opgeslagen meetrijen · {manifest['complete_trials']:,} volledig afgeronde modelproeven. Synthetische onderzoeksdata; geen patiëntvalidatie.</p><p><a href="README.md">Gebruik en methode</a> · <a href="DATA_DICTIONARY.md">Dataschema</a> · <a href="runs.csv">Alle modelproeven</a> · <a href="paired_responses.csv">Gepaarde responsen</a> · <a href="experiments.sqlite">SQLite-overzicht</a> · <a href="reproduction_cases.csv">Probleemgevallen</a></p>{cards}</main></html>'''
    (out/'report.html').write_text(body,encoding='utf-8')


def helpers(out):
    (out/'analysis_examples.py').write_text(textwrap.dedent('''\
        from pathlib import Path
        import sqlite3
        import pandas as pd
        root=Path(__file__).resolve().parent
        with sqlite3.connect(root/'experiments.sqlite') as db:
            summary=pd.read_sql_query("SELECT family,model,COUNT(*) AS trials,SUM(completed) AS completed,AVG(coverage_pct) AS coverage FROM runs GROUP BY family,model",db)
            print(summary.to_string(index=False))
            failures=pd.read_sql_query("SELECT run_id,model,condition,termination_json,trace_path FROM runs WHERE completed=0 ORDER BY coverage_pct DESC LIMIT 5",db)
            print(failures.to_string(index=False))
            response=pd.read_sql_query("SELECT model,nominal_isf,peak_effect_per_u,common_observed_min FROM paired_responses WHERE condition='bolus' AND both_completed=1 LIMIT 10",db)
            print(response.to_string(index=False))
        if len(failures):
            trace=pd.read_csv(root/failures.iloc[0].trace_path)
            print(trace.head().to_string(index=False))
        '''),encoding='utf-8')
    replay='''from pathlib import Path
import argparse,importlib.util,json,hashlib
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--sdk-root',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
job=next(j.copy() for j in protocol['jobs'] if j['run_id']==a.run_id)
source=a.sdk_root.resolve() if a.sdk_root else root/'frozen_sdk'
files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((source/'src').rglob('*.py'))}
spec=importlib.util.spec_from_file_location('bank_runner',root/'runner_executed.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fingerprint=hashlib.sha256(m.canonical(files).encode()).hexdigest()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'receipts').mkdir(exist_ok=True)
m.initialize(source)
job['run_id']=hashlib.sha256((a.run_id+fingerprint).encode()).hexdigest()[:16]
receipt=m.run_job(job,out,fingerprint)
(out/'replay_config.json').write_bytes(json.dumps({'original_run_id':a.run_id,'source_fingerprint':fingerprint,'experiment':job},indent=2).encode())
print(json.dumps(receipt,indent=2))
'''
    (out/'replay_experiment.py').write_text(replay,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'results/sdk_improvement_lab_20261012');a=ap.parse_args();out=a.output.resolve()
    protocol=json.loads((out/'protocol.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    runs=pd.read_csv(out/'runs.csv');index=pd.read_csv(out/'experiment_index.csv');devices=pd.read_csv(out/'device_runs.csv')
    jobs,frames=verify(out,protocol,runs);response=responses(out,runs,jobs,frames);splits(out,index)
    gallery=artifacts(out,runs,index,response,devices,jobs,manifest);documents(out,manifest,gallery,runs,response);helpers(out)
    print(json.dumps({'runs_verified':len(runs),'paired_responses':len(response),'figures':len(gallery),'report':str(out/'report.html')},indent=2))


if __name__=='__main__':main()
