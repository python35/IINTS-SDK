"""Large reproducible experiment bank for improving the SDK, not patient care.

Uses a frozen copy of the current working-tree source (including local changes).
Raw measurements are gzip CSV files per trial; interrupted runs can be resumed.
No production SDK files are changed. Exit success means data generation finished,
not that the SDK passed scientific validation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sqlite3
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
SDK = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def initialize(snapshot):
    sys.path.insert(0, str(Path(snapshot)/"src"))
    logging.getLogger("iints").setLevel(logging.CRITICAL)
    global SDK
    from iints.core.patient.patient_factory import PatientFactory
    from iints.core.patient.calibration import hovorka_euglycaemic_basal_rate
    from iints.core.patient.hovorka_model import HovorkaParameters
    from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
    from iints.core.algorithms.fixed_basal_bolus import FixedBasalBolus
    from iints.core.algorithms.pid_controller import PIDController
    from iints.core.devices.models import SensorModel, PumpModel
    from iints.core.simulator import Simulator, StressEvent
    SDK = dict(factory=PatientFactory, calibration=hovorka_euglycaemic_basal_rate,
               hparams=HovorkaParameters, algorithms={"clinical":ClinicalBaselineAlgorithm,"basal_bolus":FixedBasalBolus,"pid":PIDController},
               sensor=SensorModel, pump=PumpModel, simulator=Simulator, event=StressEvent)


def create_jobs(n, seed):
    jobs=[]
    def add(family, model, params, condition, duration=360, dt=5, patient_id=0, **extras):
        jobs.append(dict(family=family, model=model, patient_params=params, condition=condition,
                         duration_min=duration, dt_min=dt, patient_id=patient_id, seed=seed+patient_id,
                         **extras))
    base=dict(basal_insulin_rate=.8,insulin_sensitivity=50.,carb_factor=10.,initial_glucose=150.,
              glucose_decay_rate=.02,glucose_absorption_rate=.03,insulin_action_duration=300.)
    models=["custom","bergman","hovorka","advanced"]
    bounds={"basal_insulin_rate":[.3,1.4],"insulin_sensitivity":[25.,85.],"carb_factor":[7.,18.],
            "initial_glucose":[90.,200.],"glucose_decay_rate":[.001,.05],
            "glucose_absorption_rate":[.01,.06],"insulin_action_duration":[180.,420.]}
    names=list(bounds);b=np.array(list(bounds.values()))
    samples=qmc.scale(qmc.LatinHypercube(len(names),seed=seed).random(n),b[:,0],b[:,1])
    for pid,values in enumerate(samples):
        params=dict(zip(names,map(float,values)))
        for m in models:
            for c in ("fasting","bolus","meal","exercise","illness"):
                add("parameter_space",m,params,c,patient_id=pid,bolus_u=.5 if c=="bolus" else 0.,carbs_g=60. if c=="meal" else 0.)
    # Paired grid: every dose group includes an identical zero-dose reference.
    for m in models:
        for g in (110.,150.,200.):
            for isf in (25.,50.,85.):
                params={**base,"initial_glucose":g,"insulin_sensitivity":isf}
                group=f"G{g}_ISF{isf}"
                for dose in (0.,.25,.5,1.,2.,4.):
                    add("dose_response",m,params,"bolus",bolus_u=dose,carbs_g=0.,pair_group=group)
            for icr in (7.,10.,18.):
                params={**base,"initial_glucose":g,"carb_factor":icr}
                group=f"G{g}_ICR{icr}"
                for carbs in (0.,15.,30.,60.,90.,120.):
                    add("meal_response",m,params,"meal",bolus_u=0.,carbs_g=carbs,pair_group=group)
    # A new paired closed-loop matrix with matched input/seeds, 12-hour horizon.
    cbounds={"basal_insulin_rate":[.5,1.],"insulin_sensitivity":[40.,60.],"carb_factor":[9.,13.],"initial_glucose":[130.,160.]}
    cn=list(cbounds);b=np.array(list(cbounds.values()))
    cs=qmc.scale(qmc.LatinHypercube(len(cn),seed=seed+1).random(n//2),b[:,0],b[:,1])
    for pid,values in enumerate(cs):
        params={**base,**dict(zip(cn,map(float,values)))}
        for m in models:
            for c in ("announced","missed_meal","exercise","device_fault"):
                for a in ("clinical","basal_bolus","pid"):
                    add("closed_loop",m,params,c,duration=720,patient_id=pid,algorithm=a,meal_grams=60.)
    # Long-term dosing and memory growth; keep controlled basal driving separate.
    for m in models:
        for c in ("fasting","daily_meals","pump_interruption"):
            add("endurance",m,base,c,duration=10080,dt=5,bolus_u=0.,carbs_g=0.)
    # Grid sensitivity around a known event schedule: no event rounding.
    for m in models:
        for c in ("fasting","meal","bolus"):
            for dt in (1,2,5,10,15):
                add("time_step",m,base,c,dt=dt,carbs_g=30. if c=="meal" else 0.,
                    bolus_u=.5 if c=="bolus" else 0.,bolus_duration_min=30)
    # Probe newly introduced knobs without tuning them to these outcomes.
    for scale in (.5,1.,2.):
        for g in (110.,150.):
            basal=SDK["calibration"](SDK["hparams"](),g,transport_scale=scale,disposal_scale=scale,egp_scale=scale)
            params={**base,"initial_glucose":g,"basal_insulin_rate":float(basal),"insulin_sensitivity_scale":scale}
            for c in ("fasting","bolus","meal"):
                add("new_calibration_knobs","hovorka",params,c,carbs_g=30. if c=="meal" else 0.,bolus_u=.5 if c=="bolus" else 0.)
    for m in ("bergman","advanced"):
        for reference in ("basal","plasma_ib"):
            for tmax in (35.,55.,75.):
                params={**base,"insulin_action_reference":reference,"insulin_absorption_tmax_minutes":tmax}
                for c in ("fasting","bolus"):
                    add("new_calibration_knobs",m,params,c,bolus_u=.5 if c=="bolus" else 0.,carbs_g=0.)
    return jobs


def state_row(patient, t, interval_start=np.nan, dose=0., carbs=0.):
    compartments=patient.get_compartment_state() if hasattr(patient,"get_compartment_state") else {}
    return dict(observation_time_min=t,interval_start_min=interval_start,interval_end_min=t,
                glucose_truth_mgdl=float(patient.get_current_glucose()),iob_units=float(patient.insulin_on_board),
                cob_grams=float(patient.carbs_on_board),insulin_dose_u=dose,carbs_delivered_g=carbs,
                **{f"compartment_{k}":float(v) for k,v in compartments.items()})


def open_loop(job):
    p=SDK["factory"].create_patient(patient_type=job["model"],**job["patient_params"])
    dt=job["dt_min"];end=job["duration_min"];base=job["patient_params"]["basal_insulin_rate"]
    rows=[state_row(p,0.)];reason=None
    for t in range(0,end,dt):
        dose=base*dt/60;carbs=0.;condition=job["condition"]
        bolus_window=job.get("bolus_duration_min",dt)
        if condition=="bolus" and 60<=t<60+bolus_window:
            dose+=job.get("bolus_u",0.)*dt/bolus_window
        if condition=="meal" and t==60:carbs=job.get("carbs_g",0.)
        if condition=="exercise":
            p.start_exercise(.6) if 60<=t<150 else p.stop_exercise()
        if condition=="illness":
            start=getattr(p,"start_illness",p.start_stress);stop=getattr(p,"stop_illness",p.stop_stress)
            start(.9) if 60<=t<240 else stop()
        if condition=="daily_meals" and t%1440 in (420,780,1140):
            carbs=45.;dose+=3.
        if condition=="pump_interruption" and 1440<=t<1800:dose=0.
        try:
            p.update(dt,dose,carbs,current_time=float(t))
            rows.append(state_row(p,float(t+dt),float(t),dose,carbs))
        except Exception as exc:
            reason=dict(type=type(exc).__name__,message=str(exc),attempted_interval_start_min=t,attempted_interval_end_min=t+dt)
            break
    frame=pd.DataFrame(rows)
    completed=len(rows)==end//dt+1
    # TIR uses the left endpoint of each successfully integrated interval.
    glucose=frame.glucose_truth_mgdl.to_numpy()[:-1]
    return frame,completed,reason,glucose


def closed_loop(job):
    p=SDK["factory"].create_patient(patient_type=job["model"],**job["patient_params"])
    fault=job["condition"]=="device_fault"
    sensor=SDK["sensor"](seed=job["seed"],noise_std=15. if fault else 8.5,
                          bias=25. if fault else 0.,dropout_prob=.1 if fault else 0.,dropout_duration_steps=(2,6))
    pump=SDK["pump"](seed=job["seed"],dropout_prob=.2 if fault else 0.)
    sim=SDK["simulator"](p,SDK["algorithms"][job["algorithm"]](),time_step=job["dt_min"],
                         seed=job["seed"],sensor_model=sensor,pump_model=pump,enable_profiling=True)
    for t in (60,420):
        sim.add_stress_event(SDK["event"](t,"meal",job["meal_grams"],reported_value=0. if job["condition"]=="missed_meal" else None))
    if job["condition"]=="exercise":sim.add_stress_event(SDK["event"](480,"exercise",.6,duration=60))
    captured=[]
    def capture(row):
        captured.append(dict(row))
        return None
    # Preserve the partial trajectory even if an unexpected error escapes run_batch.
    sim.on_step=capture
    try:
        frame,report=sim.run_batch(job["duration_min"])
        reason=report.get("termination_reason")
        completed=len(frame)==job["duration_min"]//job["dt_min"]+1 and not report.get("terminated_early",False)
    except Exception as exc:
        frame=pd.DataFrame(captured)
        reason=dict(type=type(exc).__name__,message=str(exc));completed=False
    if not frame.empty:
        keep=[c for c in frame if c.startswith("patient_state_") or c in ["time_minutes","glucose_actual_mgdl","glucose_to_algo_mgdl","delivered_insulin_units","patient_iob_units","patient_cob_grams","safety_triggered","safety_reason","sensor_status","pump_status","input_validator_fail_soft","step_latency_ms","supervisor_latency_ms"]]
        frame=frame[keep].copy()
        frame["interval_start_min"]=frame.time_minutes
        frame["interval_end_min"]=frame.time_minutes+job["dt_min"]
        frame["compartment_time_min"]=frame.interval_end_min
        frame["glucose_observation_time_min"]=frame.interval_start_min
        frame["terminal_extra_interval"]=frame.interval_start_min>=job["duration_min"]
        frame["glucose_truth_mgdl"]=frame.glucose_actual_mgdl
        glucose=frame.loc[~frame.terminal_extra_interval,"glucose_truth_mgdl"].to_numpy()
    else:glucose=np.array([])
    return frame,completed,reason,glucose


def run_job(job, output, source_hash):
    out=Path(output);rid=job["run_id"];folder=out/"traces"/job["family"];folder.mkdir(parents=True,exist_ok=True)
    data_file=folder/f"{rid}.csv.gz";receipt_file=out/"receipts"/f"{rid}.json"
    if receipt_file.exists():
        receipt=json.loads(receipt_file.read_text(encoding="utf-8"))
        if data_file.exists() and hashlib.sha256(data_file.read_bytes()).hexdigest()==receipt["trace_sha256"]:
            return receipt
        raise RuntimeError(f"Existing receipt does not match trace for {rid}; refusing overwrite")
    try:
        import psutil
        process=psutil.Process();memory_before=process.memory_info().rss
    except ImportError:
        process=None;memory_before=None
    start=time.perf_counter()
    try:
        frame,completed,reason,glucose=(closed_loop(job) if job["family"]=="closed_loop" else open_loop(job))
    except Exception as exc:
        frame=pd.DataFrame();completed=False;reason=dict(type=type(exc).__name__,message=str(exc),stage="construction");glucose=np.array([])
    elapsed=time.perf_counter()-start;duration=job["duration_min"];dt=job["dt_min"]
    in_range=int(np.sum((glucose>=70)&(glucose<=180)));observed=len(glucose)*dt
    df=frame.copy();df["run_id"]=rid
    temporary=data_file.with_name(data_file.name+'.partial')
    df.to_csv(temporary,index=False,compression={"method":"gzip","mtime":0},lineterminator="\n");temporary.replace(data_file)
    receipt=dict(run_id=rid,family=job["family"],model=job["model"],condition=job["condition"],algorithm=job.get("algorithm","open_loop"),
                 patient_id=job["patient_id"],seed=job["seed"],duration_min=duration,dt_min=dt,rows=len(frame),completed=completed,
                 observed_min=observed,coverage_pct=100*observed/duration,
                 tir_full_lower=100*in_range*dt/duration,tir_full_upper=100*(in_range*dt+duration-observed)/duration,
                 glucose_min=float(np.min(glucose)) if len(glucose) else None,glucose_max=float(np.max(glucose)) if len(glucose) else None,
                 below54_observed_min=int(np.sum(glucose<54))*dt,below70_observed_min=int(np.sum(glucose<70))*dt,
                 composite_failure=(not completed) or bool(np.any(glucose<54)),
                 runtime_s=elapsed,worker_memory_before_mb=memory_before/2**20 if memory_before else None,
                 worker_memory_after_mb=process.memory_info().rss/2**20 if process else None,
                 trace_columns={c:str(df[c].dtype) for c in df},
                 termination=reason,trace_path=str(data_file.relative_to(out)),trace_sha256=hashlib.sha256(data_file.read_bytes()).hexdigest(),
                 source_fingerprint=source_hash)
    tmp=receipt_file.with_suffix(".partial.json");tmp.write_bytes(json.dumps(receipt,indent=2,allow_nan=False).encode());tmp.replace(receipt_file)
    return receipt


def device_bank(out, seed):
    """Device samples are explicitly separate from independent patient trials."""
    records=[];rows_total=0
    for phi in (0.,.5,.85,.98):
        for sigma in (2.,8.,15.):
            for rep in range(8):
                rid=f"sensor_phi{phi}_sd{sigma}_r{rep}"
                sensor=SDK["sensor"](noise_std=sigma,noise_ar1_phi=phi,noise_fbm_hurst=None,lag_minutes=0,seed=seed+rep)
                values=np.array([sensor.read(120.,float(t*5)).value-120. for t in range(2500)])
                frame=pd.DataFrame(dict(sample=np.arange(len(values)),error_mgdl=values,replicate=rep,phi=phi,sigma=sigma))
                path=out/"devices"/f"{rid}.csv.gz";frame.to_csv(path,index=False,compression="gzip")
                sample=values[500:]
                records.append(dict(kind="sensor_AR1",run_id=rid,samples=len(values),phi=phi,sigma=sigma,
                                    sample_mean=float(sample.mean()),sample_sd=float(sample.std(ddof=1)),sample_acf1=float(np.corrcoef(sample[:-1],sample[1:])[0,1])))
                rows_total+=len(values)
    for cap in (.975,1.,1.025):
        for quantum in (.01,.05,.1):
            for noise in (0.,.1,.3):
                rid=f"pump_cap{cap}_q{quantum}_sd{noise}"
                pump=SDK["pump"](max_units_per_step=cap,quantization_units=quantum,delivery_noise_std=noise,seed=seed)
                requested=np.linspace(0.,2.,10000)
                delivered=np.array([pump.deliver(float(x),5.).delivered_units for x in requested])
                grid_error=np.abs(delivered/quantum-np.rint(delivered/quantum))>1e-8
                frame=pd.DataFrame(dict(requested_u=requested,delivered_u=delivered,quantization_violation=grid_error))
                frame.to_csv(out/"devices"/f"{rid}.csv.gz",index=False,compression="gzip")
                records.append(dict(kind="pump_boundary",run_id=rid,samples=len(frame),cap=cap,quantum=quantum,noise_sd=noise,
                                    cap_violations=int(np.sum(delivered>cap+1e-12)),quantization_violations=int(grid_error.sum())))
                rows_total+=len(frame)
    # Occupancy for hold periods differs from the per-step trigger probability.
    for probability in (.01,.05,.2):
        for length in (1,4,12):
            for rep in range(8):
                sensor=SDK["sensor"](noise_std=0.,lag_minutes=0,dropout_prob=probability,dropout_duration_steps=length,seed=seed+rep)
                readings=[sensor.read(120.+10*np.sin(t/10),float(t*5)) for t in range(2500)]
                dropout=np.array(["dropout" in x.status for x in readings])
                rid=f"dropout_p{probability}_L{length}_r{rep}"
                pd.DataFrame(dict(sample=np.arange(2500),sensor_value=[x.value for x in readings],dropout=dropout)).to_csv(out/"devices"/f"{rid}.csv.gz",index=False,compression="gzip")
                theoretical=probability*length/(1+probability*(length-1))
                records.append(dict(kind="dropout_occupancy",run_id=rid,samples=2500,probability=probability,hold_steps=length,
                                    occupancy=float(dropout[500:].mean()),theoretical_occupancy=theoretical))
                rows_total+=2500
    frame=pd.DataFrame(records);frame.to_csv(out/"device_runs.csv",index=False)
    return rows_total,len(records)


def finalize(out, config, receipts, device_rows, device_runs):
    frame=pd.DataFrame(receipts)
    frame["termination_json"]=frame.termination.map(lambda x:json.dumps(x,sort_keys=True) if x else "")
    frame=frame.drop(columns=["termination","trace_columns"]);frame.to_csv(out/"runs.csv",index=False)
    index=pd.DataFrame([dict(run_id=j["run_id"],family=j["family"],model=j["model"],condition=j["condition"],
                             algorithm=j.get("algorithm","open_loop"),patient_id=j["patient_id"],seed=j["seed"],
                             dt_min=j["dt_min"],duration_min=j["duration_min"],bolus_u=j.get("bolus_u"),carbs_g=j.get("carbs_g"),
                             pair_group=j.get("pair_group"),parameters_json=canonical(j["patient_params"])) for j in config["jobs"]])
    index.to_csv(out/"experiment_index.csv",index=False)
    with sqlite3.connect(out/"experiments.sqlite") as db:
        frame.to_sql("runs",db,if_exists="replace",index=False)
        index.to_sql("experiments",db,if_exists="replace",index=False)
        pd.read_csv(out/"device_runs.csv").to_sql("device_runs",db,if_exists="replace",index=False)
        db.execute("CREATE INDEX IF NOT EXISTS run_lookup ON runs(run_id)")
        db.execute("CREATE INDEX IF NOT EXISTS experiment_lookup ON experiments(run_id)")
        db.execute("CREATE INDEX IF NOT EXISTS family_lookup ON runs(family,model,condition)")
    summary=frame.groupby(["family","model"],sort=False).agg(runs=("run_id","size"),completed=("completed","sum"),
                  failures=("composite_failure","sum"),mean_coverage=("coverage_pct","mean"),min_glucose=("glucose_min","min"),max_glucose=("glucose_max","max"))
    summary.to_csv(out/"family_summary.csv")
    total_rows=int(frame.rows.sum())
    manifest={k:v for k,v in config.items() if k!="jobs"}
    manifest.update(trials=len(frame),complete_trials=int(frame.completed.sum()),patient_trace_rows=total_rows,
                    device_series=device_runs,device_samples=device_rows,actual_measurement_rows=total_rows+device_rows,
                    full_trajectories_csv_gzip=True,index_sqlite=True,receipt_checksums=True,
                    format_reason="Parquet DLL import blocked by local application control; using gzip CSV and standard-library SQLite",
                    package_versions={n:importlib.metadata.version(n) for n in ["numpy","scipy","pandas"]})
    (out/"manifest.json").write_bytes(json.dumps(manifest,indent=2,allow_nan=False).encode())
    return frame,index,manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--samples",type=int,default=64)
    ap.add_argument("--workers",type=int,default=4);ap.add_argument("--seed",type=int,default=20261014)
    ap.add_argument("--output",type=Path,default=ROOT/"results/sdk_improvement_lab_20261012")
    ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    if args.samples<8 or args.samples%2:ap.error("samples must be an even number >=8")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for folder in ("traces","receipts","devices","figures"):(out/folder).mkdir(exist_ok=True)
    protocol_file=out/"protocol.json";snapshot=out/"frozen_sdk"
    if args.resume:
        config=json.loads(protocol_file.read_text(encoding="utf-8"))
        if config["samples"]!=args.samples or config["seed"]!=args.seed:raise RuntimeError("Resume parameters differ from frozen protocol")
    else:
        if protocol_file.exists() or snapshot.exists():raise RuntimeError("Output already used; choose --resume or a fresh directory")
        shutil.copytree(ROOT/"src",snapshot/"src",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        source_files={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.rglob("*.py"))}
        source_hash=hashlib.sha256(canonical(source_files).encode()).hexdigest()
        (snapshot/"source_hashes.json").write_bytes(json.dumps(source_files,indent=2).encode())
        initialize(snapshot)
        jobs=create_jobs(args.samples,args.seed)
        for job in jobs:job["run_id"]=hashlib.sha256((source_hash+canonical(job)).encode()).hexdigest()[:16]
        assert len(set(j["run_id"] for j in jobs))==len(jobs)
        config=dict(version="improvement-data-bank-v1",samples=args.samples,seed=args.seed,workers=args.workers,
                    frozen_source_fingerprint=source_hash,source_files=source_files,jobs=jobs,
                    frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                    working_tree_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
                    python=sys.version,platform=platform.platform(),
                    purpose="Synthetic research experiments for regression, calibration diagnostics and SDK improvements; not clinical validation or patient data",
                    threshold_policy="Broad parameter-space probes are exploratory. Preserve default constructor domain/safety limits. No thresholds adjusted to improve outcomes.",
                    units="minutes, mg/dL, insulin U per interval, basal U/hour, carbs grams. Compartment units from frozen SDK schema.",
                    endpoints="Completion, observed lows and full-horizon TIR bounds; no extrapolation after failure.",
                    initial_truth_and_phase="Open loop records initial state and exact post-interval states. Closed loop preserves SDK pre-step truth and post-step compartments with separate time columns, including flagged terminal extra interval.")
        protocol_file.write_bytes(json.dumps(config,indent=2,allow_nan=False).encode())
        (out/"protocol.sha256").write_text(hashlib.sha256(protocol_file.read_bytes()).hexdigest(),encoding="utf-8")
        shutil.copy2(Path(__file__),out/"runner_executed.py")
    initialize(snapshot)
    source_hash=config["frozen_source_fingerprint"]
    actual={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.rglob("*.py"))}
    if hashlib.sha256(canonical(actual).encode()).hexdigest()!=source_hash:raise RuntimeError("Frozen SDK changed; refusing mixed-source experiments")
    print(f"Frozen source {source_hash[:12]}: {len(config['jobs'])} model trials; {args.workers} worker processes",flush=True)
    start=time.perf_counter();receipts=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(snapshot,)) as pool:
        futures={pool.submit(run_job,j,out,source_hash):j for j in config["jobs"]}
        for future in as_completed(futures):
            receipts.append(future.result())
            if len(receipts)%100==0 or len(receipts)==len(futures):
                progress=dict(done=len(receipts),planned=len(futures),elapsed_s=time.perf_counter()-start)
                (out/"progress.json").write_bytes(json.dumps(progress).encode())
                print(f"Model trials {len(receipts)}/{len(futures)}; {sum(r['rows'] for r in receipts):,} trace rows; {progress['elapsed_s']:.1f}s",flush=True)
    device_rows,device_runs=device_bank(out,args.seed)
    _,_,manifest=finalize(out,config,receipts,device_rows,device_runs)
    print(json.dumps({k:manifest[k] for k in ["trials","complete_trials","patient_trace_rows","device_series","device_samples","actual_measurement_rows"]},indent=2),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
