"""Prospectively frozen falsification tests for the IINTS research SDK.

This separates equation verification, numerical convergence, simulated control
performance, and empirical validity. It does not tune or repair SDK parameters.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import csv
import hashlib
import html
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, linalg, stats
from scipy.stats import qmc
from iints.api.base_algorithm import AlgorithmInput
from iints.core.patient.patient_factory import PatientFactory
from iints.core.patient.models import PatientModelDomainError
from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.core.algorithms.fixed_basal_bolus import FixedBasalBolus
from iints.core.algorithms.pid_controller import PIDController
from iints.core.algorithms.mock_algorithms import ConstantDoseAlgorithm
from iints.core.devices.models import SensorModel, PumpModel
from iints.core.simulator import Simulator, StressEvent

MODELS = ["custom", "bergman", "hovorka", "advanced"]
MECHANISTIC = MODELS[1:]
ARMS = {"clinical": ClinicalBaselineAlgorithm, "basal_bolus": FixedBasalBolus, "pid": PIDController}
SCENARIOS = ["meal_day", "exercise", "sensor_fault"]
COLORS = {"custom": "#586270", "bergman": "#2463a6", "hovorka": "#c36832", "advanced": "#b18620"}
SOURCES = [
 {"id":"scipy", "title":"SciPy solve_ivp: RK45, DOP853 en Radau", "url":"https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html", "role":"Onafhankelijke solveralgoritmen; dezelfde SDK-ODE blijft gedeeld."},
 {"id":"ruan", "title":"Ruan et al. (2014/2015), aspart-PK in kinderen, DOI 10.1007/s00125-014-3483-6", "url":"https://api.repository.cam.ac.uk/server/api/core/bitstreams/ae2c1977-f763-4e32-a84f-80002c2a35ba/content", "role":"N=11; U100 plasma-insulinepiek 59.2 +/-14.4 min. Alleen context: leeftijd, populatie en protocol verschillen."},
 {"id":"ivp", "title":"Kanderian et al. (2012), identifiable virtual patient model", "url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC3380781/", "role":"Primaire externe simulatorvalidatie; onafhankelijke proef en afgestemde populatie/input nodig. Geen validatie van IINTS."},
 {"id":"ohio", "title":"OhioT1DM: officiële datasetpagina van de makers", "url":"https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html", "role":"8 weken per persoon, 12 personen, pomptherapie, toegang op aanvraag. Controle van lokale metadata/herkomst."},
]

def model(name, **kwargs):
    return PatientFactory.create_patient(patient_type=name, **kwargs)


def solver_context(name, method):
    if method == "SDK_RK45":
        return nullcontext()
    module = "iints.core.patient.hovorka_model" if name == "hovorka" else "iints.core.patient.bergman_model"
    def reference(*args, **kwargs):
        kwargs.update(method=method, rtol=1e-10, atol=1e-12, max_step=.5)
        return integrate.solve_ivp(*args, **kwargs)
    return patch(module + ".solve_ivp", reference)


def protocol(n, seed):
    return {
        "version":"scientific-audit-v1", "frozen_before_execution_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "seed":seed, "cohort_n":n, "models":MODELS, "arms":list(ARMS), "scenarios":SCENARIOS,
        "simulation_minutes":1440, "time_step_minutes":5, "bootstrap_resamples":10000,
        "primary_endpoint":"composite run failure: termination/unexpected error OR any observed glucose <54 mg/dL",
        "secondary_endpoint":"full-horizon TIR lower/upper bounds; all terminated runs retained; final t=1440 excluded from interval denominator",
        "sampling":"paired Latin hypercube; same patient and sensor/pump seeds per arm; 3 scenarios averaged per patient before inference",
        "multiplicity":"Holm correction of the 8 two-sided TIR permutation comparisons versus clinical; paired patient-cluster bootstrap",
        "criteria":{
            "solver_glucose_abs_mgdl":.1, "solver_scaled_state_error":1e-4,
            "analytic_compartment_scaled_error":1e-5, "mass_balance_relative_error":1e-10,
            "basal_24h_abs_mgdl":1., "sensor_step_abs_mgdl":1e-10,
            "noise_mean_abs_mgdl":.5, "noise_sd_ratio":[.95,1.05], "noise_acf_abs_error":.02,
            "controller_time_step_relative_difference":.05,
            "snapshot_glucose_consistency_abs_mgdl":.1,
            "noninferiority_tir_margin_pp":5., "failure_margin_pp":5.,
        },
        "criterion_authority":"Engineering criteria chosen before this run. Not clinical standards; basal balance is a mechanistic hypothesis, exploratory for Custom.",
        "numeric_independence":"Matrix exponential is independent of SDK ODE implementation. Cross-solver tests share the SDK equations, and cannot validate those equations biologically.",
        "parameter_ranges":{"basal_insulin_rate":[.5,1.],"insulin_sensitivity":[40.,60.],"carb_factor":[9.,13.],"initial_glucose":[130.,160.],"meal_grams":[45.,75.]},
        "fixed_parameters":{"glucose_decay_rate":.02,"glucose_absorption_rate":.03,"insulin_action_duration":300.,"max_glucose_rate_mgdl_per_min":3.},
        "scenario_events":"Meals at 60,420,780 min. Exercise intensity .6 at 480 for 60 min. Sensor_fault: +25 bias, SD12, dropout .05, duration2-4 samples. Otherwise SD8.5, no bias/dropout.",
        "literature_context":"Ruan peak and variability are contextual references only: adult SDK defaults do not match the pediatric clinical study.",
        "data_rule":"Local files require consistency and a traceable source chain. File labels alone cannot establish true patient origin. No fitting to unverifiable snippets.",
        "limits":"No clinical success verdict. No new training, no calibrating to outcomes, no production changes. Protocol was formulated after the earlier exploratory stress review, not retrospectively registered as an independent clinical preregistration.",
        "sources":SOURCES,
    }


def emit_check(checks, family, name, value, threshold, passed, model_name="all", scope="equation_verification", detail=""):
    checks.append(dict(family=family, name=name, model=model_name, value=value, criterion=threshold,
                       status="PASS" if passed else "FAIL", scope=scope, detail=detail))


def data_audit(out, checks):
    rows=[]
    for p in sorted((ROOT/"data_packs/public").rglob("timeseries.csv")):
        frame=pd.read_csv(p)
        meta_path=p.with_name("metadata.yaml")
        metadata=yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        declared=metadata.get("patient_profile",{}).get("data_points")
        t=pd.to_datetime(frame.timestamp)
        rows.append(dict(file=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         rows=len(frame),declared_rows=declared,hours=(t.max()-t.min()).total_seconds()/3600,
                         missing_glucose=int(frame.glucose_mg_dl.isna().sum()),duplicates=int(t.duplicated().sum()),
                         basal_column=any("basal" in c.lower() for c in frame),
                         metadata_regimen=metadata.get("patient_profile",{}).get("demographics",{}).get("insulin_regimen"),
                         source_url_present=bool(metadata.get("source_url") or metadata.get("download_url")),
                         verification="NOT_ESTABLISHED"))
    df=pd.DataFrame(rows)
    df.to_csv(out/"data_provenance_audit.csv",index=False)
    consistent=bool(len(df) and (df.rows==df.declared_rows).all())
    emit_check(checks,"data","local_metadata_consistency",int((df.rows==df.declared_rows).sum()) if len(df) else 0,
               "declared row count matches available file; origin chain required",consistent,
               scope="empirical_evidence_gate",detail="Afwijkende aantallen kunnen een extract betekenen, maar dat is niet gedocumenteerd. MDI-metadata wijkt af van de officiële Ohio-pomppopulatie. Geen bewijs dat deze bestanden volledige of geverifieerde patiëntdata zijn.")
    return df


def fasting(out, checks):
    rows=[]
    for name in MODELS:
        for g in (110.,150.):
            for basal in (.4,.8,1.2):
                p=model(name,initial_glucose=g,basal_insulin_rate=basal)
                values=[g]
                status="completed"
                for t in range(0,1440,5):
                    try:
                        values.append(p.update(5.,basal*5/60,current_time=float(t)))
                    except (PatientModelDomainError,RuntimeError) as exc:
                        status=str(exc)
                        break
                drift=float(np.max(np.abs(np.asarray(values)-g)))
                rows.append(dict(model=name,initial_glucose=g,basal=basal,max_abs_drift=drift,
                                 min_glucose=min(values),minutes_observed=(len(values)-1)*5,completed=status=="completed",reason=status))
                emit_check(checks,"basal",f"fasting_G{g}_basal{basal}",drift,"24h complete and drift <=1 mg/dL",
                           status=="completed" and drift<=1,name,
                           scope="exploratory_backend_limitation" if name=="custom" else "mechanistic_equilibrium")
    df=pd.DataFrame(rows);df.to_csv(out/"fasting.csv",index=False);return df


def analytic_components(out, checks):
    rows=[];peak_rows=[]
    for name in MECHANISTIC:
        p=model(name,initial_glucose=160.,basal_insulin_rate=.8)
        if name=="hovorka":
            idx=[2,3,4]; k=1/p._insulin_tmax_minutes(); ke=p.params.k_e; volume=p.params.V_I_per_kg*p.params.body_weight_kg
            meal_idx=[8,9]; rates=[1/p.params.t_max_G]*2; bio=p.params.A_G
        else:
            idx=[6,7,2]; k=p.params.k_a; ke=p.params.n; volume=p.params.Vi*p.params.body_weight_kg
            meal_idx=[3,4,5];rates=[1.5/p.params.tau_meal,1/p.params.tau_meal,p.params.k_abs];bio=p.params.f_bio
        # Augmented affine system: exact constant-infusion solution exp(M*dt).
        a=np.zeros((4,4));a[:3,:3]=[[-k,0,0],[k,-k,0],[0,k/volume,-ke]];a[0,3]=1.
        y=np.array([*p._state[idx],0.])
        initial_insulin=float(y[2]); meal=20.; m=len(meal_idx)
        meal_matrix=np.zeros((m+1,m+1))
        for j, rate in enumerate(rates):
            meal_matrix[j,j]=-rate
            meal_matrix[j+1,j]=rate
        injected=meal*1000*(bio if name!="hovorka" else 1.)
        meal_state=np.zeros(m+1);meal_state[0]=injected
        errors=[];meal_errors=[];balance_errors=[];plasma=[]
        for t in range(0,360,5):
            dose=.8*5/60+(.5 if t==0 else 0.)
            y[3]=dose*1000/5
            y=linalg.expm(a*5)@y
            meal_state=linalg.expm(meal_matrix*5)@meal_state
            p.update(5.,dose,carb_intake=meal if t==0 else 0.,current_time=float(t))
            actual=p._state[idx]
            actual_meal=p._state[meal_idx]
            scaled=float(np.max(np.abs(actual-y[:3])/np.maximum(1.,np.abs(y[:3]))))
            me=float(np.max(np.abs(actual_meal-meal_state[:m])))/injected
            balance=abs(float(meal_state.sum())-injected)/injected
            errors.append(scaled);meal_errors.append(me);balance_errors.append(balance)
            plasma.append(float(actual[2]-initial_insulin))
            rows.append(dict(model=name,time=t+5,insulin_s1=actual[0],oracle_s1=y[0],insulin_s2=actual[1],oracle_s2=y[1],
                             plasma_increment=plasma[-1],oracle_plasma_increment=y[2]-initial_insulin,
                             reported_iob=p.insulin_on_board,scaled_pk_error=scaled,meal_error=me,
                             remaining_meal_mg=float(actual_meal.sum()),oracle_remaining_mg=float(meal_state[:m].sum()),
                             mass_balance_relative=balance))
        emit_check(checks,"analytic","insulin_pk_expm",max(errors),"scaled error <=1e-5",max(errors)<=1e-5,name)
        emit_check(checks,"analytic","meal_chain_expm",max(meal_errors),"error / injected mass <=1e-5",max(meal_errors)<=1e-5,name)
        emit_check(checks,"analytic","reference_mass_conservation",max(balance_errors),"reference residual <=1e-10",max(balance_errors)<=1e-10,name,
                   scope="oracle_self_check",detail="Controle van de onafhankelijke massabalans; SDK-compartimenten worden afzonderlijk met de oracle vergeleken.")
        peak_rows.append(dict(model=name,plasma_peak_time_min=5*(int(np.argmax(plasma))+1),
                              source_mean_min=59.2,source_sd_min=14.4,population_match=False))
    df=pd.DataFrame(rows);df.to_csv(out/"analytic_compartments.csv",index=False)
    peaks=pd.DataFrame(peak_rows);peaks.to_csv(out/"literature_pk_context.csv",index=False)
    return df,peaks


def numerical(out, checks):
    rows=[];traces=[]
    for name in MECHANISTIC:
        for condition in ("fasting","meal","meal_bolus"):
            method_states={};method_g={}
            for method in ("SDK_RK45","DOP853","Radau"):
                p=model(name,initial_glucose=140.,basal_insulin_rate=.8)
                states=[];gs=[]
                with solver_context(name,method):
                    for t in range(0,360,5):
                        dose=.8*5/60+(2. if condition=="meal_bolus" and t==60 else 0.)
                        carbs=40. if condition!="fasting" and t==60 else 0.
                        p.update(5.,dose,carbs,current_time=float(t))
                        states.append(p._state.copy());gs.append(p.get_current_glucose())
                        traces.append(dict(model=name,condition=condition,method=method,time=t+5,glucose=gs[-1]))
                method_states[method]=np.array(states);method_g[method]=np.array(gs)
            for method in ("SDK_RK45","Radau"):
                glucose_error=float(np.max(np.abs(method_g[method]-method_g["DOP853"])))
                state_error=float(np.max(np.abs(method_states[method]-method_states["DOP853"])/np.maximum(1.,np.abs(method_states["DOP853"]))))
                rows.append(dict(model=name,condition=condition,method=method,max_glucose_error=glucose_error,max_scaled_state_error=state_error))
                emit_check(checks,"solver",f"{condition}_{method}",glucose_error,"G<=0.1 mg/dL and scaled state error<=1e-4",
                           glucose_error<=.1 and state_error<=1e-4,name,scope="numerical_verification_shared_ode")
            # Same rectangular bolus delivery duration (15 min), event-aligned grids.
            trajectories={}
            for dt in (1,5,15):
                p=model(name,initial_glucose=140.,basal_insulin_rate=.8)
                vals=[]
                for t in range(0,360,dt):
                    extra=2.*dt/15 if condition=="meal_bolus" and 60<=t<75 else 0.
                    carbs=40. if condition!="fasting" and t==60 else 0.
                    vals.append(p.update(dt,.8*dt/60+extra,carbs,current_time=float(t)))
                trajectories[dt]=(np.arange(dt,361,dt),np.array(vals))
            for dt in (5,15):
                ts,vs=trajectories[dt]
                ref=np.interp(ts,*trajectories[1])
                err=float(np.max(np.abs(vs-ref)))
                rows.append(dict(model=name,condition=condition,method=f"grid_{dt}min_vs_1min",max_glucose_error=err,max_scaled_state_error=np.nan))
                emit_check(checks,"timestep",f"{condition}_dt{dt}",err,"matched inputs: max G error<=0.1 mg/dL",err<=.1,name,scope="time_discretization")
        print(f"numerical + convergence: {name}",flush=True)
    pd.DataFrame(traces).to_csv(out/"solver_traces.csv",index=False)
    df=pd.DataFrame(rows);df.to_csv(out/"numerical_comparisons.csv",index=False);return df


def device_tests(out, checks, seed):
    step=[]
    for dt in (1,5,15):
        sensor=SensorModel(noise_std=0.,lag_minutes=0,isf_tau_minutes=5.)
        sensor.read(100.,0.)
        for t in range(dt,121,dt):
            value=sensor.read(180.,float(t)).value
            oracle=180.-80.*np.exp(-t/5.)
            step.append(dict(dt=dt,time=t,actual=value,oracle=oracle,error=abs(value-oracle)))
    step=pd.DataFrame(step)
    emit_check(checks,"sensor","constant_step_exact",float(step.error.max()),"absolute error <=1e-10",step.error.max()<=1e-10)
    rows=[]
    for rep in range(100):
        sensor=SensorModel(noise_std=8.,noise_ar1_phi=.85,noise_fbm_hurst=None,lag_minutes=0,seed=seed+rep)
        noise=np.array([sensor.read(120.,float(t*5)).value-120. for t in range(1500)])[200:]
        rows.append(dict(rep=rep,mean=noise.mean(),sd=noise.std(ddof=1),acf1=np.corrcoef(noise[:-1],noise[1:])[0,1]))
    noise=pd.DataFrame(rows)
    mean=float(noise['mean'].mean());sd=float(noise.sd.mean());acf=float(noise.acf1.mean())
    emit_check(checks,"sensor","AR1_stationary_mean",mean,"abs(mean)<=0.5 mg/dL",abs(mean)<=.5,scope="stochastic_model_verification")
    emit_check(checks,"sensor","AR1_stationary_sd",sd/8,"0.95<=SD/configured_SD<=1.05",.95<=sd/8<=1.05,scope="stochastic_model_verification")
    emit_check(checks,"sensor","AR1_lag1",acf,"abs(ACF1-0.85)<=0.02",abs(acf-.85)<=.02,scope="stochastic_model_verification")
    pumps=[]
    for rep in range(100):
        pump=PumpModel(max_units_per_step=.975,quantization_units=.05,delivery_noise_std=.2,seed=seed+rep)
        for _ in range(100):
            v=pump.deliver(1.,5.).delivered_units
            pumps.append(dict(value=v,cap_violation=v>.975+1e-12,grid_violation=abs(v/.05-round(v/.05))>1e-9))
    pumps=pd.DataFrame(pumps)
    emit_check(checks,"pump","cap_is_hard",int(pumps.cap_violation.sum()),"0 cap violations",not pumps.cap_violation.any())
    emit_check(checks,"pump","cap_and_quantization_compatible",int(pumps.grid_violation.sum()),"every final command remains a quantization multiple",not pumps.grid_violation.any(),
               scope="engineering_boundary",detail="Non-grid cap 0.975 U with 0.05 U quantization; final cap can create a half-step. Tests the claimed final quantization contract, not a real motor.")
    step.to_csv(out/"sensor_step.csv",index=False);noise.to_csv(out/"sensor_noise_replicates.csv",index=False);pumps.to_csv(out/"pump_boundary.csv",index=False)
    return step,noise,pumps


def controller_grid(out, checks):
    rows=[]
    for arm, constructor in ARMS.items():
        totals=[]
        for dt in (1,5,15):
            algorithm=constructor()
            # Constant error and no IOB: isolates sample-period dependence.
            data=AlgorithmInput(current_glucose=121.,time_step=float(dt),insulin_on_board=0.,
                                basal_rate_u_per_hr=.8,carb_intake=0.,glucose_trend_mgdl_min=0.)
            total=sum(algorithm.predict_insulin(data)["total_insulin_delivered"] for _ in range(0,60,dt))
            totals.append(total);rows.append(dict(arm=arm,dt=dt,hour_total_units=total))
        variation=(max(totals)-min(totals))/max(1e-12,totals[1])
        emit_check(checks,"controller",f"{arm}_sample_period",variation,"max-min / 5min total <=5%",variation<=.05,
                   model_name=arm,scope="sample_time_portability",detail="Unchanged gains; no supervisor. Failure means dt changes are not interchangeable without retuning, not that a tuned 5-min controller is necessarily unusable.")
    df=pd.DataFrame(rows);df.to_csv(out/"controller_sample_period.csv",index=False);return df


def snapshot_test(out, checks):
    rows=[]
    for name in MECHANISTIC:
        p=model(name,initial_glucose=150.)
        sim=Simulator(p,ConstantDoseAlgorithm(dose=0.),seed=4,sensor_model=SensorModel(noise_std=0.,lag_minutes=0))
        sim.add_stress_event(StressEvent(0,"meal",20.))
        frame, report=sim.run_batch(120)
        if name=="hovorka":
            compartment_g=frame.patient_state_Q1/p._glucose_volume_dl()
        else:
            compartment_g=frame.patient_state_G
        discrepancy=float(np.max(np.abs(frame.glucose_actual_mgdl-compartment_g)))
        # Post-step compartment at t equals next row's pre-step glucose.
        shift_error=float(np.max(np.abs(compartment_g.iloc[:-1].to_numpy()-frame.glucose_actual_mgdl.iloc[1:].to_numpy())))
        rows.extend(dict(model=name,time=float(t),reported_g=float(g),compartment_g=float(c),difference=float(c-g))
                    for t,g,c in zip(frame.time_minutes,frame.glucose_actual_mgdl,compartment_g))
        emit_check(checks,"timeline","same_row_glucose_snapshot",discrepancy,"abs(truth-compartment G)<=0.1 mg/dL",discrepancy<=.1,name,
                   scope="export_temporal_consistency",detail=f"Next-row alignment error {shift_error:.3g} mg/dL. Since defect D1 was fixed every state column in a row shares the row's timestamp; the state after the interval is exported separately.")
        emit_check(checks,"timeline","exact_duration_integration",p._last_ode_time,"ODE ends at requested duration 120 min",abs(p._last_ode_time-120.)<1e-9,name,
                   scope="horizon_contract",detail="Since defect D2 was fixed the loop stops before the endpoint, so no extra interval or terminal delivery is executed; the endpoint observation is available through Simulator.endpoint_observation().")
    df=pd.DataFrame(rows);df.to_csv(out/"export_timeline.csv",index=False);return df


def paired_cohort(out, config, checks):
    n=config['cohort_n'];seed=config['seed'];bounds=np.array(list(config['parameter_ranges'].values()))
    samples=qmc.scale(qmc.LatinHypercube(len(bounds),seed=seed).random(n),bounds[:,0],bounds[:,1])
    params=pd.DataFrame(samples,columns=list(config['parameter_ranges']))
    params.to_csv(out/"paired_parameters.csv",index_label="patient_id")
    rows=[];traces=[]
    for name in MODELS:
        for scenario in SCENARIOS:
            for pid, param in params.iterrows():
                for arm, constructor in ARMS.items():
                    kwargs={**config['fixed_parameters'],**param.to_dict()};meal=kwargs.pop('meal_grams')
                    p=model(name,**kwargs)
                    sensor=SensorModel(seed=seed+pid,noise_std=12. if scenario=='sensor_fault' else 8.5,
                                       bias=25. if scenario=='sensor_fault' else 0.,dropout_prob=.05 if scenario=='sensor_fault' else 0.,dropout_duration_steps=(2,4))
                    sim=Simulator(p,constructor(),seed=seed+pid,sensor_model=sensor)
                    for t,mult in [(60,1.),(420,1.2),(780,1.)]:
                        sim.add_stress_event(StressEvent(t,'meal',meal*mult))
                    if scenario=='exercise':sim.add_stress_event(StressEvent(480,'exercise',.6,duration=60))
                    unexpected=None
                    try:
                        frame,report=sim.run_batch(1440)
                    except Exception as exc:
                        unexpected=f'{type(exc).__name__}: {exc}';frame=pd.DataFrame();report={}
                    completed=len(frame)==289 and not report.get('terminated_early',False) and unexpected is None
                    observed=frame.loc[frame.time_minutes<1440] if not frame.empty else frame
                    g=observed.glucose_actual_mgdl.to_numpy() if not observed.empty else np.array([])
                    tir_count=int(((g>=70)&(g<=180)).sum());denominator=288
                    fail=(not completed) or bool(np.any(g<54.))
                    rows.append(dict(model=name,scenario=scenario,patient_id=pid,arm=arm,completed=completed,composite_failure=fail,
                                     observed_steps=len(g),tir_observed=100*tir_count/max(1,len(g)),
                                     tir_full_lower=100*tir_count/denominator,tir_full_upper=100*(tir_count+denominator-len(g))/denominator,
                                     below54_observed_pct=100*np.mean(g<54.) if len(g) else np.nan,
                                     min_g=float(g.min()) if len(g) else np.nan,
                                     termination_reason=json.dumps(report.get('termination_reason',unexpected)),
                                     **param.to_dict()))
                    if not frame.empty:
                        keep=[c for c in ['time_minutes','glucose_actual_mgdl','glucose_to_algo_mgdl','delivered_insulin_units','safety_triggered','input_validator_fail_soft','patient_iob_units'] if c in frame]
                        trace=frame[keep].copy();trace['model']=name;trace['arm']=arm;trace['scenario']=scenario;trace['patient_id']=pid
                        traces.append(trace)
            print(f"paired control: {name}/{scenario} ({n*3} runs)",flush=True)
    runs=pd.DataFrame(rows);runs.to_csv(out/'paired_runs.csv',index=False)
    pd.concat(traces,ignore_index=True).to_csv(out/'paired_traces.csv.gz',index=False,compression='gzip')
    comparisons=[]
    for name in MODELS:
        for arm in ['basal_bolus','pid']:
            # One inference unit per LHS patient, after averaging 3 scenario outcomes.
            group=runs[runs.model==name].groupby(['patient_id','arm'])[['tir_full_lower','composite_failure']].mean()
            d=group.xs(arm,level='arm').tir_full_lower.to_numpy()-group.xs('clinical',level='arm').tir_full_lower.to_numpy()
            risk=group.xs(arm,level='arm').composite_failure.to_numpy()-group.xs('clinical',level='arm').composite_failure.to_numpy()
            ci=stats.bootstrap((d,),np.mean,method='percentile',n_resamples=10000,rng=np.random.default_rng(seed)).confidence_interval
            rci=stats.bootstrap((risk,),np.mean,method='percentile',n_resamples=10000,rng=np.random.default_rng(seed)).confidence_interval
            perm=stats.permutation_test((d,),lambda x:np.mean(x),permutation_type='samples',n_resamples=9999,
                                        alternative='two-sided',rng=np.random.default_rng(seed))
            comparisons.append(dict(model=name,arm=arm,n_patients=n,tir_delta_pp=d.mean(),ci_low=ci.low,ci_high=ci.high,
                                    failure_delta_pp=100*risk.mean(),failure_ci_low=100*rci.low,failure_ci_high=100*rci.high,
                                    permutation_p=perm.pvalue,noninferiority_demonstrated=ci.low>-5. and rci.high<=.05))
    result=pd.DataFrame(comparisons)
    order=np.argsort(result.permutation_p.to_numpy());adjusted=np.zeros(len(result));previous=0.
    for rank,idx in enumerate(order):
        previous=max(previous,min(1.,float(result.iloc[idx].permutation_p)*(len(result)-rank)));adjusted[idx]=previous
    result['holm_p']=adjusted;result.to_csv(out/'paired_comparisons.csv',index=False)
    return runs,result


def charts(out, datasets):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    gallery=[]
    def save(fig,name,title,note):
        fig.savefig(out/'figures'/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)
        gallery.append(dict(src=f'figures/{name}.png',title=title,note=note))
    data,fasting,analytic,peaks,numeric,step,noise,pumps,grid,timeline,runs,comparisons=datasets
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.bar(np.arange(len(data))-.2,data.rows,.4,label='Beschikbaar',color='#2463a6');ax.bar(np.arange(len(data))+.2,data.declared_rows,.4,label='Metadata',color='#b18620')
    ax.set_xticks(np.arange(len(data)),[Path(f).parent.name for f in data.file]);ax.set(ylabel='Aantal rijen',title='Lokale bronbestanden versus metadata');ax.legend()
    save(fig,'01_data_gate','Herkomstcontrole','20, 85 en 85 beschikbare rijen tegenover 16.128 per metadatabestand; patiëntvalidatie niet vastgesteld.')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for name in MODELS:
        f=fasting[fasting.model==name];ax.scatter(f.basal,f.max_abs_drift,label=name,color=COLORS[name],marker='x',s=55)
    ax.axhline(1.,color='black',ls='--');ax.set(xlabel='Geprogrammeerde basal (U/uur)',ylabel='Maximale afwijking (mg/dL)',title='24-uurs basaal evenwicht');ax.legend()
    save(fig,'02_fasting','Basaal evenwicht','Fysiologische balanshypothese bij twee beginglucoses; voortijdige stops blijven mislukt. Custom is hier een verkennende beperking.')
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,name in zip(axes,MECHANISTIC):
        f=analytic[analytic.model==name];ax.plot(f.time,f.plasma_increment,label='SDK',color=COLORS[name]);ax.plot(f.time,f.oracle_plasma_increment,'--',color='black',label='Matrix-exponentiële oracle');ax.set(title=name,xlabel='Tijd (min)',ylabel='Plasma-insuline boven basal (mU/L)');ax.legend(fontsize=8)
    save(fig,'03_pk_oracle','Insuline-PK versus exacte lineaire oplossing','Zelfde volumeeenheden en rechthoekige infusie; onafhankelijke vergelijking van compartimenten, geen patiëntvalidatie.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for name in MECHANISTIC:
        f=analytic[analytic.model==name];axes[0].plot(f.time,f.remaining_meal_mg/1000,label=name,color=COLORS[name]);axes[1].semilogy(f.time,np.maximum(f.meal_error,1e-16),label=name,color=COLORS[name])
    axes[0].set(xlabel='Tijd (min)',ylabel='Maaltijdmassa in de keten (g)',title='Gastro-intestinale restmassa');axes[1].set(xlabel='Tijd (min)',ylabel='Genormaliseerde oracle-afwijking',title='Afwijking van matrix-exponentiële oplossing');axes[1].axhline(1e-5,color='black',ls='--');axes[0].legend()
    save(fig,'04_meal_mass','Maaltijdcompartimenten','Bergman/Advanced injecteren bioavailable massa; Hovorka past bioavailability toe op uitstroom. Verschillende invoersemantiek is expliciet meegenomen.')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for name in MECHANISTIC:
        f=numeric[(numeric.model==name)&~numeric.method.str.startswith('grid')];ax.scatter(np.arange(len(f)),np.maximum(f.max_glucose_error,1e-14),label=name,color=COLORS[name])
    ax.axhline(.1,color='black',ls='--');ax.set(yscale='log',ylabel='Maximale glucoseafwijking (mg/dL)',title='SDK RK45 en Radau tegenover strakke DOP853');ax.legend()
    ax.set_xticks(range(6),['fasting\nRK45','fasting\nRadau','meal\nRK45','meal\nRadau','meal + bolus\nRK45','meal + bolus\nRadau'])
    save(fig,'05_solvers','Solververgelijking','Drie inputcondities. Alle solvers delen de SDK-vergelijkingen; dit verifieert integratie, geen biologische juistheid.')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for name in MECHANISTIC:
        f=numeric[numeric.model.eq(name)&numeric.method.str.startswith('grid')];ax.plot(range(len(f)),np.maximum(f.max_glucose_error,1e-14),'o-',label=name,color=COLORS[name])
    ax.axhline(.1,color='black',ls='--');ax.set(yscale='log',ylabel='Max. afwijking versus 1 min (mg/dL)',title='Tijdstapconvergentie met afgestemde infusieduur');ax.legend()
    ax.set_xticks(range(6),['fasting\n5 min','fasting\n15 min','meal\n5 min','meal\n15 min','meal + bolus\n5 min','meal + bolus\n15 min'])
    save(fig,'06_grid_convergence','Tijdstapconvergentie','Maaltijdevent is exact uitgelijnd; bolus duurt dezelfde 15 minuten op elk raster. Log-schaal; afwijkingen kleiner dan 1e-14 zijn op 1e-14 geplot.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for dt in (1,5,15):
        f=step[step.dt==dt];axes[0].plot(f.time,f.actual,label=f'dt={dt} min')
    axes[0].set(xlabel='Tijd sinds stap (min)',ylabel='Sensorwaarde (mg/dL)',title='Ruisvrije eerste-orde staprespons');axes[0].legend()
    axes[1].hist(noise.sd,bins=20,color='#2463a6');axes[1].axvline(8,color='black',ls='--');axes[1].set(xlabel='SD per onafhankelijke ruisrealisatie (mg/dL)',ylabel='Aantal realisaties',title='100 AR(1)-ruisrealisaties')
    save(fig,'07_sensor','Sensorvergelijkingen','Exacte stapoplossing en 100 onafhankelijke seeded AR(1)-reeksen met burn-in. Geen apparaatkalibratie.')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for arm,color in zip(ARMS,['#2463a6','#b18620','#c36832']):
        f=grid[grid.arm==arm];ax.plot(f.dt,f.hour_total_units,'o-',label=arm,color=color)
    ax.set(xlabel='Controllertijdstap (min)',ylabel='Insulinevoorstel in 60 min (U)',title='Sampleperiode bij dezelfde constante input');ax.legend()
    save(fig,'08_controller_grid','Controllerafhankelijkheid van sampletijd','Zonder supervisor, constante fout 1 mg/dL, vaste gains. Alleen portability getest; geen klinische dosisvergelijking.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for name in MECHANISTIC:
        f=timeline[timeline.model==name];axes[0].plot(f.time,f.difference,label=name,color=COLORS[name])
    f=timeline[timeline.model=='hovorka'];axes[1].plot(f.time,f.reported_g,label='Glucosewaarheid: vóór stap');axes[1].plot(f.time,f.compartment_g,'--',label='Q1/volume: na stap')
    axes[0].set(xlabel='Gelabelde tijd (min)',ylabel='Compartiment-G minus gerapporteerde G (mg/dL)',title='Tijdconsistentie van geëxporteerde rijen');axes[0].legend()
    axes[1].set(xlabel='Gelabelde tijd (min)',ylabel='Glucose (mg/dL)',title='Hovorka faseverschil');axes[1].legend(fontsize=9)
    save(fig,'09_export_time','Snapshot- en horizonconsistentie','Dezelfde rij combineert pre-step glucose met post-step compartimenten; controleer tijdlabels vóór wetenschappelijke analyses.')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    summary=runs.groupby(['model','arm'],sort=False).agg(completion=('completed','mean'),failure=('composite_failure','mean'))
    labels=[f'{m}/{a}' for m,a in summary.index]
    axes[0].barh(labels,100*summary.completion,color='#2463a6');axes[1].barh(labels,100*summary.failure,color='#c36832')
    axes[0].set(xlim=(0,100),xlabel='Volledig afgerond (%)',title='Volledigheid - alle runs');axes[1].set(xlim=(0,100),xlabel='Samengesteld falen (%)',title='Afgebroken of waargenomen glucose <54')
    save(fig,'10_paired_outcomes','Gepaarde controlleruitkomsten','Alle 3 scenario’s per patiënt; mislukte runs blijven behouden. Samengesteld falen is een engineeringendpoint, geen incidentie in mensen.')
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for j,r in comparisons.iterrows():
        ax.errorbar(r.tir_delta_pp,j,xerr=[[r.tir_delta_pp-r.ci_low],[r.ci_high-r.tir_delta_pp]],fmt='o',color=COLORS[r.model],capsize=4)
    ax.axvline(0,color='black');ax.axvline(-5,color='black',ls='--');ax.set_yticks(range(len(comparisons)),[f'{r.model}/{r.arm}' for _,r in comparisons.iterrows()]);ax.set(xlabel='TIR-ondergrens verschil versus Clinical (procentpunt)',title='Gepaarde verschillen met 95% cluster-bootstrapinterval')
    save(fig,'11_paired_inference','Gepaarde inferentie','Eén eenheid per virtuele patiënt, 3 scenario’s gemiddeld. 10.000 resamples; secundaire conservatieve TIR-ondergrens. Holm-p-waarden in CSV.')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    grouped=runs.groupby(['model','arm'],sort=False)[['tir_full_lower','tir_full_upper']].mean()
    for j,((m,a),r) in enumerate(grouped.iterrows()):
        ax.plot([r.tir_full_lower,r.tir_full_upper],[j,j],color=COLORS[m],lw=4)
        ax.scatter([r.tir_full_lower,r.tir_full_upper],[j,j],color=COLORS[m],s=30,zorder=3)
    ax.set_yticks(range(len(grouped)),[f'{m}/{a}' for m,a in grouped.index]);ax.set(xlim=(0,100),xlabel='TIR voor volledig 24-uursvenster (%)',title='Identificeerbare TIR-grenzen bij uitval')
    save(fig,'12_missing_bounds','TIR-onzekerheid door ontbrekende trajecten','Ondergrens: ontbrekende stappen buiten gebied; bovengrens: ontbrekende stappen binnen gebied. Dit zijn grenzen, geen statistische betrouwbaarheidsintervallen.')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.axhspan(59.2-14.4,59.2+14.4,color='#b18620',alpha=.16,label='Ruan U100 gemiddelde +/- SD, N=11 kinderen')
    ax.scatter(peaks.model,peaks.plasma_peak_time_min,color='#2463a6',s=65)
    ax.set(ylabel='Plasma-insulinepiek (min)',title='Literatuurcontext voor ongetunede standaard-PK');ax.legend(fontsize=9)
    save(fig,'13_pk_literature','Externe PK-context','Bron: Ruan et al. De populatie en het protocol zijn niet gematcht: geen pass/fail en geen klinische validatie op basis van deze band.')
    return gallery


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--patients',type=int,default=16);ap.add_argument('--seed',type=int,default=20261013)
    ap.add_argument('--output',type=Path,default=ROOT/'results/scientific_validation');args=ap.parse_args()
    if args.patients<8:ap.error('at least 8 paired patients required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Use a fresh output directory: frozen protocols must not be overwritten')
    config=protocol(args.patients,args.seed);raw=json.dumps(config,indent=2,allow_nan=False)
    # Freeze exact UTF-8 bytes, including line endings, before execution.
    (out/'protocol.json').write_bytes(raw.encode('utf-8'));(out/'protocol.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest(),encoding='utf-8')
    logging.getLogger('iints').setLevel(logging.CRITICAL);start=time.perf_counter();checks=[]
    data=data_audit(out,checks);print('provenance audit complete',flush=True)
    fasting_data=fasting(out,checks);print('24-hour fasting matrix complete',flush=True)
    analytic,peaks=analytic_components(out,checks);print('independent matrix-exponential references complete',flush=True)
    numeric=numerical(out,checks)
    step,noise,pumps=device_tests(out,checks,args.seed);grid=controller_grid(out,checks);timeline=snapshot_test(out,checks)
    pd.DataFrame(checks).to_csv(out/'scientific_checks.csv',index=False)
    print('core scientific checks:',pd.DataFrame(checks).status.value_counts().to_dict(),flush=True)
    runs,comparisons=paired_cohort(out,config,checks)
    gallery=charts(out,(data,fasting_data,analytic,peaks,numeric,step,noise,pumps,grid,timeline,runs,comparisons))
    pd.DataFrame(checks).to_csv(out/'scientific_checks.csv',index=False)
    record={'protocol_sha256':hashlib.sha256(raw.encode()).hexdigest(),'elapsed_s':time.perf_counter()-start,
            'python':sys.version,'platform':platform.platform(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'package_versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','matplotlib']},
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*(ROOT/'src/iints/core').rglob('*.py')]},
            'run_count':len(runs),'completed':int(runs.completed.sum()),'composite_failure':int(runs.composite_failure.sum()),'checks':checks,'sources':SOURCES}
    (out/'manifest.json').write_text(json.dumps(record,indent=2,default=float),encoding='utf-8')
    (out/'gallery.json').write_text(json.dumps(gallery,indent=2),encoding='utf-8')
    print(json.dumps({'runs':len(runs),'completed':int(runs.completed.sum()),'checks':pd.DataFrame(checks).status.value_counts().to_dict(),'output':str(out)},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
