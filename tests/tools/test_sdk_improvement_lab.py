"""Integrity checks for the reusable synthetic experiment bank."""
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def runner():
    path=Path(__file__).resolve().parents[2]/'tools/analysis/sdk_improvement_lab.py'
    spec=importlib.util.spec_from_file_location('sdk_improvement_lab',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


class Patient:
    def __init__(self,abort_at=None):
        self.glucose=120.;self.insulin_on_board=0.;self.carbs_on_board=0.;self.abort_at=abort_at
    def get_current_glucose(self):return self.glucose
    def update(self,dt,dose,carbs,current_time):
        if self.abort_at is not None and current_time>=self.abort_at:raise RuntimeError('planned domain boundary')
        return self.glucose


def prepare(m,abort_at=None):
    class Factory:
        @staticmethod
        def create_patient(**kwargs):return Patient(abort_at)
    m.SDK={'factory':Factory}


def job():
    return dict(run_id='integrity_test',family='parameter_space',model='fake',condition='fasting',
                patient_params={'basal_insulin_rate':.8},dt_min=5,duration_min=20,patient_id=0,seed=1)


def test_exact_horizon_has_no_terminal_extra_interval():
    m=runner();prepare(m)
    frame,completed,reason,g=m.open_loop(job())
    assert completed and reason is None
    assert frame.observation_time_min.tolist()==[0,5,10,15,20]
    assert len(g)==4
    assert frame.interval_end_min.max()==20


def test_partial_run_has_bounds_and_original_receipt(tmp_path):
    m=runner();prepare(m,abort_at=10);(tmp_path/'receipts').mkdir()
    result=m.run_job(job(),tmp_path,'source_hash')
    assert not result['completed']
    assert result['observed_min']==10
    assert result['tir_full_lower']==50
    assert result['tir_full_upper']==100
    assert result['termination']['attempted_interval_start_min']==10
    trace=tmp_path/result['trace_path']
    assert hashlib.sha256(trace.read_bytes()).hexdigest()==result['trace_sha256']
    assert len(pd.read_csv(trace))==3


def test_resume_reuses_data_and_detects_corruption(tmp_path):
    m=runner();prepare(m);(tmp_path/'receipts').mkdir()
    first=m.run_job(job(),tmp_path,'source_hash')
    assert m.run_job(job(),tmp_path,'source_hash')==first
    trace=tmp_path/first['trace_path'];trace.write_bytes(b'corrupted fixture')
    with pytest.raises(RuntimeError,match='refusing overwrite'):
        m.run_job(job(),tmp_path,'source_hash')
