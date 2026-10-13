"""Validate the scientific evaluator, including intentional fault injection.

Passing these tests means the evaluator detects controlled faults. SDK scientific
acceptance/failure is recorded separately in scientific_checks.csv.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import hashlib


def runner():
    path=Path(__file__).resolve().parents[2]/'tools/analysis/scientific_validation_suite.py'
    spec=importlib.util.spec_from_file_location('scientific_validation_suite',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_analytic_oracles_agree_with_unmodified_sdk(tmp_path):
    m=runner();checks=[]
    m.analytic_components(tmp_path,checks)
    assert all(c['status']=='PASS' for c in checks)
    assert sum(c['name']=='insulin_pk_expm' for c in checks)==3


def test_pk_oracle_detects_ten_percent_transfer_fault(tmp_path,monkeypatch):
    m=runner()
    from iints.core.patient.bergman_model import BergmanPatientModel
    original=BergmanPatientModel._ode
    def faulty(self,t,y,u,g,current_time,record=None):
        derivative=original(self,t,y,u,g,current_time,record)
        k=self.params.k_a
        volume=self.params.Vi*self.params.body_weight_kg
        derivative[6]-=.1*k*y[6]
        derivative[7]+=.1*k*(y[6]-y[7])
        derivative[2]+=.1*k*y[7]/volume
        return derivative
    monkeypatch.setattr(BergmanPatientModel,'_ode',faulty)
    checks=[];m.analytic_components(tmp_path,checks)
    assert any(c['model']=='bergman' and c['name']=='insulin_pk_expm' and c['status']=='FAIL' for c in checks)


def test_meal_oracle_detects_input_mass_conversion_fault(tmp_path,monkeypatch):
    m=runner()
    from iints.core.patient.hovorka_model import HovorkaPatientModel
    original=HovorkaPatientModel.update
    def faulty(self,time_step,delivered_insulin,carb_intake=0.,**kwargs):
        return original(self,time_step,delivered_insulin,carb_intake*1.1,**kwargs)
    monkeypatch.setattr(HovorkaPatientModel,'update',faulty)
    checks=[];m.analytic_components(tmp_path,checks)
    assert any(c['model']=='hovorka' and c['name']=='meal_chain_expm' and c['status']=='FAIL' for c in checks)


def test_sample_period_probe_separates_invariant_and_discrete_gains(tmp_path):
    m=runner();checks=[]
    data=m.controller_grid(tmp_path,checks)
    assert np.allclose(data[data.arm=='basal_bolus'].hour_total_units,.8)
    assert np.allclose(data[data.arm=='clinical'].hour_total_units,.8)
    # This is a diagnostic of the currently unretuned PID, not a clinical claim.
    assert data[data.arm=='pid'].hour_total_units.nunique()>1


def test_protocol_is_frozen_as_exact_utf8_bytes_before_work(tmp_path,monkeypatch):
    m=runner()
    def stop_after_freeze(*args):
        raise RuntimeError('stop-after-freeze')
    monkeypatch.setattr(m,'data_audit',stop_after_freeze)
    monkeypatch.setattr(m.sys,'argv',['scientific_validation_suite','--patients','8','--output',str(tmp_path)])
    with pytest.raises(RuntimeError,match='stop-after-freeze'):
        m.main()
    frozen=(tmp_path/'protocol.json').read_bytes()
    recorded=(tmp_path/'protocol.sha256').read_text().strip()
    assert hashlib.sha256(frozen).hexdigest()==recorded
    assert b'\r\n' not in frozen
