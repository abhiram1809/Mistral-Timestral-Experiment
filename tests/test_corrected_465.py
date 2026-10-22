import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from corrected_465 import calibrated_probability, oracle_history, oracle_count
from confidence_200 import select_leverage


def test_all_losses_cannot_become_high_confidence_from_a_large_raw_score():
    raw=np.full((300,3),.01);wins=np.zeros((300,3),dtype=bool)
    p,*_=calibrated_probability(raw,wins,np.ones(300,dtype=bool),[.99,.99,.99])
    assert np.all(p<.01)
    np.testing.assert_array_equal(select_leverage(p[None,:]),[0])


def test_sparse_perfect_bin_backs_off_instead_of_manufacturing_confidence():
    raw=np.full((300,3),.25);raw[:10]=.75
    wins=np.zeros((300,3),dtype=bool);wins[:10]=True
    p,n,w,fallback=calibrated_probability(raw,wins,np.ones(300,dtype=bool),[.75,.75,.75])
    np.testing.assert_allclose(p,11/302)
    assert n==[300]*3 and w==[10]*3 and all(fallback)


def test_supported_local_wins_can_pass_gates_without_an_old_range_ceiling():
    raw=np.full((300,3),.25);raw[:20]=.75
    wins=np.zeros((300,3),dtype=bool);wins[:16]=True
    p,n,_,fallback=calibrated_probability(raw,wins,np.ones(300,dtype=bool),[.75,.75,.75])
    np.testing.assert_allclose(p,17/22)
    assert n==[20]*3 and not any(fallback)
    np.testing.assert_array_equal(select_leverage(p[None,:]),[3])


def test_higher_leverage_cannot_have_higher_effective_profit_probability():
    raw=np.full((300,3),.5);wins=np.zeros((300,3),dtype=bool)
    wins[:150,0]=True;wins[:200,1]=True;wins[:250,2]=True
    p,*_=calibrated_probability(raw,wins,np.ones(300,dtype=bool),[.5,.5,.5])
    assert (np.diff(p)<=0).all()


def test_oracle_skip_bound_has_a_strict_past_only_embargo():
    returns=np.zeros(699);returns[249:260]=.01
    prices=np.r_[100.,100*np.cumprod(1+returns)]
    times=pd.date_range("2020-01-01",periods=700,tz="UTC")
    prefix,_=oracle_history(prices,times,.001)
    assert oracle_count(prefix,550)==11 and (11+1)/(20+2)<.55
    altered=prices.copy();altered[550:]*=2
    future_prefix,_=oracle_history(altered,times,.001)
    assert oracle_count(future_prefix,550)==11
