import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from confidence_200 import pnl,profit_threshold,ProfitCalibrator,select_leverage,equity_stats


@pytest.mark.parametrize("direction",[1.,-1.])
@pytest.mark.parametrize("leverage",[1.,2.,3.])
def test_break_even_probability_threshold_matches_exact_cash_pnl(direction,leverage):
    entry=np.array([100.]);years=np.array([3/365.25]);cost=.001
    boundary=profit_threshold(entry,direction,leverage,cost,years)/entry
    net,*_=pnl(boundary,direction,leverage,cost,years)
    np.testing.assert_allclose(net,0,atol=1e-12)
    profit,*_=pnl(boundary+direction*.001,direction,leverage,cost,years)
    assert profit[0]>0


def test_confidence_gates_each_leverage_according_to_its_costs():
    p=np.array([[.54,.59,.69],[.56,.59,.69],[.66,.61,.69],[.76,.72,.71]])
    np.testing.assert_array_equal(select_leverage(p),[0,1,2,3])


def test_calibration_is_monotone_finite_and_frozen():
    p=np.linspace(.01,.99,300)[:,None]*np.ones((1,3))
    y=np.repeat((np.arange(300)%3==0)[:,None],3,axis=1)
    calibrator=ProfitCalibrator().fit(p,y)
    predicted=calibrator.predict(np.linspace(.01,.99,100))
    assert np.all(np.diff(predicted)>=0)
    assert np.isfinite(predicted).all()
    before=calibrator.predict(.7);calibrator.predict(np.array([.9,.1]));assert calibrator.predict(.7)==before


def test_all_losses_do_not_create_high_confidence():
    calibrator=ProfitCalibrator().fit(np.full((300,3),.01),np.zeros((300,3)))
    assert calibrator.predict(.99)<.55


def test_skip_has_no_trading_or_funding_cost():
    net,fees,funding,exposure=pnl(np.array([1.1]),1.,0.,.001,1/365.25)
    np.testing.assert_array_equal(net,0);np.testing.assert_array_equal(fees,0)
    np.testing.assert_array_equal(funding,0);np.testing.assert_array_equal(exposure,0)


def test_equity_compounds_indexed_pandas_returns_in_observation_order():
    stats,equity=equity_stats(pd.Series([.1,-.1],index=[5,9]))
    np.testing.assert_allclose(equity,[110000.,99000.])
    assert stats["ending"]==pytest.approx(99000.)
    assert stats["return"]==pytest.approx(-.01)
    assert stats["max_drawdown"]==pytest.approx(-.1)
