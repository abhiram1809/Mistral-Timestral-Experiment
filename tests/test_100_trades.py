import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from simulate_100_trades import trade_returns


def test_long_trade_charges_both_fills():
    gross,net=trade_returns([1],[100.],[110.],10)
    assert gross[0] == pytest.approx(.1)
    assert 1+net[0] == pytest.approx(1.1*(1-.001)/(1+.001))


def test_short_trade_and_unchanged_price_have_explicit_costs():
    gross,net=trade_returns([-1,1],[100.,100.],[90.,100.],10)
    np.testing.assert_allclose(gross,[.1,0])
    np.testing.assert_allclose(net,[(.1-.001*1.9)/1.001,-.002/1.001])


def test_zero_friction_preserves_model_pnl():
    gross,net=trade_returns([1,-1],[100.,100.],[110.,110.],0)
    np.testing.assert_allclose(gross,net)
