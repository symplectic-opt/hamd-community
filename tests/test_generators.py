"""
Tests for HUBO and portfolio instance generators.

Run:
    pytest tests/test_generators.py -v
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from hamd.generators.cubic_hubo import generate as hubo_generate
from hamd.generators.cubic_portfolio_toy import generate as port_generate
from hamd.core.utils import load_instance


def test_hubo_generator_creates_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, 'cubic_n20.json')
        hubo_generate(n_vars=20, filepath=fp, seed=0, alpha=3.0)
        assert os.path.exists(fp)
        inst = load_instance(fp)
        assert inst['n'] == 20
        assert inst['K'] == 10
        assert len(inst['cubic_terms']) > 0
        assert inst['Q_aug'].shape[0] == inst['n_aug']


def test_portfolio_generator_creates_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, 'cubicport_n50_k10.json')
        port_generate(n=50, K=10, filepath=fp, seed=0)
        assert os.path.exists(fp)
        inst = load_instance(fp)
        assert inst['n'] == 50
        assert inst['K'] == 10
        assert inst['n_aug'] == inst['n'] + inst['n_cubic']


def test_portfolio_community_limit():
    """Generator refuses n > 200 in this release."""
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, 'too_big.json')
        with pytest.raises(ValueError, match='supported range for this release'):
            port_generate(n=300, K=60, filepath=fp)


def test_hubo_augmentation_ratio():
    """n_aug = n + m for HUBO instances."""
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, 'cubic_n30.json')
        hubo_generate(n_vars=30, filepath=fp, seed=1)
        inst = load_instance(fp)
        assert inst['n_aug'] == inst['n'] + len(inst['cubic_terms'])
