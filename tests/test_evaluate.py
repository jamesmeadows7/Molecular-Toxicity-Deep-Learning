"""Tests for model evaluation."""

import numpy as np
import pytest

from tox21.evaluate import per_task_auc


def test_single_task_auc():
    y_true = np.array([[0], [1], [np.nan], [1]])
    y_score = np.array([[0.1], [0.9], [0.5], [0.8]])
    assert per_task_auc(y_true, y_score) == [1.0]


def test_multi_task_auc():
    y_true = np.array(
        [
            [0, 1],
            [1, 0],
            [np.nan, 1],
            [1, 0],
        ]
    )
    y_score = np.array(
        [
            [0.1, 0.2],
            [0.9, 0.7],
            [0.5, 0.3],
            [0.8, 0.6],
        ]
    )
    assert per_task_auc(y_true, y_score) == [1.0, 0.0]


def test_single_class_raises():
    y_true = np.array([[0], [0], [np.nan]])
    y_score = np.array([[0.1], [0.2], [0.3]])
    with pytest.raises(ValueError):
        per_task_auc(y_true, y_score)
