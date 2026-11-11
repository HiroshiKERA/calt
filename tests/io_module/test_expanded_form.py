"""Tests for the C/E expanded-form conversion."""

from sympy import GF, ring

from calt.io.preprocessor.load_preprocessors.expanded_form import (
    obj_to_expanded_form,
    poly_to_expanded_form,
)


def test_nonzero_polynomial():
    _, x, y = ring("x,y", GF(7))
    assert poly_to_expanded_form(2 * x**2 * y + 3) == "C2 E2 E1 + C3 E0 E0"


def test_zero_polynomial_keeps_exponent_slots():
    R, x, y = ring("x,y", GF(7))
    assert poly_to_expanded_form(R.zero) == "C0 E0 E0"


def test_zero_polynomial_width_follows_number_of_variables():
    R, *_ = ring("x,y,z", GF(7))
    assert poly_to_expanded_form(R.zero) == "C0 E0 E0 E0"


def test_zero_polynomial_in_list():
    R, x, y = ring("x,y", GF(7))
    assert obj_to_expanded_form([x, R.zero]) == "C1 E1 E0 || C0 E0 E0"
