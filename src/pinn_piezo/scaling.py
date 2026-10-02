"""Characteristic scales for the residual-only mixed piezoelectric PINNs.

The network still predicts the eight physical fields
``(u, v, phi, sigma_xx, sigma_yy, tau_xy, D_x, D_y)``.  Internally its raw
outputs and coordinates are dimensionless, and every equation is divided by
the scale of its own physical quantity before entering the loss.  No energy or
enthalpy functional is used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import materials
from . import config
from .config import HEIGHT, REFERENCE_FORCE, WIDTH


@dataclass(frozen=True)
class CharacteristicScales:
    """Positive reference magnitudes used by a single boundary-value problem."""

    length: float
    height: float
    u: float
    v: float
    phi: float
    stress_x: float
    stress_y: float
    shear: float
    electric_displacement: float
    traction: float

    @property
    def stress(self) -> float:
        """Backward-compatible dominant normal-stress scale."""
        return self.stress_x

    @property
    def equilibrium_x(self) -> float:
        """Scale for ``sigma_xx,x + tau_xy,y``."""
        return max(self.stress_x / self.length, self.shear / self.height)

    @property
    def equilibrium_y(self) -> float:
        """Scale for ``tau_xy,x + sigma_yy,y``."""
        # In a slender beam sigma_yy is itself O(tau H/L), so both terms scale
        # with tau/L.  Using the nominal tau/H scale underweights vertical
        # equilibrium by L/H and lets the applied shear resultant leak.
        return self.shear / self.length

    @property
    def gauss(self) -> float:
        """Scale for ``div(D)``."""
        return self.electric_displacement / self.height


@dataclass(frozen=True)
class ConstitutiveScales:
    """Normalizing scale for each of the five constitutive residuals."""

    sigma_xx: float
    sigma_yy: float
    tau_xy: float
    d_x: float
    d_y: float


def dominant_term_scales(scales: CharacteristicScales) -> ConstitutiveScales:
    """Normalize each constitutive residual by its own largest term.

    A constitutive residual is a *difference*, and dividing it by the scale of
    the quantity it represents is only correct when that quantity is also the
    largest term in the equation.  In a slender beam it is not.  With
    ``L/H = 100`` the shear law

        tau_xy = G (u_y + v_x)

    has ``G u_y ~ G u_c/H`` and ``G v_x ~ G v_c/L`` both of order ``2.1e6 Pa``
    while their difference is the applied traction, ``1e2 Pa``.  Dividing by the
    traction therefore amplifies any relative representation error in ``(u, v)``
    by ``G u_c / (H tau_c) ~ 2.1e4``, and the trivial field ``u = v = 0`` is the
    only field that pays nothing.  The transverse normal law suffers the same
    defect, by a factor ``174``.

    Normalizing by the dominant term restores unit sensitivity: a relative
    representation error ``eps`` produces a residual of order ``eps`` in every
    family, so no single equation can price the physical solution out of the
    minimum.  Equilibrium and Gauss already use dominant-term scales and are
    untouched.

    ``max`` against the represented-quantity scale makes this monotone: a family
    that is already well conditioned keeps its current normalization, and no
    residual is ever given *more* weight than it has today.
    """
    length, height = scales.length, scales.height
    axial_strain = scales.u / length
    electric_field = scales.phi / height
    coupling = max(abs(float(materials.pze_E[0, 1])),
                   abs(float(materials.pze_E[1, 1])))
    return ConstitutiveScales(
        sigma_xx=max(scales.stress_x, float(materials.C11) * axial_strain),
        sigma_yy=max(scales.stress_y, float(materials.C12) * axial_strain),
        # The through-thickness slope of u is the bending rotation; it, not the
        # resulting shear traction, sets the size of the terms in the shear law.
        tau_xy=max(scales.shear, float(materials.G) * scales.u / height),
        d_x=max(scales.electric_displacement,
                float(materials.epsilon_1) * electric_field),
        d_y=max(scales.electric_displacement,
                float(materials.epsilon_2) * electric_field,
                coupling * axial_strain),
    )


def represented_term_scales(scales: CharacteristicScales, *,
                            strict_transverse: bool = False
                            ) -> ConstitutiveScales:
    """Historical normalization: divide by the represented quantity's scale."""
    return ConstitutiveScales(
        sigma_xx=scales.stress_x,
        sigma_yy=scales.stress_y if strict_transverse else scales.stress_x,
        tau_xy=scales.shear,
        d_x=scales.electric_displacement,
        d_y=scales.electric_displacement,
    )


def _material_references() -> tuple[float, float, float]:
    stiffness = float(materials.C11)
    coupling = max(abs(float(materials.pze_E[0, 1])),
                   abs(float(materials.pze_E[1, 1])))
    permittivity = max(float(materials.epsilon_1), float(materials.epsilon_2))
    return stiffness, coupling, permittivity


def indirect_scales(voltage: float = 100.0) -> CharacteristicScales:
    """Scales for the voltage-driven (converse/indirect) problem."""
    stiffness, coupling, permittivity = _material_references()
    phi = max(abs(float(voltage)), 1.0)
    electric_field = phi / HEIGHT
    strain = max(coupling * electric_field / stiffness, 1e-16)
    stress = stiffness * strain
    slenderness = HEIGHT / WIDTH
    if config.INDIRECT_SLENDER_TRANSVERSE_SCALING:
        shear = stress * slenderness
        stress_y = stress * slenderness**2
    else:
        shear = stress
        stress_y = stress
    d_scale = max(permittivity * electric_field, coupling * strain)
    return CharacteristicScales(
        length=WIDTH,
        height=HEIGHT,
        u=strain * WIDTH,
        v=strain * WIDTH**2 / HEIGHT,
        phi=phi,
        stress_x=stress,
        stress_y=stress_y,
        shear=shear,
        electric_displacement=d_scale,
        traction=stress,
    )


def direct_scales(force: float = REFERENCE_FORCE) -> CharacteristicScales:
    """Scales for a unit-depth cantilever loaded by a tip resultant.

    ``stress`` is the Euler--Bernoulli root bending scale ``6 F L/H^2``;
    ``traction`` is the actually prescribed right-edge traction ``F/H``.
    Keeping both prevents the small tip traction from disappearing when the
    boundary loss is normalized by the much larger root bending stress.
    """
    stiffness, coupling, permittivity = _material_references()
    force_magnitude = max(abs(float(force)), 1e-16)
    traction = force_magnitude / HEIGHT
    stress = 6.0 * force_magnitude * WIDTH / HEIGHT**2
    strain = stress / stiffness
    phi = max(coupling * strain * HEIGHT / permittivity, 1e-12)
    electric_field = phi / HEIGHT
    d_scale = max(coupling * strain, permittivity * electric_field)
    return CharacteristicScales(
        length=WIDTH,
        height=HEIGHT,
        u=strain * WIDTH,
        v=strain * WIDTH**2 / HEIGHT,
        phi=phi,
        stress_x=stress,
        # In a slender tip-loaded beam sigma_xx carries the root bending
        # scale, whereas transverse normal/shear stresses carry the applied
        # traction scale.  A common scale would underweight them by 6 L/H.
        stress_y=traction,
        shear=traction,
        electric_displacement=d_scale,
        traction=traction,
    )
