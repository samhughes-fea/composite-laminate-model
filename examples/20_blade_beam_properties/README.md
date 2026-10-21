# 20 · Blade beam properties and bend-twist tailoring

Aeroelastic codes model a blade or wing as a beam. They need, at each station,
the section's stiffness matrix, its shear and tension centres and its mass per
length. This example extracts all of them from a shell model of a blade segment,
then turns the spar cap fibres off-axis to couple flap bending to twist, the
mechanism behind load-alleviating, bend-twist-coupled blades.

![The section and its centres, stiffness against cap angle, and the bend-twist coupling coefficient](figures/blade_beam_properties.png)

## The model

| Item | Value |
|---|---|
| Section | NACA 0024, 1 m chord, pitch axis at 30% chord |
| Segment | 6 m prismatic, Quad4 FSDT, 20 elements along the span |
| Skins | E-glass/epoxy [0/45/−45]s, 0.5 mm plies (3 mm) |
| Spar caps | 16 × 0.5 mm T300/5208 at 18–48% chord, at +θ on the suction side and −θ on the pressure side |
| Webs | E-glass/epoxy [±45]s at 18% and 48% chord |
| Sweep | θ = 0° to 30° in 5° steps |

```bash
python examples/20_blade_beam_properties/blade_beam_properties.py
```

## Walkthrough

1. **Building the section.** Two stations with the same airfoil make a prismatic
   segment. Regions and webs are given by chord fraction; laminates are given by
   region name, with `_ss` and `_ps` suffixes when the suction and pressure sides
   differ:

   ```python
   REGIONS = [SkinRegion("le_panel", 0.0, 0.18, 4), SkinRegion("spar_cap", 0.18, 0.48, 6),
              SkinRegion("te_panel", 0.48, 1.0, 8)]
   WEBS = [ShearWeb("web_fwd", 0.18, 4), ShearWeb("web_aft", 0.48, 4)]
   blade = BladeBuilder(stations, REGIONS, WEBS, n_span=20).build(
       {"le_panel": skin, "te_panel": skin, "web_fwd": web, "web_aft": web,
        "spar_cap_ss": cap(+theta), "spar_cap_ps": cap(-theta)}, FSDTModel())
   ```

2. **Beam properties.** `beam_section_properties(blade)` clamps the root, closes
   the tip with a rigid end rib (an RBE2 on the translations) and applies unit axial
   force, two bending moments and a torque there. Two distributing links (RBE3) in
   the middle fifth of the span, clear of the root and tip, measure the mean
   section strains and curvatures. Inverting the 4 × 4 compliance gives the
   stiffness matrix about the pitch axis, couplings included:

   ```python
   props = beam_section_properties(blade)
   props.stiffness            # [N, Mx, My, Mz] against [axial strain, flap, edge curvature, twist rate]
   props.EI_x, props.GJ, props.shear_centre, props.tension_centre, props.mass_per_length
   props.bend_twist_coupling  # S14 / sqrt(S11 S44) for flap, and for edge
   ```

   Two tip shear loads then locate the shear centre (the twist each causes, less
   the twist its bending moment causes through the coupling terms).

3. **Checks.** On a thin circular tube, which has closed-form properties, the
   extracted EA, EI and GJ are exact with curved Quad8 shells and within about 1%
   with Quad4 (`tests/test_blade.py`).

## Reading the figure

- **Left:** the section with its centres. The tension centre (51 mm aft of the
  pitch axis) sits where the carbon caps put it; the shear centre is just aft of
  the pitch axis (17 mm) and the mass centre further aft (130 mm), pulled back by
  the long trailing-edge skin. The section weighs 17.5 kg/m.
- **Middle:** turning the cap fibres costs axial and flap stiffness fast (EI flap
  falls to 0.43 of its 0° value at 20°) and gains torsional stiffness (GJ rises by
  about 40%).
- **Right:** the bend-twist coupling coefficient. With +θ on one cap and −θ on the
  other, flap bending shears the two caps in opposite directions, and the closed
  section turns that into twist. The coupling peaks at about −0.49 near 20°, the
  practical upper bound for carbon caps, and falls beyond as the caps lose
  stiffness. Equal angles on both caps would couple extension to twist instead.

## Things to try

- Set the same angle on both caps (`cap(theta)` on both sides) and read the
  extension-twist term `stiffness[0, 3]`.
- Move the webs or widen the caps and watch the shear centre move.
- Use `element="quad8_curved"` in `BladeBuilder` for curved skins (slower,
  slightly more accurate torsion).
- Build a prismatic segment for each station of a tapered blade and tabulate the
  properties along the span, as input for an aeroelastic code.
