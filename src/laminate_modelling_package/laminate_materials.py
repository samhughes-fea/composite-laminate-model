"""Material cards and a repository-pattern material library."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional

from .laminate_hygrothermal import HygrothermalProperties
from .laminate_materials_3d import ElasticProperties3D, StrengthProperties3D


@dataclass(frozen=True)
class MaterialCard:
    """Complete lamina definition: elastic, strength and environmental properties."""

    name: str
    elastic: ElasticProperties3D
    strength: StrengthProperties3D
    hygrothermal: Optional[HygrothermalProperties] = None
    density: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("MaterialCard.name must be a non-empty string")
        if self.density < 0.0:
            raise ValueError("density must be non-negative")

    def with_elastic(self, elastic: ElasticProperties3D, suffix: str = "") -> "MaterialCard":
        """Return a copy with replaced elastic constants (used by damage models)."""
        name = f"{self.name}{suffix}" if suffix else self.name
        return replace(self, name=name, elastic=elastic)


class MaterialNotFoundError(KeyError):
    """Raised when a material name is not present in a repository."""


class DuplicateMaterialError(ValueError):
    """Raised when registering a name that already exists without overwrite."""


class MaterialRepository(ABC):
    """Abstract persistence boundary for material cards (Repository Pattern)."""

    @abstractmethod
    def register(self, card: MaterialCard, overwrite: bool = False) -> None:
        """Store a card under ``card.name``."""

    @abstractmethod
    def get(self, name: str) -> MaterialCard:
        """Retrieve a card by name or raise MaterialNotFoundError."""

    @abstractmethod
    def remove(self, name: str) -> None:
        """Delete a card by name or raise MaterialNotFoundError."""

    @abstractmethod
    def names(self) -> List[str]:
        """Return all registered names in registration order."""

    def contains(self, name: str) -> bool:
        return name in self.names()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.contains(name)


class MaterialLibrary(MaterialRepository):
    """In-memory material repository."""

    def __init__(self, cards: Optional[List[MaterialCard]] = None) -> None:
        self._cards: Dict[str, MaterialCard] = {}
        for card in cards or []:
            self.register(card)

    def register(self, card: MaterialCard, overwrite: bool = False) -> None:
        if not isinstance(card, MaterialCard):
            raise TypeError("MaterialLibrary only stores MaterialCard instances")
        if card.name in self._cards and not overwrite:
            raise DuplicateMaterialError(f"Material '{card.name}' is already registered")
        self._cards[card.name] = card

    def get(self, name: str) -> MaterialCard:
        try:
            return self._cards[name]
        except KeyError as exc:
            raise MaterialNotFoundError(
                f"Material '{name}' not found. Available: {', '.join(self._cards) or 'none'}"
            ) from exc

    def remove(self, name: str) -> None:
        if name not in self._cards:
            raise MaterialNotFoundError(f"Material '{name}' not found")
        del self._cards[name]

    def names(self) -> List[str]:
        return list(self._cards)

    def contains(self, name: str) -> bool:
        return name in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[MaterialCard]:
        return iter(list(self._cards.values()))


def _transversely_isotropic_card(
    name: str,
    E1: float,
    E2: float,
    v12: float,
    G12: float,
    v23: float,
    strengths: StrengthProperties3D,
    alpha: tuple,
    beta: tuple,
    density: float,
    description: str,
) -> MaterialCard:
    return MaterialCard(
        name=name,
        elastic=ElasticProperties3D.transversely_isotropic(E1, E2, v12, G12, v23),
        strength=strengths,
        hygrothermal=HygrothermalProperties(alpha1=alpha[0], alpha2=alpha[1], beta1=beta[0], beta2=beta[1]),
        density=density,
        description=description,
    )


def create_default_library() -> MaterialLibrary:
    """Library of representative unidirectional laminae in SI units (Pa, 1/K, kg/m^3).

    The values are typical handbook figures (Tsai, Daniel & Ishai, Herakovich)
    intended for examples and verification; use certified allowables for design.
    """
    MPa, GPa = 1.0e6, 1.0e9
    library = MaterialLibrary()
    library.register(
        _transversely_isotropic_card(
            "T300/5208",
            181 * GPa, 10.3 * GPa, 0.28, 7.17 * GPa, 0.40,
            StrengthProperties3D(1500 * MPa, 1500 * MPa, 40 * MPa, 246 * MPa, 68 * MPa, S23=40 * MPa),
            alpha=(0.02e-6, 22.5e-6), beta=(0.0, 0.6),
            density=1600.0,
            description="Carbon/epoxy (Tsai & Hahn)",
        )
    )
    library.register(
        _transversely_isotropic_card(
            "AS4/3501-6",
            142 * GPa, 10.3 * GPa, 0.27, 7.2 * GPa, 0.54,
            StrengthProperties3D(2280 * MPa, 1725 * MPa, 57 * MPa, 228 * MPa, 76 * MPa, S23=50 * MPa),
            alpha=(-0.9e-6, 27.0e-6), beta=(0.0, 0.2),
            density=1600.0,
            description="Carbon/epoxy (Daniel & Ishai)",
        )
    )
    library.register(
        _transversely_isotropic_card(
            "IM7/8552",
            171.4 * GPa, 9.08 * GPa, 0.32, 5.29 * GPa, 0.50,
            StrengthProperties3D(2326 * MPa, 1200 * MPa, 62.3 * MPa, 199.8 * MPa, 92.3 * MPa, S23=60 * MPa),
            alpha=(-0.1e-6, 31.0e-6), beta=(0.0, 0.3),
            density=1570.0,
            description="Intermediate-modulus carbon/epoxy",
        )
    )
    library.register(
        _transversely_isotropic_card(
            "E-glass/epoxy",
            38.6 * GPa, 8.27 * GPa, 0.26, 4.14 * GPa, 0.35,
            StrengthProperties3D(1062 * MPa, 610 * MPa, 31 * MPa, 118 * MPa, 72 * MPa, S23=40 * MPa),
            alpha=(8.6e-6, 22.1e-6), beta=(0.0, 0.6),
            density=1800.0,
            description="Glass/epoxy (Tsai & Hahn)",
        )
    )
    library.register(
        MaterialCard(
            name="Al-6061-T6",
            elastic=ElasticProperties3D.isotropic(68.9 * GPa, 0.33),
            strength=StrengthProperties3D(276 * MPa, 276 * MPa, 276 * MPa, 276 * MPa, 159 * MPa),
            hygrothermal=HygrothermalProperties(alpha1=23.6e-6, alpha2=23.6e-6),
            density=2700.0,
            description="Isotropic aluminium alloy (yield used as strength)",
        )
    )
    return library
