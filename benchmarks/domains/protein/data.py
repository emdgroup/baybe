"""Lazy loading of protein benchmark case data, cached locally after first fetch."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import Self

from benchmarks.data.utils import DATA_PATH
from benchmarks.persistence import CachedFileLoader, S3ObjectRetrieval

_PROTEIN_DATA_PATH = DATA_PATH / "protein"


@define(frozen=True)
class ProteinCaseMetadata:
    """Metadata describing the available data files of a protein benchmark case."""

    score_skewness: float = field(validator=instance_of(float))
    """The skewness of the fitness score distribution."""

    mutations_file: str = field(validator=instance_of(str))
    """The name of the file containing the mutation data."""

    embeddings: dict[str, str] = field(validator=instance_of(dict))
    """Mapping from embedding model name to the name of its embedding file."""

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Create metadata from a JSON manifest file.

        Args:
            path: The path to the metadata JSON file.

        Returns:
            The parsed metadata.
        """
        with open(path) as file:
            content = json.load(file)
        return cls(
            score_skewness=content["score_skewness"],
            mutations_file=content["mutations_file"],
            embeddings=content["embeddings"],
        )


@define(slots=False)
class ProteinCaseLoader:
    """Lazily loads and locally caches protein benchmark case data."""

    case_name: str = field(validator=instance_of(str))
    """The name of the protein benchmark case, e.g. ``"jones"``."""

    _loader: CachedFileLoader = field(
        factory=lambda: CachedFileLoader(fetcher=S3ObjectRetrieval())
    )
    """The generic file loader used to fetch and cache case files."""

    @property
    def _case_path(self) -> Path:
        """The local cache directory for this case."""
        return _PROTEIN_DATA_PATH / self.case_name

    @cached_property
    def metadata(self) -> ProteinCaseMetadata:
        """The metadata for this case."""
        return ProteinCaseMetadata.from_json(self._get_file("metadata.json"))

    def _get_file(self, file_name: str) -> Path:
        """Return the local cache path to a case file, downloading it if needed.

        Args:
            file_name: The name of the file within the case folder.

        Returns:
            The local path to the file.
        """
        key = f"protein/{self.case_name}/{file_name}"
        return self._loader.get_file(key, self._case_path / file_name)

    def _validate_mutation_key(self, data: pd.DataFrame, path: Path) -> None:
        """Validate that a table has a unique mutation-code (e.g. ``"A32G"``) key.

        Args:
            data: The table to validate.
            path: The file ``data`` was read from (for error messages only).

        Raises:
            ValueError: If the table has no ``mutation`` column, or contains
                duplicate mutation codes.
        """
        if "mutation" not in data.columns:
            raise ValueError(
                f"File '{path}' has no 'mutation' column. Protein benchmark files "
                f"must include an explicit mutation-code column so they can be "
                f"safely aligned with each other."
            )
        if not data["mutation"].is_unique:
            raise ValueError(f"File '{path}' contains duplicate mutation codes.")

    def get_aligned_data(self, model_name: str) -> tuple[pd.DataFrame, list[str]]:
        """Load mutation records joined with their embeddings for a given model.

        Mutations and embeddings are always needed together: the embeddings provide
        the computational representation, the mutation records provide the score to
        look up. This loads both files in full and aligns them via an explicit
        label-based join on the mutation code, instead of leaving that to callers.

        Args:
            model_name: The name of the embedding model, e.g. ``"ProtT5XL"``.

        Returns:
            A tuple of the mutation table merged with the embeddings (``mutation``
            remains a regular column), and the list of embedding feature columns.

        Raises:
            KeyError: If no embeddings are available for the given model.
            ValueError: If the mutation or embeddings file has no ``mutation``
                column, contains duplicate mutation codes, or if the two files do
                not cover the exact same set of mutation codes.
        """
        embeddings_files = self.metadata.embeddings
        if model_name not in embeddings_files:
            raise KeyError(
                f"No embeddings available for model '{model_name}' in case "
                f"'{self.case_name}'. Available models: {list(embeddings_files)}."
            )

        mutations_path = self._get_file(self.metadata.mutations_file)
        embeddings_path = self._get_file(embeddings_files[model_name])
        mutations = pd.read_table(mutations_path)
        embeddings = pd.read_parquet(embeddings_path)
        self._validate_mutation_key(mutations, mutations_path)
        self._validate_mutation_key(embeddings, embeddings_path)
        embedding_columns = [c for c in embeddings.columns if c != "mutation"]

        data = mutations.merge(
            embeddings, on="mutation", how="inner", validate="one_to_one"
        )
        if len(data) != len(mutations):
            missing = sorted(set(mutations["mutation"]) - set(data["mutation"]))
            raise ValueError(
                f"{len(missing)} mutation(s) in case '{self.case_name}' have no "
                f"matching embedding: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}. Mutation and embedding files "
                f"must cover the exact same set of mutation codes."
            )
        return data, embedding_columns
