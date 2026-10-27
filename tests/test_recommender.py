import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import scipy.sparse as sp

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import app


class TestModelArtifacts:
    """Verifies that model artifacts exist, load correctly, and have valid dimensions."""

    def test_artifacts_exist(self):
        movies_csv = BASE_DIR / "model" / "movies.csv"
        vectors_npz = BASE_DIR / "model" / "vectors.npz"
        assert movies_csv.exists(), "model/movies.csv is missing!"
        assert vectors_npz.exists(), "model/vectors.npz is missing!"

    def test_load_artifacts(self):
        movies, vectors, missing = app.load_artifacts()
        assert not missing, f"Missing artifacts: {missing}"
        assert isinstance(movies, pd.DataFrame), "movies is not a DataFrame"
        assert sp.issparse(vectors), "vectors is not a sparse matrix"
        assert len(movies) == vectors.shape[0], "Row count mismatch between movies and vectors"
        assert vectors.shape[1] == 5000, "Feature dimension should be 5000"


class TestRecommenderEngine:
    """Verifies recommendation algorithm correctness and edge cases."""

    @pytest.fixture
    def artifacts(self):
        movies, vectors, _ = app.load_artifacts()
        return movies, vectors

    def test_recommend_returns_top_k(self, artifacts):
        movies, vectors = artifacts
        payload = app.recommend("The Dark Knight", movies, vectors, api_key=None, k=5)
        assert isinstance(payload, dict)
        assert "seed_poster" in payload
        assert "results" in payload
        assert len(payload["results"]) == 5

    def test_recommend_does_not_return_seed(self, artifacts):
        movies, vectors = artifacts
        seed = "Batman Begins"
        payload = app.recommend(seed, movies, vectors, api_key=None, k=5)
        recommended_titles = [r["title"] for r in payload["results"]]
        assert seed not in recommended_titles, "Recommender should never recommend the film back to itself"

    def test_recommend_invalid_movie(self, artifacts):
        movies, vectors = artifacts
        payload = app.recommend("A Nonexistent Movie 12345XYZ", movies, vectors, api_key=None, k=5)
        assert payload == []


class TestSecurityAndEnv:
    """Ensures no sensitive API keys are hardcoded and .gitignore protects secrets."""

    def test_no_hardcoded_keys_in_codebase(self):
        # Checks that no 32-character hex API key is hardcoded in app.py or build_model.py
        pattern = re.compile(r'["\'][a-f0-9]{32}["\']')
        py_files = [BASE_DIR / "app.py", BASE_DIR / "build_model.py"]
        for py_file in py_files:
            if py_file.exists():
                content = py_file.read_text(encoding="utf-8")
                matches = pattern.findall(content)
                assert not matches, f"Found potential hardcoded API key in {py_file.name}: {matches}"

    def test_gitignore_protects_env(self):
        gitignore = BASE_DIR / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text(encoding="utf-8")
        assert ".env" in content, ".env MUST be ignored in .gitignore"
        assert ".streamlit/secrets.toml" in content, "secrets.toml MUST be ignored in .gitignore"
