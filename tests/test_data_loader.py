"""Unit tests for the document loaders / text extractors."""

from __future__ import annotations

from sportsee_rag.ingestion.data_loader import (
    extract_text_from_csv,
    extract_text_from_pdf,
    extract_text_from_txt,
    load_and_parse_files,
)


def test_extract_txt(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")
    assert extract_text_from_txt(str(path)) == "hello world"


def test_extract_csv_flattens_to_text(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,points\nAnt,30\n", encoding="utf-8")
    out = extract_text_from_csv(str(path))
    assert out is not None
    assert "name" in out and "points" in out and "Ant" in out


def test_load_and_parse_skips_unsupported(tmp_path) -> None:
    (tmp_path / "keep.txt").write_text("keep me", encoding="utf-8")
    (tmp_path / "drop.xyz").write_text("ignore me", encoding="utf-8")
    docs = load_and_parse_files(tmp_path)
    assert len(docs) == 1
    assert docs[0]["page_content"] == "keep me"
    assert docs[0]["metadata"]["filename"] == "keep.txt"
    assert docs[0]["metadata"]["source"] == "keep.txt"


def test_load_and_parse_missing_dir_returns_empty() -> None:
    assert load_and_parse_files("does/not/exist") == []


def test_extract_pdf_missing_file_returns_none(monkeypatch) -> None:
    # Avoid spinning up EasyOCR: stub the lazy reader to "unavailable".
    monkeypatch.setattr(
        "sportsee_rag.ingestion.data_loader._get_ocr_reader", lambda: None
    )
    assert extract_text_from_pdf("nope.pdf") is None
