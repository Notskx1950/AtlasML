from app.utils.hashing import file_sha256


def test_file_sha256_same_file_same_hash(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"x": 1}\n{"x": 2}\n')

    h1 = file_sha256(path)
    h2 = file_sha256(path)

    assert h1 == h2
    assert len(h1) == 64


def test_file_sha256_changes_when_file_changes(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"x": 1}\n')

    h1 = file_sha256(path)

    path.write_text('{"x": 1}\n{"x": 2}\n')

    h2 = file_sha256(path)

    assert h1 != h2