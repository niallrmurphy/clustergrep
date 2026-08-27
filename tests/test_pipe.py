"""`clustergrep ... | head` is the normal way to look at a big result, and it
closes the pipe as soon as head has what it wants. Reporting that as a
traceback would be noise: the user got exactly what they asked for."""

import subprocess
import sys

import pytest

CORPUS = "the prisoner escaped\na breakout on B wing\ntwo inmates fled\n" * 2000
THESAURUS = "escape\tbreakout\t0.25\nescape\tflee\t0.20\n"


@pytest.fixture
def files(tmp_path):
    (tmp_path / "t.tsv").write_text(THESAURUS)
    (tmp_path / "c.log").write_text(CORPUS)
    return tmp_path


def run_piped(files, *extra):
    """clustergrep | head -2, as a shell would run it."""
    producer = subprocess.Popen(
        [sys.executable, "-m", "clustergrep", "--color", "never",
         "-b", "thesaurus", "--thesaurus", str(files / "t.tsv"),
         "-t", "0.25", "escape", str(files / "c.log"), *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    head = subprocess.Popen(["head", "-2"], stdin=producer.stdout,
                            stdout=subprocess.PIPE, text=True)
    producer.stdout.close()
    out = head.communicate()[0]
    err = producer.stderr.read()
    producer.stderr.close()
    producer.wait(timeout=30)
    return producer.returncode, out, err


def test_a_closed_pipe_is_not_an_error_report(files):
    code, out, err = run_piped(files)
    assert "Traceback" not in err
    assert "BrokenPipeError" not in err
    assert len(out.splitlines()) == 2


def test_the_same_holds_for_excerpt_output(files):
    """The reported failure was on --excerpt specifically."""
    code, out, err = run_piped(files, "--excerpt")
    assert "Traceback" not in err
    assert "BrokenPipeError" not in err


def test_nothing_is_reported_on_the_way_out(files):
    """Python raises a second time while flushing stdout at exit unless the
    stream has been pointed somewhere harmless."""
    _, _, err = run_piped(files, "--excerpt")
    assert "Exception ignored" not in err


def test_the_exit_code_is_the_conventional_one(files):
    code, _, _ = run_piped(files)
    assert code in (0, 141), f"unexpected exit {code}"
