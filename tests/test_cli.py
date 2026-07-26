from __future__ import annotations

import json


from graph_looper.cli import main


def test_validate_bundled_graph(capsys):
    assert main(["validate", "reviewer-loop", "-v"]) == 0
    out = capsys.readouterr().out
    assert "✓ reviewer-loop is valid" in out
    assert "loops (" in out


def test_validate_reports_a_bad_graph(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\nnodes:\n  - id: a\n    type: input\nedges: []\n")
    assert main(["validate", str(path)]) == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_graph_name_lists_the_bundled_ones(capsys):
    assert main(["validate", "no-such-graph"]) == 2
    assert "Known graphs: draft-critique" in capsys.readouterr().err


def test_examples_lists_graphs(capsys):
    assert main(["examples"]) == 0
    assert "reviewer-loop" in capsys.readouterr().out


def test_viz_writes_mermaid(tmp_path, capsys):
    assert main(["viz", "draft-critique"]) == 0
    assert "flowchart LR" in capsys.readouterr().out

    target = tmp_path / "graph.md"
    assert main(["viz", "draft-critique", "-o", str(target)]) == 0
    assert "```mermaid" in target.read_text()


def test_run_dry_run_reaches_an_output(capsys):
    assert main(["run", "triage-router", "-i", "my card was charged twice", "--dry-run"]) == 0
    assert capsys.readouterr().out.strip()


def test_run_with_a_mock_script_and_trace(tmp_path, capsys):
    script = tmp_path / "mock.json"
    script.write_text(
        json.dumps(
            {
                "writer": ["draft one", "draft two"],
                "critic": ["needs a title", "Ship it."],
                "good_enough": [
                    {"choice": "revise", "reason": "no title"},
                    {"choice": "ship", "reason": "fine"},
                ],
            }
        )
    )
    trace = tmp_path / "trace.json"
    code = main(
        [
            "run",
            "draft-critique",
            "-i",
            "write a note",
            "--mock",
            str(script),
            "--trace",
            str(trace),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["output"] == "draft two"
    saved = json.loads(trace.read_text())
    assert any(e["kind"] == "edge" for e in saved["trace"])


def test_run_can_overlay_the_trace_on_a_diagram(tmp_path):
    target = tmp_path / "run.md"
    assert (
        main(
            ["run", "draft-critique", "-i", "x", "--dry-run", "--viz", str(target), "-q"]
        )
        == 0
    )
    assert "```mermaid" in target.read_text()


def test_viz_can_reload_a_saved_trace(tmp_path):
    trace = tmp_path / "trace.json"
    main(["run", "draft-critique", "-i", "x", "--dry-run", "--trace", str(trace), "-q"])
    out = tmp_path / "out.mmd"
    assert main(["viz", "draft-critique", "--trace", str(trace), "-o", str(out)]) == 0
    assert "stroke:#2f8f4e" in out.read_text()


def test_run_needs_a_task(capsys):
    assert main(["run", "draft-critique", "--dry-run"]) == 2
    assert "--input" in capsys.readouterr().err


def test_run_reports_failure_with_a_nonzero_exit(tmp_path, capsys):
    script = tmp_path / "mock.json"
    script.write_text(json.dumps({"good_enough": [{"choice": "nonsense", "reason": "x"}]}))
    assert main(["run", "draft-critique", "-i", "x", "--mock", str(script), "-q"]) == 1
    assert "not one of" in capsys.readouterr().err


def test_var_parsing_rejects_bare_values(capsys):
    assert main(["run", "draft-critique", "-i", "x", "--dry-run", "--var", "oops"]) == 2
    assert "key=value" in capsys.readouterr().err
