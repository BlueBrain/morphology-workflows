"""Tests for the morphology_workflows.cli module."""
import re

from morphology_workflows.tasks import cli


class TestCLI:
    """Test the CLI of the morphology-workflows package."""

    def test_help(self, capsys):
        """Test the --help argument."""
        try:
            cli.main(arguments=["--help"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert (
            re.match(
                r"usage: \S+ .*Run the workflow\n\npositional arguments:\s*"
                r"{Fetch,Placeholders,Curate,Annotate,Repair}\s*Possible workflows.*",
                captured.out,
                flags=re.DOTALL,
            )
            is not None
        )

    def test_dependency_graph(self, tmpdir, data_dir):
        """Test the --help argument."""
        output_path = (data_dir / "neuromorpho_config_download.json").resolve()
        cli.main(
            arguments=[
                "--create-dependency-graph",
                str(tmpdir / "dependency_graph.png"),
                "Fetch",
                "--source",
                "NeuroMorpho",
                "--config-file",
                str(output_path),
            ]
        )

        assert output_path.exists()
