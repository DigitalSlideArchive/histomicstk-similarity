from pathlib import Path

from slicer_cli_web import CLIArgumentParser

import histomicstk_similarity.EmbeddingSimilarity.EmbeddingSimilarity


def test_basic():
    xml_path = Path(
        histomicstk_similarity.EmbeddingSimilarity.__file__).parent / 'EmbeddingSimilarity.xml'
    sample_path = Path(__file__).parent / 'sample_Easy1.jpeg'
    args = CLIArgumentParser(xml_path).parse_args([str(sample_path)])
    args.model = 'Test'
    assert histomicstk_similarity.EmbeddingSimilarity.EmbeddingSimilarity.main(args)
