# Long-context fixtures

`tale-of-two-cities.txt.gz` contains the complete original Project Gutenberg UTF-8 text of Charles Dickens's *A Tale of Two Cities* (ebook 98), including the Gutenberg header and license text. It is the exact raw input used in the recorded 2026-09-11 experiment.

Source: https://www.gutenberg.org/ebooks/98.txt.utf-8

Run `python3 scripts/prepare_book.py` from the repository root to extract it and produce the cleaned benchmark input. Raw/clean SHA-256 values are recorded in `measurements/2026-09-11/book-source.json`. Compression uses a zero gzip timestamp for reproducible bytes. Preserve the complete raw file and its license/header when redistributing this fixture.

The novel itself is public domain in the United States and many other jurisdictions. The Project Gutenberg license included in the original text sets out terms for use of its name and distributions. The cookbook's Apache license does not replace those terms.

## 512K validation corpus

The additional `moby-dick.txt.gz` (Herman Melville, ebook 2701) and `pride-and-prejudice.txt.gz` (Jane Austen, ebook 1342) contain complete downloaded Gutenberg texts, including their headers and license notices. Source URLs, raw byte counts and SHA256 hashes for all three novels are in [extended-context-sources.json](extended-context-sources.json). Compression uses a zero gzip timestamp.

`benchmarks/check_extended_context.py`, run inside the pinned serving image, verifies those source hashes and selects excerpts surrounding the complete *A Tale of Two Cities*. It counts the actual chat template to produce a 522,240-token prompt. The recorded response retrieves every code but swaps two in order; see the [context validation report](../../docs/CONTEXT.md). These texts have the same Gutenberg attribution/redistribution considerations described above.
